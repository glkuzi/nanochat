"""
Base class for all Tasks.
A Task is basically a dataset of conversations, together with some
metadata and often also evaluation criteria.
Example tasks: MMLU, ARC-Easy, ARC-Challenge, GSM8K, HumanEval, SmolTalk.
"""

import os
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from nanochat.common import get_base_dir

_HUB_REQUEST_TIMEOUT = 60
_HUB_MAX_ATTEMPTS = 5
_HUB_RETRYABLE_HTTP_CODES = {400, 403, 408, 409, 425, 429, 500, 502, 503, 504}
_HUB_USER_AGENT = "nanochat/1.0"


def _urlopen(url):
    request = urllib.request.Request(url, headers={"User-Agent": _HUB_USER_AGENT})
    return urllib.request.urlopen(request, timeout=_HUB_REQUEST_TIMEOUT)


def _is_retryable_download_error(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in _HUB_RETRYABLE_HTTP_CODES
    return isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError, OSError, pa.ArrowInvalid))


def _retry_delay(attempt):
    # Short exponential backoff. In particular, HF occasionally returns a transient
    # 400 while its parquet service is resolving or refreshing an export.
    time.sleep(min(2 ** attempt, 8))


def _read_json(url):
    with _urlopen(url) as response:
        return json.loads(response.read())


def _list_parquet_shards(repo_id, subset, split):
    """List one Hub dataset cell, preferring stable resolve URLs from datasets-server."""
    query = urllib.parse.urlencode({"dataset": repo_id, "config": subset})
    datasets_server_url = f"https://datasets-server.huggingface.co/parquet?{query}"
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_subset = urllib.parse.quote(subset, safe="")
    quoted_split = urllib.parse.quote(split, safe="")
    hub_url = f"https://huggingface.co/api/datasets/{quoted_repo}/parquet/{quoted_subset}/{quoted_split}"

    def parse_datasets_server(payload):
        if not isinstance(payload, dict):
            return []
        return [
            item["url"] for item in payload.get("parquet_files", [])
            if item.get("config") == subset and item.get("split") == split
        ]

    errors = []
    for attempt in range(_HUB_MAX_ATTEMPTS):
        # The datasets-server response may contain every split in a config even when
        # a split query parameter is supplied, so always filter it ourselves.
        for endpoint, parser in (
            (datasets_server_url, parse_datasets_server),
            (hub_url, lambda payload: payload if isinstance(payload, list) else []),
        ):
            try:
                shard_urls = parser(_read_json(endpoint))
                if shard_urls:
                    return shard_urls
                errors.append(f"{endpoint}: no parquet files for config={subset!r}, split={split!r}")
            except Exception as error:
                errors.append(f"{endpoint}: {error}")
        if attempt + 1 < _HUB_MAX_ATTEMPTS:
            _retry_delay(attempt)

    details = "; ".join(errors[-4:])
    raise RuntimeError(
        f"Unable to list parquet shards for {repo_id}/{subset}/{split} after "
        f"{_HUB_MAX_ATTEMPTS} attempts. Last errors: {details}"
    )


def _valid_parquet(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        pq.ParquetFile(path)
    except (OSError, pa.ArrowInvalid):
        return False
    return True


def _download_parquet(shard_url, destination):
    """Stream a shard to an atomic temporary file and retry transient failures."""
    partial_path = destination + ".partial"
    last_error = None
    for attempt in range(_HUB_MAX_ATTEMPTS):
        try:
            with _urlopen(shard_url) as response, open(partial_path, "wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            # Do not publish an HTML error body or a truncated response as a shard.
            if not _valid_parquet(partial_path):
                raise pa.ArrowInvalid(f"Downloaded file is not valid parquet: {shard_url}")
            os.replace(partial_path, destination)
            return
        except Exception as error:
            if not _is_retryable_download_error(error):
                raise
            last_error = error
            if attempt + 1 < _HUB_MAX_ATTEMPTS:
                _retry_delay(attempt)
    raise RuntimeError(
        f"Unable to download parquet shard {shard_url} after {_HUB_MAX_ATTEMPTS} attempts"
    ) from last_error


def _read_valid_manifest(manifest_path, shards_dir):
    try:
        with open(manifest_path, "r") as f:
            filenames = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(filenames, list) or not filenames:
        return None
    if not all(isinstance(name, str) and os.path.basename(name) == name for name in filenames):
        return None
    if not all(_valid_parquet(os.path.join(shards_dir, name)) for name in filenames):
        return None
    return filenames


class HubDataset:
    """
    Minimal stand-in for a HuggingFace datasets Dataset: wraps a pyarrow
    Table and offers lazy row access and a seeded shuffle.
    """

    def __init__(self, table, permutation=None):
        self.table = table
        self.permutation = permutation

    def __len__(self):
        return self.table.num_rows

    def shuffle(self, seed):
        # matches datasets.Dataset.shuffle(seed=seed) exactly, row order comes out identical
        permutation = np.random.default_rng(seed).permutation(len(self))
        return HubDataset(self.table, permutation)

    def __getitem__(self, index):
        physical_index = index if self.permutation is None else int(self.permutation[index])
        row = {column: self.table[column][physical_index].as_py() for column in self.table.column_names}
        return row


def load_hub_dataset(repo_id, subset="default", split="train"):
    """
    Minimal stand-in for HuggingFace datasets.load_dataset(repo_id, subset, split=split).
    Every dataset on the hub has an auto-generated parquet export. We list the parquet
    shards via the hub API, download them (once) into the local cache directory, and
    read them with pyarrow. Under torchrun, only one rank downloads, the others wait.
    """
    base_dir = get_base_dir()
    slug = repo_id.replace("/", "--")
    shards_dir = os.path.join(base_dir, "task_data", slug, subset, split)
    # The manifest is written last. Validate it as well, so a corrupt cache repairs
    # itself rather than failing every subsequent training run.
    manifest_path = os.path.join(shards_dir, "manifest.json")
    filenames = _read_valid_manifest(manifest_path, shards_dir)
    if filenames is None:
        os.makedirs(shards_dir, exist_ok=True)
        with FileLock(manifest_path + ".lock"):
            # only a single rank acquires the lock and downloads, the others block
            # here and then skip the download because they recheck the manifest
            filenames = _read_valid_manifest(manifest_path, shards_dir)
            if filenames is None:
                shard_urls = _list_parquet_shards(repo_id, subset, split)
                filenames = []
                for shard_index, shard_url in enumerate(shard_urls):
                    filename = f"{shard_index:05d}.parquet"
                    shard_path = os.path.join(shards_dir, filename)
                    if not _valid_parquet(shard_path):
                        print(f"Downloading {shard_url} ...")
                        _download_parquet(shard_url, shard_path)
                    filenames.append(filename)
                partial_manifest = manifest_path + ".partial"
                with open(partial_manifest, "w") as f:
                    json.dump(filenames, f)
                os.replace(partial_manifest, manifest_path)
    shard_paths = [os.path.join(shards_dir, filename) for filename in filenames]
    tables = [pq.read_table(path) for path in shard_paths]
    table = pa.concat_tables(tables)
    return HubDataset(table)


class Task:
    """
    Base class of a Task. Allows for lightweight slicing of the underlying dataset.
    """

    def __init__(self, start=0, stop=None, step=1):
        # allows a lightweight logical view over a dataset
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, f"Stop should be greater than or equal to start, got {stop} and {start}"
        assert step >= 1, f"Step must be strictly positive, got {step}"
        self.start = start
        self.stop = stop # could be None here
        self.step = step

    @property
    def eval_type(self):
        # one of 'generative' | 'categorical'
        raise NotImplementedError

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def __len__(self):
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step # ceil_div(span, step)
        assert num >= 0, f"Negative number of examples???: {num}" # prevent footguns
        return num

    def __getitem__(self, index: int):
        assert isinstance(index, int), f"Index must be an integer, got {type(index)}"
        physical_index = self.start + index * self.step
        conversation = self.get_example(physical_index)
        return conversation

    def evaluate(self, problem, completion):
        raise NotImplementedError


class TaskMixture(Task):
    """
    For SFT Training it becomes useful to train on a mixture of datasets.
    Fun trick: if you wish to oversample any task, just pass it in multiple times in the list.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        # tasks is a list of Task objects
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)
        # Build list of all (task_idx, local_idx) pairs
        self.index_map = []
        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))
        # Deterministically shuffle to mix tasks throughout training
        rng = random.Random(42)
        rng.shuffle(self.index_map)
        # Note: this is not the most elegant or best solution, but it's ok for now

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        """
        Access conversations according to a deterministic shuffle of all examples.
        This ensures tasks are mixed throughout training, regardless of dataset size.
        """
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for mixture with {self.num_conversations} conversations"
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


class TaskSequence(Task):
    """
    For SFT Training sometimes we want to sequentially train on a list of tasks.
    This is useful for cases that require a training curriculum.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for sequence with {self.num_conversations} conversations"
        for task_idx, task_length in enumerate(self.lengths):
            if index < task_length:
                return self.tasks[task_idx][index]
            index -= task_length


def render_mc(question, letters, choices):
    """
    The common multiple choice rendering format we will use.

    Note two important design decisions:
    1)
    Bigger models don't care as much, but smaller models prefer to have
    the letter *after* the choice, which results in better binding.
    2)
    There is no whitespace between the delimiter (=) and the letter.
    This is actually critical because the tokenizer has different token ids
    for " A" vs. "A". The assistant responses will be just the letter itself,
    i.e. "A", so it is important that here in the prompt it is the exact same
    token, i.e. "A" with no whitespace before it. Again, bigger models don't care
    about this too much, but smaller models do care about some of these details.
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query


if __name__ == "__main__":
    # very lightweight test of slicing
    from tasks.mmlu import MMLU

    ds = MMLU(subset="all", split="auxiliary_train")
    print("Length of MMLU: ", len(ds))
    ex = ds[5]
    print("5th example: ", ex)

    ds = MMLU(subset="all", split="auxiliary_train", start=5, stop=10)
    print("Length of sliced MMLU[5:10]: ", len(ds))
    print("0th example of sliced MMLU: ", ds[0])

    print("They match: ", ex == ds[0])
