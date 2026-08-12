"""
Test the Task container machinery: slicing views, mixtures, and the
HubDataset parquet wrapper and resilient Hub parquet loading.

python -m pytest tests/test_tasks.py -v
"""

import io
import json
import urllib.error

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tasks.common as common
from tasks.common import Task, TaskMixture, HubDataset, load_hub_dataset, render_mc


class ToyTask(Task):
    """A trivial task: example i is just {'i': i, 'tag': tag}."""

    def __init__(self, n=10, tag="a", **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.tag = tag

    def num_examples(self):
        return self.n

    def get_example(self, index):
        return {"i": index, "tag": self.tag}


def test_task_full():
    task = ToyTask(n=10)
    assert len(task) == 10
    assert task[0] == {"i": 0, "tag": "a"}
    assert task[9] == {"i": 9, "tag": "a"}


def test_task_slicing():
    # a view of [5, 10) has 5 examples and maps logical to physical indices
    task = ToyTask(n=10, start=5, stop=10)
    assert len(task) == 5
    assert task[0]["i"] == 5
    # step slicing uses ceil division for the length
    task = ToyTask(n=10, start=0, stop=10, step=3) # 0, 3, 6, 9
    assert len(task) == 4
    assert [task[i]["i"] for i in range(4)] == [0, 3, 6, 9]


def test_mixture_covers_all_examples_deterministically():
    mixture = TaskMixture([ToyTask(n=3, tag="a"), ToyTask(n=5, tag="b")])
    assert len(mixture) == 8
    examples = [mixture[i] for i in range(8)]
    # every example appears exactly once
    keys = sorted((ex["tag"], ex["i"]) for ex in examples)
    assert keys == [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1), ("b", 2), ("b", 3), ("b", 4)]
    # the shuffle is deterministic: a second instance yields the same order
    mixture2 = TaskMixture([ToyTask(n=3, tag="a"), ToyTask(n=5, tag="b")])
    assert examples == [mixture2[i] for i in range(8)]
    # and the tasks are actually interleaved, not concatenated
    assert [ex["tag"] for ex in examples] != ["a"] * 3 + ["b"] * 5


def test_mixture_oversampling():
    # passing a task twice doubles its examples
    mixture = TaskMixture([ToyTask(n=3), ToyTask(n=3)])
    assert len(mixture) == 6


def test_hub_dataset_rows():
    table = pa.table({"x": list(range(100)), "y": [str(i) for i in range(100)]})
    ds = HubDataset(table)
    assert len(ds) == 100
    assert ds[7] == {"x": 7, "y": "7"}


def test_hub_dataset_shuffle_matches_numpy():
    # the shuffle must reproduce datasets.Dataset.shuffle(seed) exactly,
    # which is a np.random.default_rng(seed) permutation
    table = pa.table({"x": list(range(100))})
    ds = HubDataset(table).shuffle(seed=42)
    perm = np.random.default_rng(42).permutation(100)
    assert [ds[i]["x"] for i in range(100)] == [int(p) for p in perm]
    # shuffling returns a view; the original order is untouched
    assert HubDataset(table)[0] == {"x": 0}


def _parquet_bytes(table):
    output = pa.BufferOutputStream()
    pq.write_table(table, output)
    return output.getvalue().to_pybytes()


def test_load_hub_dataset_falls_back_after_transient_400(tmp_path, monkeypatch):
    table_bytes = _parquet_bytes(pa.table({"x": [1, 2], "answer": ["a", "b"]}))
    calls = []

    def fake_urlopen(request, timeout):
        url = request.full_url
        calls.append(url)
        if "datasets-server.huggingface.co" in url:
            raise urllib.error.HTTPError(url, 400, "temporary", {}, None)
        if url.endswith("/parquet/8k/qa1"):
            return io.BytesIO(json.dumps(["https://example.test/shard.parquet"]).encode())
        if url == "https://example.test/shard.parquet":
            return io.BytesIO(table_bytes)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(common, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(common.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(common.time, "sleep", lambda _seconds: None)

    dataset = load_hub_dataset("RMT-team/babilong-train-5k-samples", "8k", "qa1")
    assert len(dataset) == 2
    assert dataset[1] == {"x": 2, "answer": "b"}
    assert "datasets-server.huggingface.co" in calls[0]
    assert any(url.endswith("/parquet/8k/qa1") for url in calls)


def test_load_hub_dataset_filters_datasets_server_response(tmp_path, monkeypatch):
    table_bytes = _parquet_bytes(pa.table({"x": [7]}))
    listing = {
        "parquet_files": [
            {"config": "8k", "split": "qa2", "url": "https://example.test/wrong.parquet"},
            {"config": "8k", "split": "qa1", "url": "https://example.test/right.parquet"},
        ]
    }
    download_attempts = 0

    def fake_urlopen(request, timeout):
        nonlocal download_attempts
        url = request.full_url
        if "datasets-server.huggingface.co" in url:
            return io.BytesIO(json.dumps(listing).encode())
        if url == "https://example.test/right.parquet":
            download_attempts += 1
            if download_attempts == 1:
                raise urllib.error.HTTPError(url, 400, "temporary", {}, None)
            return io.BytesIO(table_bytes)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(common, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(common.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(common.time, "sleep", lambda _seconds: None)

    dataset = load_hub_dataset("RMT-team/babilong-train-5k-samples", "8k", "qa1")
    assert dataset[0] == {"x": 7}
    assert download_attempts == 2

    # A second load uses the validated local manifest and makes no network calls.
    monkeypatch.setattr(common.urllib.request, "urlopen", lambda *_args, **_kwargs: 1 / 0)
    assert load_hub_dataset("RMT-team/babilong-train-5k-samples", "8k", "qa1")[0] == {"x": 7}


def test_render_mc_letter_binding():
    query = render_mc("What is 1+1?", ("A", "B"), ("1", "2"))
    # the letter must directly follow '=' with no whitespace, so that the
    # prompt token for "A" matches the assistant's bare "A" response token
    assert "=A\n" in query and "=B\n" in query
