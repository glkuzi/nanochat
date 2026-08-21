"""On-the-fly BABILong training data built from bAbI facts and PG-19 noise.

This is the nanochat adaptation of the original BABILong training pipeline in
``babilong_utils.py``. It is intentionally separate from the pre-generated
Hugging Face training stream and is only used for train/validation loss.
"""

from __future__ import annotations

import glob
import os
import random
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset

from nanochat.common import get_base_dir


DEFAULT_BABILONG_BABI_PATH = (
    "/home/jovyan/gkuzmin/rmt_it/babilong/data/tasks_1-20_v1-2/en-10k"
)
DEFAULT_BABILONG_NOISE_DATASET = "pg19"
BABILONG_QA_MARGIN = 20


@dataclass(frozen=True)
class BabiSample:
    """One single-question bAbI example, including all preceding facts."""

    facts: tuple[str, ...]
    question: str
    answer: str
    references: tuple[str, ...]


def load_babi_task_file(
    path: str,
    max_n_facts: int | None = None,
    max_n_samples: int | None = None,
) -> list[BabiSample]:
    """Parse the original bAbI text format into single-question examples.

    A story containing several questions produces one example per question.
    Earlier question lines are omitted from later contexts, matching
    ``get_dataset_df`` in the reference implementation.
    """
    samples: list[BabiSample] = []
    facts: list[tuple[int, str]] = []
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            phrase, text = line.split(" ", maxsplit=1)
            phrase_num = int(phrase)
            if phrase_num == 1:
                facts = []
            if "\t" not in text:
                facts.append((phrase_num, text))
                continue

            fields = text.split("\t")
            if len(fields) < 2:
                raise ValueError(f"Malformed bAbI question in {path}: {line!r}")
            question, answer = fields[:2]
            reference_ids = {
                int(value)
                for value in re.split(r"\s+", fields[2].strip())
                if value
            } if len(fields) > 2 else set()
            # The reference code applies max_n_facts to the slice containing
            # facts plus its question row, so retain that exact threshold.
            if max_n_facts is not None and len(facts) + 1 > max_n_facts:
                continue
            samples.append(BabiSample(
                facts=tuple(value for _, value in facts),
                question=question,
                answer=answer,
                references=tuple(value for number, value in facts if number in reference_ids),
            ))
            if max_n_samples is not None and len(samples) >= max_n_samples:
                break
    if not samples:
        raise ValueError(f"No usable bAbI examples found in {path}")
    return samples


def resolve_babi_task_path(babi_path: str, task: str, split: str) -> str:
    """Resolve ``qaN`` to the corresponding descriptive bAbI filename."""
    if split not in {"train", "test"}:
        raise ValueError("bAbI split must be 'train' or 'test'")
    pattern = os.path.join(os.path.expanduser(babi_path), f"{task}_*_{split}.txt")
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one bAbI file for {task} {split} at {pattern!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def generated_sample_sizes(
    max_sequence_length: int,
    segment_size: int,
    vary_lengths: bool,
) -> tuple[int, ...]:
    """Return the reference pipeline's discrete segment-multiple budgets."""
    if max_sequence_length <= 0 or segment_size <= 0:
        raise ValueError("max_sequence_length and segment_size must be positive")
    if segment_size > max_sequence_length:
        raise ValueError("segment_size must not exceed max_sequence_length")
    if not vary_lengths:
        return (max_sequence_length,)
    sizes = list(range(segment_size, max_sequence_length, segment_size))
    sizes.append(max_sequence_length)
    return tuple(sizes)


def load_pg19_noise_split(
    dataset_name: str,
    split: str,
    config_name: str | None = None,
    cache_dir: str | None = None,
):
    """Load one PG-19 split through ``datasets`` and its local disk cache."""
    kwargs = {
        "split": split,
        #"cache_dir": cache_dir,
        "keep_in_memory": False,
        "revision": "refs/convert/parquet",
    }
    if config_name:
        kwargs["name"] = config_name
    return load_dataset(dataset_name, **kwargs)


def _noise_text(row: dict) -> str:
    if "text" in row:
        return str(row["text"])
    if "page" in row:
        return str(row["page"])
    raise KeyError("Noise dataset rows must contain a 'text' or 'page' column")


class SentenceSampler:
    """Sample exact token budgets from sentence-tokenized PG-19 documents."""

    def __init__(
        self,
        dataset,
        tokenizer,
        min_sentence_len: int | None = 10,
        max_sentence_len: int | None = None,
        shuffle: bool = False,
        random_seed: int = 42,
        sentence_splitter: Callable[[str], Sequence[str]] | None = None,
    ) -> None:
        self.sample_ind = 0
        self.dataset = dataset
        self.sentences: list[str] = []
        self.tokenizer = tokenizer
        self.min_sentence_len = min_sentence_len
        self.max_sentence_len = max_sentence_len
        self.shuffle = shuffle
        self.gen = np.random.default_rng(seed=random_seed)
        if sentence_splitter is None:
            try:
                from nltk import PunktSentenceTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "The generated BABILong backend requires nltk; install the "
                    "nanochat project dependencies before training"
                ) from exc
            sentence_splitter = PunktSentenceTokenizer().tokenize
        self.sentence_splitter = sentence_splitter

    def _length_is_ok(self, tokenized: Sequence[int]) -> bool:
        if self.max_sentence_len is not None and len(tokenized) > self.max_sentence_len:
            return False
        if self.min_sentence_len is not None and len(tokenized) < self.min_sentence_len:
            return False
        return True

    def _next_sample(self) -> str:
        if not len(self.dataset):
            raise ValueError("Noise dataset is empty")
        if self.shuffle:
            sample_ind = int(self.gen.choice(len(self.dataset)))
        else:
            sample_ind = self.sample_ind
            self.sample_ind = (self.sample_ind + 1) % len(self.dataset)
        return _noise_text(self.dataset[sample_ind])

    def _sample_sentences(self, sample_size: int, range_multiply: int = 0) -> None:
        sentences: list[str] = []
        attempts = 0
        range_cutoff = 2 ** min(range_multiply, 16)
        while len(sentences) < 10 and attempts < 100:
            attempts += 1
            text = self._next_sample()
            if self.shuffle:
                if not text:
                    continue
                text = text[int(self.gen.choice(len(text))) // range_cutoff :]
                text = text[: sample_size * 10 * range_cutoff]
            new_sentences = list(self.sentence_splitter(text))
            sentences.extend(new_sentences[1:-1] if self.shuffle else new_sentences)
        self.sentences.extend(sentences)

    def get_sample(self, sample_size: int) -> list[list[int]]:
        if sample_size < 0:
            raise ValueError("sample_size must be non-negative")
        if sample_size == 0:
            return []
        sample: list[list[int]] = []
        total_len = 0
        range_multiply = 0
        while True:
            sentences = list(self.sentences)
            for index, sentence in enumerate(sentences):
                tokenized = self.tokenizer.encode(sentence)
                if not self._length_is_ok(tokenized):
                    continue
                total_len += len(tokenized)
                sample.append(tokenized)
                if total_len >= sample_size:
                    self.sentences = self.sentences[index + 1 :]
                    cutoff = total_len - sample_size
                    if cutoff:
                        sample[-1] = sample[-1][:-cutoff]
                    return sample
            self.sentences = []
            self._sample_sentences(sample_size, range_multiply)
            range_multiply += 1
            if not self.sentences and range_multiply > 16:
                raise RuntimeError(
                    "Could not obtain usable PG-19 sentences; reduce "
                    "--generated-min-sentence-tokens or check the noise split"
                )


def inject_babi_facts(
    background_sentences: list[list[int]],
    fact_tokens: list[list[int]],
    sample_size: int,
    rng: np.random.Generator,
    task_start_pct: float | None = None,
    task_end_pct: float | None = None,
) -> list[int]:
    """Insert facts between noise sentences exactly as the reference dataset."""
    if (task_start_pct is None) != (task_end_pct is None):
        raise ValueError("task_start_pct and task_end_pct must be set together")
    if task_start_pct is None:
        possible_positions = list(range(len(background_sentences) + 1))
    else:
        assert task_end_pct is not None
        if not 0.0 <= task_start_pct < task_end_pct <= 1.0:
            raise ValueError("task placement percentages must satisfy 0 <= start < end <= 1")
        task_start = int(sample_size * task_start_pct)
        task_end = int(sample_size * task_end_pct)
        total_facts_len = sum(map(len, fact_tokens))
        possible_positions = []
        current_length = 0
        for index, sentence in enumerate(background_sentences):
            if current_length >= task_start and current_length < task_end - total_facts_len:
                possible_positions.append(index)
            current_length += len(sentence)
        if not possible_positions:
            raise IndexError(
                "Unable to insert facts in the requested task-position range; "
                "widen the range or increase the sequence length"
            )

    positions = sorted(int(value) for value in rng.choice(
        possible_positions, size=len(fact_tokens), replace=True
    ))
    updated: list[list[list[int]]] = [
        [] for _ in range(len(background_sentences) + 1)
    ]
    for fact, position in zip(fact_tokens, positions):
        updated[position].append(fact)
    for index, sentence in enumerate(background_sentences):
        updated[index].append(sentence)
    return [token for group in updated for sentence in group for token in sentence]


def encode_generated_babilong_example(
    sample: BabiSample,
    tokenizer,
    noise_sampler: SentenceSampler,
    desired_sequence_len: int,
    rng: np.random.Generator,
    qa_margin: int = BABILONG_QA_MARGIN,
    task_start_pct: float | None = None,
    task_end_pct: float | None = None,
    oversize_policy: str = "skip",
) -> tuple[list[int], list[int]] | None:
    """Mix one bAbI sample with PG-19 and create answer-only targets."""
    if oversize_policy not in {"skip", "error"}:
        raise ValueError("oversize_policy must be 'skip' or 'error'")
    context_size = desired_sequence_len - qa_margin
    if context_size <= 0:
        raise ValueError("desired_sequence_len must be larger than qa_margin")

    fact_tokens = tokenizer.encode(list(sample.facts))
    facts_len = sum(map(len, fact_tokens))
    if facts_len > context_size:
        if oversize_policy == "skip":
            return None
        raise ValueError(
            f"bAbI facts need {facts_len} context tokens, but only {context_size} are available"
        )
    background = noise_sampler.get_sample(context_size - facts_len)
    context = inject_babi_facts(
        background,
        fact_tokens,
        context_size,
        rng,
        task_start_pct,
        task_end_pct,
    )

    bos = tokenizer.get_bos_token_id()
    question_tokens = tokenizer.encode(sample.question)
    generation_tokens = tokenizer.encode("GEN")[:1]
    answer_tokens = tokenizer.encode(sample.answer)
    sequence = [bos] + context + question_tokens + generation_tokens + answer_tokens + [bos]
    inputs = sequence[:-1]
    if len(inputs) > desired_sequence_len:
        if oversize_policy == "skip":
            return None
        raise ValueError(
            f"Generated BABILong example needs {len(inputs)} tokens, but the selected "
            f"sequence length is {desired_sequence_len}; increase --generated-qa-margin"
        )
    targets = sequence[1:]
    first_answer_target = len(sequence) - len(answer_tokens) - 2
    targets[:first_answer_target] = [-1] * first_answer_target
    return inputs, targets


class GeneratedBABILongBatchStream:
    """Infinite deterministic stream matching the original on-the-fly pipeline."""

    def __init__(
        self,
        tokenizer,
        batch_size: int,
        max_sequence_length: int,
        segment_size: int,
        device,
        tasks: str | Sequence[str],
        babi_path: str = DEFAULT_BABILONG_BABI_PATH,
        split: str = "train",
        noise_dataset_name: str = DEFAULT_BABILONG_NOISE_DATASET,
        noise_config_name: str | None = None,
        noise_train_split: str = "train",
        noise_validation_split: str = "test",
        vary_lengths: bool = False,
        mixed_length_ratio: float = 0.0,
        max_n_facts: int | None = None,
        task_start_pct: float | None = None,
        task_end_pct: float | None = None,
        qa_margin: int = BABILONG_QA_MARGIN,
        min_sentence_len: int | None = 10,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        oversize_policy: str = "skip",
        cache_dir: str | None = None,
        noise_dataset=None,
        noise_loader: Callable | None = None,
        sentence_splitter: Callable[[str], Sequence[str]] | None = None,
        pin_memory: bool | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        if not 0 <= mixed_length_ratio <= 1:
            raise ValueError("mixed_length_ratio must be between zero and one")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if qa_margin <= 0:
            raise ValueError("qa_margin must be positive")

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.segment_size = segment_size
        self.device = torch.device(device)
        self.tasks = tuple(tasks.split(";")) if isinstance(tasks, str) else tuple(tasks)
        self.babi_path = os.path.abspath(os.path.expanduser(babi_path))
        self.split = split
        self.vary_lengths = vary_lengths
        self.mixed_length_ratio = mixed_length_ratio
        self.max_n_facts = max_n_facts
        self.task_start_pct = task_start_pct
        self.task_end_pct = task_end_pct
        self.qa_margin = qa_margin
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.oversize_policy = oversize_policy
        self.pin_memory = self.device.type == "cuda" if pin_memory is None else pin_memory
        self.sample_sizes = generated_sample_sizes(
            max_sequence_length, segment_size, vary_lengths
        )
        if min(self.sample_sizes) <= qa_margin:
            raise ValueError("Every generated sample length must exceed qa_margin")

        split_seed = seed + (0 if split == "train" else 50_000_003)
        rank_seed = split_seed + rank * 1_000_003
        self._task_rng = random.Random(rank_seed + 11)
        self._row_rng = random.Random(rank_seed + 23)
        self._length_rng = random.Random(split_seed + 37)
        self._injection_rng = np.random.default_rng(rank_seed + 53)
        self._validation_task_position = 0
        self._validation_row_positions = {task: 0 for task in self.tasks}

        babi_split = "train" if split == "train" else "test"
        self._task_samples: dict[str, list[BabiSample]] = {}
        for task in self.tasks:
            rows = load_babi_task_file(
                resolve_babi_task_path(self.babi_path, task, babi_split),
                max_n_facts=max_n_facts,
            )
            local_rows = rows[rank::world_size]
            if not local_rows:
                raise ValueError(
                    f"No {split} bAbI rows for {task} on rank {rank} "
                    f"(world size {world_size})"
                )
            self._task_samples[task] = local_rows

        if cache_dir is None and noise_dataset is None:
            cache_dir = os.path.join(get_base_dir(), "babilong_train_cache")
        self.cache_dir = None if cache_dir is None else os.path.abspath(
            os.path.expanduser(cache_dir)
        )
        if noise_dataset is None:
            loader = load_pg19_noise_split if noise_loader is None else noise_loader
            noise_split = noise_train_split if split == "train" else noise_validation_split
            noise_dataset = loader(
                noise_dataset_name,
                split=noise_split,
                config_name=noise_config_name,
                cache_dir=self.cache_dir,
            )

        max_sentence_len = None
        if task_start_pct is not None and task_end_pct is not None:
            max_sentence_len = int(
                (task_end_pct - task_start_pct) * 0.5 * (max_sequence_length - qa_margin)
            )
        self.noise_sampler = SentenceSampler(
            noise_dataset,
            tokenizer,
            min_sentence_len=min_sentence_len,
            max_sentence_len=max_sentence_len,
            shuffle=True,
            random_seed=rank_seed + 71,
            sentence_splitter=sentence_splitter,
        )
        self.batches_emitted = 0
        self.examples_emitted = 0
        self.oversize_examples_skipped = 0
        self.sample_size_counts: Counter[int] = Counter()
        self.batch_sequence_length_counts: Counter[int] = Counter()
        self.tokens_emitted = 0

    def __iter__(self):
        return self

    def _next_task_and_sample(self) -> tuple[str, BabiSample]:
        if self.split == "train":
            task = self._task_rng.choice(self.tasks)
            return task, self._row_rng.choice(self._task_samples[task])
        task = self.tasks[self._validation_task_position % len(self.tasks)]
        self._validation_task_position += 1
        rows = self._task_samples[task]
        position = self._validation_row_positions[task]
        self._validation_row_positions[task] = position + 1
        return task, rows[position % len(rows)]

    def _next_sample_size(self) -> int:
        if len(self.sample_sizes) == 1:
            return self.sample_sizes[0]
        # Preserve NoiseInjectionDataset.get_sample_size from the reference:
        # mixed_length_ratio is the probability of forcing the longest sample.
        if self._length_rng.random() > self.mixed_length_ratio:
            return self._length_rng.choice(self.sample_sizes)
        return self.sample_sizes[-1]

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        rows: list[tuple[list[int], list[int], int]] = []
        max_attempts = max(self.batch_size * 50, 100)
        attempts = 0
        while len(rows) < self.batch_size and attempts < max_attempts:
            attempts += 1
            _, sample = self._next_task_and_sample()
            sample_size = self._next_sample_size()
            encoded = encode_generated_babilong_example(
                sample,
                self.tokenizer,
                self.noise_sampler,
                sample_size,
                self._injection_rng,
                qa_margin=self.qa_margin,
                task_start_pct=self.task_start_pct,
                task_end_pct=self.task_end_pct,
                oversize_policy=self.oversize_policy,
            )
            if encoded is None:
                self.oversize_examples_skipped += 1
                continue
            rows.append((*encoded, sample_size))
        if len(rows) != self.batch_size:
            raise RuntimeError(
                "Could not construct a full generated BABILong batch; increase "
                "the sample length or QA margin, or use --oversize-policy error"
            )

        batch_sequence_len = max(sample_size for _, _, sample_size in rows)
        bos = self.tokenizer.get_bos_token_id()
        input_rows: list[list[int]] = []
        target_rows: list[list[int]] = []
        for inputs, targets, sample_size in rows:
            padding = batch_sequence_len - len(inputs)
            input_rows.append(inputs + [bos] * padding)
            target_rows.append(targets + [-1] * padding)
            self.sample_size_counts[sample_size] += 1

        inputs_tensor = torch.tensor(
            input_rows, dtype=torch.int32, pin_memory=self.pin_memory
        )
        targets_tensor = torch.tensor(
            target_rows, dtype=torch.int64, pin_memory=self.pin_memory
        )
        non_blocking = self.device.type == "cuda"
        inputs_tensor = inputs_tensor.to(self.device, non_blocking=non_blocking).contiguous()
        targets_tensor = targets_tensor.to(self.device, non_blocking=non_blocking).contiguous()
        self.batches_emitted += 1
        self.examples_emitted += self.batch_size
        self.batch_sequence_length_counts[batch_sequence_len] += 1
        self.tokens_emitted += inputs_tensor.numel()
        return inputs_tensor, targets_tensor

    def stats(self) -> dict:
        return {
            "backend": "generated_babi_pg19",
            "mixing": "random_tasks_and_pg19_sentence_noise",
            "split": self.split,
            "tasks": list(self.tasks),
            "sample_sizes": list(self.sample_sizes),
            "vary_lengths": self.vary_lengths,
            "mixed_length_ratio": self.mixed_length_ratio,
            "batches_emitted": self.batches_emitted,
            "examples_emitted": self.examples_emitted,
            "tokens_emitted": self.tokens_emitted,
            "oversize_examples_skipped": self.oversize_examples_skipped,
            "sample_size_counts": dict(sorted(self.sample_size_counts.items())),
            "batch_sequence_length_counts": dict(
                sorted(self.batch_sequence_length_counts.items())
            ),
        }


__all__ = [
    "BABILONG_QA_MARGIN",
    "DEFAULT_BABILONG_BABI_PATH",
    "DEFAULT_BABILONG_NOISE_DATASET",
    "BabiSample",
    "GeneratedBABILongBatchStream",
    "SentenceSampler",
    "encode_generated_babilong_example",
    "generated_sample_sizes",
    "inject_babi_facts",
    "load_babi_task_file",
    "load_pg19_noise_split",
    "resolve_babi_task_path",
]
