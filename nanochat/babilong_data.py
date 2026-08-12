"""Deterministic BABILong training batches for nanochat GPT and ARMTGPT."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from datasets import load_dataset

from nanochat.babilong_eval import (
    BABILONG_LENGTHS_TRAIN,
    format_babilong_prompt,
    parse_babilong_lengths,
    parse_babilong_lengths_train,
    parse_babilong_tasks,
)
from nanochat.common import get_base_dir


BABILONG_TRAIN_DATASET = "RMT-team/babilong-train-5k-samples"


def load_babilong_training_dataset(
    dataset_name: str,
    length: str,
    task: str,
    cache_dir: str | None = None,
):
    """Load one BABILong train cell through the Hugging Face datasets cache."""
    return load_dataset(
        dataset_name, name=length, split=task, cache_dir=cache_dir, keep_in_memory=False
    )


def _cell_seed(seed: int, task: str, length: str) -> int:
    """Stable per-cell seed that does not depend on Python's hash randomization."""
    task_index = int(task.removeprefix("qa"))
    length_index = BABILONG_LENGTHS_TRAIN.index(length)
    return seed + 1009 * task_index + 9176 * length_index


def split_babilong_indices(
    num_examples: int,
    validation_examples: int,
    seed: int,
    task: str,
    length: str,
) -> tuple[list[int], list[int]]:
    """Return deterministic, disjoint train/validation indices for one cell."""
    if num_examples <= 1:
        raise ValueError("A BABILong training cell must contain at least two examples")
    if validation_examples <= 0 or validation_examples >= num_examples:
        raise ValueError(
            "validation_examples must be positive and smaller than the cell size "
            f"({validation_examples} vs {num_examples})"
        )
    indices = list(range(num_examples))
    random.Random(_cell_seed(seed, task, length)).shuffle(indices)
    validation = indices[:validation_examples]
    train = indices[validation_examples:]
    return train, validation


def encode_babilong_training_example(
    sample: dict,
    task: str,
    tokenizer,
    sequence_len: int,
    oversize_policy: str = "skip",
) -> tuple[list[int], list[int]] | None:
    """Encode one example and mask every target except answer and terminal BOS.

    The prompt is exactly the prompt used by :mod:`nanochat.babilong_eval`.
    BABILong answers are prefixed with one space, and BOS acts as nanochat's
    document terminator. Returned input and target rows both have
    ``sequence_len`` elements; ignored targets are ``-1``.
    """
    if sequence_len <= 0:
        raise ValueError("sequence_len must be positive")
    if oversize_policy not in {"skip", "error"}:
        raise ValueError("oversize_policy must be 'skip' or 'error'")
    target = str(sample["target"]).strip()
    if not target:
        raise ValueError("BABILong target must not be empty")

    bos = tokenizer.get_bos_token_id()
    prompt = format_babilong_prompt(sample, task)
    prompt_ids = tokenizer(prompt, prepend=bos)
    answer_ids = tokenizer(" " + target)
    sequence = prompt_ids + answer_ids + [bos]
    row_capacity = sequence_len + 1
    if len(sequence) > row_capacity:
        if oversize_policy == "skip":
            return None
        raise ValueError(
            f"BABILong {task} example needs {len(sequence) - 1} tokens, "
            f"but sequence_len is {sequence_len}"
        )

    inputs = sequence[:-1]
    targets = sequence[1:]
    # targets[i] predicts sequence[i + 1]. The first answer token is at
    # sequence[len(prompt_ids)], so its prediction index is one position back.
    answer_target_start = len(prompt_ids) - 1
    targets[:answer_target_start] = [-1] * answer_target_start

    padding = sequence_len - len(inputs)
    inputs.extend([bos] * padding)
    targets.extend([-1] * padding)
    return inputs, targets


@dataclass
class _CellState:
    base_indices: list[int]
    order: list[int]
    position: int = 0
    epoch: int = 0


class BABILongBatchStream:
    """Infinite deterministic stream over length-bucketed task mixtures.

    All selected tasks for one context length are materialized together and
    interleaved within batches. After ``cell_batches`` batches the stream moves
    to another length, avoiding padding short and long contexts together. DDP
    ranks share the length/task schedule and consume disjoint rows per task.
    """

    def __init__(
        self,
        tokenizer,
        batch_size: int,
        sequence_len: int,
        device,
        tasks: str | Sequence[str] | None = None,
        lengths: str | Sequence[str] | None = None,
        dataset_name: str = BABILONG_TRAIN_DATASET,
        split: str = "train",
        validation_examples: int = 100,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        cell_batches: int = 64,
        oversize_policy: str = "skip",
        dataset_loader: Callable | None = None,
        pin_memory: bool | None = None,
        cache_dir: str | None = None,
    ) -> None:
        if batch_size <= 0 or sequence_len <= 0:
            raise ValueError("batch_size and sequence_len must be positive")
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if cell_batches <= 0:
            raise ValueError("cell_batches must be positive")
        if oversize_policy not in {"skip", "error"}:
            raise ValueError("oversize_policy must be 'skip' or 'error'")

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.device = torch.device(device)
        self.tasks = parse_babilong_tasks(tasks)
        self.lengths = parse_babilong_lengths_train(lengths)
        self.dataset_name = dataset_name
        self.split = split
        self.validation_examples = validation_examples
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.cell_batches = cell_batches
        self.oversize_policy = oversize_policy
        self.dataset_loader = dataset_loader
        if cache_dir is None and dataset_loader is None:
            cache_dir = os.path.join(get_base_dir(), "babilong_train_cache")
        self.cache_dir = (
            None
            if cache_dir is None
            else os.path.abspath(os.path.expanduser(cache_dir))
        )
        self.pin_memory = self.device.type == "cuda" if pin_memory is None else pin_memory

        self._states: dict[tuple[str, str], _CellState] = {}
        self._length_order: list[str] = []
        self._length_position = 0
        self._length_cycle = 0
        self._task_orders: dict[str, list[str]] = {}
        self._task_positions: dict[str, int] = {}
        self._task_cycles: dict[str, int] = {}
        self._task_batch_counts: dict[str, int] = {}
        self._active_length: str | None = None
        self._active_datasets: dict[str, object] = {}
        self._active_batches = 0
        self.batches_emitted = 0
        self.examples_emitted = 0
        self.oversize_examples_skipped = 0

    def __iter__(self):
        return self

    def _next_length(self) -> str:
        if self._length_position >= len(self._length_order):
            self._length_order = list(self.lengths)
            random.Random(self.seed + 1_000_003 * self._length_cycle).shuffle(
                self._length_order
            )
            self._length_position = 0
            self._length_cycle += 1
        length = self._length_order[self._length_position]
        self._length_position += 1
        return length

    def _next_task(self, length: str) -> str:
        order = self._task_orders.get(length, [])
        position = self._task_positions.get(length, 0)
        if position >= len(order):
            cycle = self._task_cycles.get(length, 0)
            order = list(self.tasks)
            length_index = BABILONG_LENGTHS_TRAIN.index(length)
            random.Random(
                self.seed + 2_000_003 * cycle + 19_997 * length_index
            ).shuffle(order)
            position = 0
            self._task_orders[length] = order
            self._task_cycles[length] = cycle + 1
        task = order[position]
        self._task_positions[length] = position + 1
        return task

    def _tasks_for_batch(self, length: str) -> list[str]:
        """Build a balanced task mixture, including every task when it fits."""
        task_count = len(self.tasks)
        repeats, remainder = divmod(self.batch_size, task_count)
        batch_tasks = list(self.tasks) * repeats
        draw_count = remainder if repeats else self.batch_size
        batch_tasks.extend(self._next_task(length) for _ in range(draw_count))

        batch_number = self._task_batch_counts.get(length, 0)
        length_index = BABILONG_LENGTHS_TRAIN.index(length)
        random.Random(
            self.seed + 3_000_017 * batch_number + 29_987 * length_index
        ).shuffle(batch_tasks)
        self._task_batch_counts[length] = batch_number + 1
        return batch_tasks

    def _make_order(self, cell: tuple[str, str], indices: list[int], epoch: int) -> list[int]:
        order = indices.copy()
        if self.split == "train":
            task, length = cell
            random.Random(
                _cell_seed(self.seed, task, length) + 10_000_019 * (epoch + 1)
            ).shuffle(order)
        return order

    def _activate_next_length(self) -> None:
        self._active_datasets = {}
        length = self._next_length()
        for task in self.tasks:
            if self.dataset_loader is None:
                data = load_babilong_training_dataset(
                    self.dataset_name, length, task, cache_dir=self.cache_dir
                )
            else:
                # Keep loader injection lightweight for deterministic unit tests.
                data = self.dataset_loader(
                    self.dataset_name, subset=length, split=task
                )
            self._active_datasets[task] = data
            cell = (task, length)
            if cell not in self._states:
                train_indices, validation_indices = split_babilong_indices(
                    len(data), self.validation_examples, self.seed, task, length
                )
                indices = train_indices if self.split == "train" else validation_indices
                local_indices = indices[self.rank :: self.world_size]
                if not local_indices:
                    raise ValueError(
                        f"No {self.split} rows for {task} @ {length} on rank "
                        f"{self.rank} (world size {self.world_size})"
                    )
                self._states[cell] = _CellState(
                    base_indices=local_indices,
                    order=self._make_order(cell, local_indices, epoch=0),
                )
        self._active_length = length
        self._active_batches = 0

    def _next_index(self, state: _CellState, cell: tuple[str, str]) -> int:
        if state.position >= len(state.order):
            state.epoch += 1
            state.order = self._make_order(cell, state.base_indices, state.epoch)
            state.position = 0
        index = state.order[state.position]
        state.position += 1
        return index

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._active_datasets or self._active_batches >= self.cell_batches:
            self._activate_next_length()
        assert self._active_length is not None
        length = self._active_length

        input_rows = []
        target_rows = []
        batch_tasks = self._tasks_for_batch(length)
        for task in batch_tasks:
            cell = (task, length)
            state = self._states[cell]
            max_attempts = max(len(state.base_indices) * 2, 20)
            encoded = None
            for _ in range(max_attempts):
                index = self._next_index(state, cell)
                encoded = encode_babilong_training_example(
                    self._active_datasets[task][index],
                    task,
                    self.tokenizer,
                    self.sequence_len,
                    self.oversize_policy,
                )
                if encoded is not None:
                    break
                self.oversize_examples_skipped += 1
            if encoded is None:
                raise RuntimeError(
                    f"Could not find a fitting BABILong example for {cell}; "
                    "increase --max-seq-len or use --oversize-policy error for details"
                )
            inputs, targets = encoded
            input_rows.append(inputs)
            target_rows.append(targets)

        inputs = torch.tensor(input_rows, dtype=torch.int32, pin_memory=self.pin_memory)
        targets = torch.tensor(target_rows, dtype=torch.int64, pin_memory=self.pin_memory)
        non_blocking = self.device.type == "cuda"
        inputs = inputs.to(self.device, non_blocking=non_blocking).contiguous()
        targets = targets.to(self.device, non_blocking=non_blocking).contiguous()
        self._active_batches += 1
        self.batches_emitted += 1
        self.examples_emitted += self.batch_size
        return inputs, targets

    def stats(self) -> dict:
        return {
            "mixing": "tasks_within_length_batches",
            "batches_emitted": self.batches_emitted,
            "examples_emitted": self.examples_emitted,
            "oversize_examples_skipped": self.oversize_examples_skipped,
            "length_cycle": self._length_cycle,
            "active_length": self._active_length,
            "task_cycles": dict(self._task_cycles),
            "task_batch_counts": dict(self._task_batch_counts),
        }


__all__ = [
    "BABILONG_TRAIN_DATASET",
    "BABILongBatchStream",
    "encode_babilong_training_example",
    "load_babilong_training_dataset",
    "split_babilong_indices",
]
