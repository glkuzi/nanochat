"""BABILong loading, generation, scoring, and aggregation for nanochat models."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist

from nanochat.common import print0
from nanochat.engine import Engine
from tasks.common import load_hub_dataset


BABILONG_DATASET = "RMT-team/babilong"
BABILONG_TASKS = tuple(f"qa{i}" for i in range(1, 21))
# Configs available in the 100-example evaluation dataset. The official 1000
# sample variant additionally contains 64k, but it is intentionally not used by
# default here.
BABILONG_LENGTHS = (
    "0k", "1k", "2k", "4k", "8k", "16k", "32k",
    "128k", "256k", "512k", "1M",
)
DEFAULT_BABILONG_TASKS = BABILONG_TASKS[:5]
DEFAULT_BABILONG_LENGTHS = BABILONG_LENGTHS[:7]


_TASK_INSTRUCTIONS = {
    "qa1": "Answer the question using the most recent stated location of the person.",
    "qa2": "Track people and the objects they carry; an object moves with its carrier unless it is dropped.",
    "qa3": "Track the successive locations of people and the objects they carry.",
    "qa4": "Use the stated spatial relations between locations to answer the question.",
    "qa5": "Track who gives each object, who receives it, and which object is transferred.",
}


def parse_babilong_selection(
    value: str | Sequence[str] | None,
    available: Sequence[str],
    default: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    """Parse comma/space selections and inclusive ranges such as ``qa1-qa5``."""
    if value is None or value == "":
        return tuple(default)
    raw_parts = value.replace(",", " ").split() if isinstance(value, str) else list(value)
    if not raw_parts:
        return tuple(default)

    positions = {name.lower(): index for index, name in enumerate(available)}
    selected = []
    for raw_part in raw_parts:
        part = raw_part.strip()
        key = part.lower()
        if key in positions:
            selected.append(available[positions[key]])
            continue
        endpoints = part.split("-", maxsplit=1)
        if len(endpoints) == 2 and all(endpoint.lower() in positions for endpoint in endpoints):
            start = positions[endpoints[0].lower()]
            end = positions[endpoints[1].lower()]
            if start > end:
                raise ValueError(f"Invalid descending {label} range: {part!r}")
            selected.extend(available[start : end + 1])
            continue
        raise ValueError(
            f"Unknown BABILong {label} {part!r}; available values: {', '.join(available)}"
        )

    # Preserve the requested order while removing duplicates.
    return tuple(dict.fromkeys(selected))


def parse_babilong_tasks(value: str | Sequence[str] | None = None) -> tuple[str, ...]:
    return parse_babilong_selection(value, BABILONG_TASKS, DEFAULT_BABILONG_TASKS, "task")


def parse_babilong_lengths(value: str | Sequence[str] | None = None) -> tuple[str, ...]:
    return parse_babilong_selection(value, BABILONG_LENGTHS, DEFAULT_BABILONG_LENGTHS, "length")


def format_babilong_prompt(sample: dict, task: str) -> str:
    """Render a stable zero-shot prompt from a BABILong dataset row."""
    instruction = _TASK_INSTRUCTIONS.get(
        task,
        "Answer the question using only facts found in the context.",
    )
    context = sample["input"].strip()
    question = sample["question"].strip()
    return (
        f"{instruction}\n\n"
        f"<context>\n{context}\n</context>\n\n"
        f"Question: {question}\nAnswer:"
    )


def answer_in_completion(target: str, completion: str) -> bool:
    """Requested BABILong metric: target occurs in the generated completion."""
    normalized_target = str(target).strip().casefold()
    return bool(normalized_target) and normalized_target in str(completion).casefold()


def aggregate_babilong_scores(results: dict[str, dict[str, float]]) -> dict:
    """Macro-average a task-by-length score matrix in both directions."""
    if not results:
        raise ValueError("Cannot aggregate empty BABILong results")
    tasks = tuple(results)
    lengths = tuple(next(iter(results.values())))
    if not lengths:
        raise ValueError("Cannot aggregate BABILong results without lengths")
    for task in tasks:
        if tuple(results[task]) != lengths:
            raise ValueError("All BABILong tasks must contain the same ordered lengths")

    task_results = {
        task: sum(results[task].values()) / len(lengths)
        for task in tasks
    }
    length_results = {
        length: sum(results[task][length] for task in tasks) / len(tasks)
        for length in lengths
    }
    overall = sum(task_results.values()) / len(task_results)
    return {
        "task_results": task_results,
        "length_results": length_results,
        "babilong_metric": overall,
    }


def _required_rotary_positions(model, real_tokens: int) -> int:
    """Account for ARMT's inserted memory/sink positions when extending RoPE."""
    config = model.config
    if not hasattr(config, "num_mem_tokens"):
        return real_tokens
    extras = config.num_mem_tokens + int(getattr(config, "use_sink", False))
    if getattr(config, "segment_position_mode", "absolute") == "reset":
        return min(real_tokens, config.segment_size) + extras
    num_segments = math.ceil(real_tokens / config.segment_size)
    return real_tokens + num_segments * extras


@torch.no_grad()
def _ensure_rotary_capacity(model, real_tokens: int) -> None:
    """Grow nanochat's non-persistent RoPE cache for long evaluation prompts."""
    raw_model = getattr(model, "_orig_mod", model)
    if not all(hasattr(raw_model, name) for name in ("cos", "sin", "_precompute_rotary_embeddings")):
        return
    required = _required_rotary_positions(raw_model, real_tokens)
    if required <= raw_model.cos.size(1):
        return
    new_size = max(required, raw_model.cos.size(1) * 2)
    head_dim = raw_model.config.n_embd // raw_model.config.n_head
    cos, sin = raw_model._precompute_rotary_embeddings(new_size, head_dim)
    raw_model.cos, raw_model.sin = cos, sin
    raw_model.rotary_seq_len = new_size
    print0(f"Extended rotary cache to {new_size:,} positions for BABILong")


@torch.inference_mode()
def generate_babilong_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 20,
) -> str:
    """Greedily generate only the completion portion of one BABILong prompt."""
    if not 10 <= max_new_tokens <= 30:
        raise ValueError("BABILong max_new_tokens must be between 10 and 30")
    prompt_tokens = tokenizer(prompt, prepend=tokenizer.get_bos_token_id())
    _ensure_rotary_capacity(model, len(prompt_tokens) + max_new_tokens)

    # ARMT currently uses an exact prefix-recomputation generation path. Calling
    # it directly avoids allocating the base GPT KV cache, which ARMT does not use.
    raw_model = getattr(model, "_orig_mod", model)
    if hasattr(raw_model, "armt_layers"):
        completion_tokens = list(
            raw_model.generate(
                prompt_tokens,
                max_tokens=max_new_tokens,
                temperature=0.0,
            )
        )
    else:
        engine = Engine(model, tokenizer)
        generated, _ = engine.generate_batch(
            prompt_tokens,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        completion_tokens = generated[0][len(prompt_tokens) :]
    return tokenizer.decode(completion_tokens).strip()


def evaluate_babilong(
    model,
    tokenizer,
    device,
    tasks: str | Sequence[str] | None = None,
    lengths: str | Sequence[str] | None = None,
    max_per_task: int = -1,
    max_new_tokens: int = 20,
    dataset_name: str = BABILONG_DATASET,
    dataset_loader: Callable | None = None,
    generation_fn: Callable | None = None,
) -> dict:
    """Load and score BABILong, returning cell, task, length, and overall metrics.

    Accuracies are in ``[0, 1]``. Aggregation is macro averaging: every selected
    task/length cell has equal weight regardless of its example count.
    """
    selected_tasks = parse_babilong_tasks(tasks)
    selected_lengths = parse_babilong_lengths(lengths)
    if max_per_task == 0 or max_per_task < -1:
        raise ValueError("max_per_task must be -1 or a positive integer")
    if not 10 <= max_new_tokens <= 30:
        raise ValueError("BABILong max_new_tokens must be between 10 and 30")
    dataset_loader = dataset_loader or load_hub_dataset
    generation_fn = generation_fn or generate_babilong_completion

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    results = {task: {} for task in selected_tasks}
    example_counts = {task: {} for task in selected_tasks}

    for length in selected_lengths:
        for task in selected_tasks:
            started = time.time()
            print0(f"Evaluating BABILong: {task} @ {length}... ", end="")
            data = dataset_loader(dataset_name, subset=length, split=task)
            # Match CORE's deterministic subsampling without assuming the full
            # Hugging Face datasets package is installed.
            indices = list(range(len(data)))
            random.Random(1337).shuffle(indices)
            if max_per_task > 0:
                indices = indices[:max_per_task]
            if not indices:
                raise ValueError(f"BABILong dataset is empty for {task} @ {length}")

            local_correct = 0
            local_count = 0
            for logical_index in range(rank, len(indices), world_size):
                sample = data[indices[logical_index]]
                prompt = format_babilong_prompt(sample, task)
                completion = generation_fn(model, tokenizer, prompt, max_new_tokens)
                local_correct += int(answer_in_completion(sample["target"], completion))
                local_count += 1

            counts = torch.tensor(
                [local_correct, local_count],
                dtype=torch.float32,
                device=device,
            )
            if world_size > 1:
                dist.barrier()
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            correct_count, evaluated_count = counts.tolist()
            accuracy = correct_count / evaluated_count
            results[task][length] = accuracy
            example_counts[task][length] = int(evaluated_count)
            print0(
                f"accuracy: {accuracy:.4f} "
                f"({int(correct_count)}/{int(evaluated_count)}) | "
                f"time: {time.time() - started:.2f}s"
            )

    aggregates = aggregate_babilong_scores(results)
    return {
        "results": results,
        "task_results": aggregates["task_results"],
        "length_results": aggregates["length_results"],
        "babilong_metric": aggregates["babilong_metric"],
        "example_counts": example_counts,
        "tasks": list(selected_tasks),
        "lengths": list(selected_lengths),
    }


__all__ = [
    "BABILONG_DATASET",
    "BABILONG_TASKS",
    "BABILONG_LENGTHS",
    "DEFAULT_BABILONG_TASKS",
    "DEFAULT_BABILONG_LENGTHS",
    "parse_babilong_tasks",
    "parse_babilong_lengths",
    "format_babilong_prompt",
    "answer_in_completion",
    "aggregate_babilong_scores",
    "generate_babilong_completion",
    "evaluate_babilong",
]
