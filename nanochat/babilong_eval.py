"""BABILong loading, generation, scoring, and aggregation for nanochat models."""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist

from nanochat.common import COMPUTE_DTYPE, print0
from nanochat.engine import KVCache
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
DEFAULT_BABILONG_BATCH_SIZE = 8


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


def _terminal_token_ids(tokenizer) -> tuple[int, int]:
    return tokenizer.encode_special("<|assistant_end|>"), tokenizer.get_bos_token_id()


@torch.inference_mode()
def _generate_gpt_completions(
    model,
    tokenizer,
    prompt_tokens: Sequence[list[int]],
    max_new_tokens: int,
) -> list[str]:
    """Prefill variable-length prompts separately, then decode them as one batch."""
    batch_size = len(prompt_tokens)
    raw_model = getattr(model, "_orig_mod", model)
    device = raw_model.get_device()
    config = raw_model.config
    kv_kwargs = {
        "num_heads": config.n_kv_head,
        "head_dim": config.n_embd // config.n_head,
        "num_layers": config.n_layer,
    }
    max_cache_len = max(map(len, prompt_tokens)) + max_new_tokens
    batch_cache = KVCache(
        batch_size=batch_size,
        seq_len=max_cache_len,
        device=device,
        dtype=COMPUTE_DTYPE,
        **kv_kwargs,
    )

    next_logits = [None] * batch_size
    previous_embeddings = [None] * batch_size
    groups: dict[int, list[int]] = defaultdict(list)
    for row, tokens in enumerate(prompt_tokens):
        groups[len(tokens)].append(row)

    # Prefill equal-length rows together. Uneven rows cannot be padded because
    # nanochat GPT has no padding mask, but all rows share one cache for decode.
    for rows in groups.values():
        prompt_length = len(prompt_tokens[rows[0]])
        prefill_cache = KVCache(
            batch_size=len(rows),
            seq_len=prompt_length,
            device=device,
            dtype=COMPUTE_DTYPE,
            **kv_kwargs,
        )
        ids = torch.tensor(
            [prompt_tokens[row] for row in rows], dtype=torch.long, device=device
        )
        logits = raw_model.forward(ids, kv_cache=prefill_cache)
        length = prefill_cache.get_pos()
        for group_row, row in enumerate(rows):
            next_logits[row] = logits[group_row : group_row + 1, -1, :]
            batch_cache.k_cache[:, row : row + 1, :length].copy_(
                prefill_cache.k_cache[:, group_row : group_row + 1, :length]
            )
            batch_cache.v_cache[:, row : row + 1, :length].copy_(
                prefill_cache.v_cache[:, group_row : group_row + 1, :length]
            )
            batch_cache.cache_seqlens[row] = length
            previous_embeddings[row] = prefill_cache.prev_embedding[
                group_row : group_row + 1
            ]
    batch_cache.prev_embedding = torch.cat(previous_embeddings, dim=0)

    logits = torch.cat(next_logits, dim=0)
    assistant_end, bos = _terminal_token_ids(tokenizer)
    generated = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for step in range(max_new_tokens):
        next_ids = torch.argmax(logits, dim=-1)
        terminal = (next_ids == assistant_end) | (next_ids == bos)
        active = ~finished
        for row, token in enumerate(next_ids.tolist()):
            if active[row].item() and not terminal[row].item():
                generated[row].append(token)
        finished |= terminal
        if finished.all() or step + 1 == max_new_tokens:
            break
        # Finished rows remain in the tensor so active rows can continue decoding.
        # Their state is irrelevant, and BOS is a stable filler token.
        decode_ids = torch.where(finished, torch.full_like(next_ids, bos), next_ids)
        logits = raw_model.forward(
            decode_ids.unsqueeze(1), kv_cache=batch_cache
        )[:, -1, :]
    return [tokenizer.decode(tokens).strip() for tokens in generated]


@torch.inference_mode()
def _generate_armt_completions(
    model,
    tokenizer,
    prompt_tokens: Sequence[list[int]],
    max_new_tokens: int,
) -> list[str]:
    """Generate a true mixed-length batch through ARMT's padded forward path."""
    raw_model = getattr(model, "_orig_mod", model)
    device = raw_model.get_device()
    assistant_end, bos = _terminal_token_ids(tokenizer)
    sequences = [tokens.copy() for tokens in prompt_tokens]
    generated = [[] for _ in prompt_tokens]
    finished = torch.zeros(len(prompt_tokens), dtype=torch.bool, device=device)
    for step in range(max_new_tokens):
        lengths = torch.tensor([len(tokens) for tokens in sequences], device=device)
        padded_width = int(lengths.max().item())
        ids = torch.full(
            (len(sequences), padded_width),
            bos,
            dtype=torch.long,
            device=device,
        )
        for row, tokens in enumerate(sequences):
            ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
        logits = raw_model.forward_padded(ids, lengths)
        batch_rows = torch.arange(len(sequences), device=device)
        next_ids = torch.argmax(logits[batch_rows, lengths - 1], dim=-1)
        terminal = (next_ids == assistant_end) | (next_ids == bos)
        active = ~finished
        for row, token in enumerate(next_ids.tolist()):
            if active[row].item() and not terminal[row].item():
                sequences[row].append(token)
                generated[row].append(token)
        finished |= terminal
        if finished.all() or step + 1 == max_new_tokens:
            break
    return [tokenizer.decode(tokens).strip() for tokens in generated]


@torch.inference_mode()
def generate_babilong_completions(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int = 20,
) -> list[str]:
    """Greedily generate completion-only strings for a batch of prompts."""
    if not 10 <= max_new_tokens <= 30:
        raise ValueError("BABILong max_new_tokens must be between 10 and 30")
    if isinstance(prompts, str) or not prompts:
        raise ValueError("prompts must be a non-empty sequence of strings")
    bos = tokenizer.get_bos_token_id()
    prompt_tokens = [tokenizer(prompt, prepend=bos) for prompt in prompts]
    _ensure_rotary_capacity(model, max(map(len, prompt_tokens)) + max_new_tokens)
    raw_model = getattr(model, "_orig_mod", model)
    if hasattr(raw_model, "armt_layers"):
        return _generate_armt_completions(
            raw_model, tokenizer, prompt_tokens, max_new_tokens
        )
    return _generate_gpt_completions(raw_model, tokenizer, prompt_tokens, max_new_tokens)


def generate_babilong_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 20,
) -> str:
    """Backward-compatible single-prompt wrapper."""
    return generate_babilong_completions(
        model, tokenizer, [prompt], max_new_tokens=max_new_tokens
    )[0]


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
    batch_size: int = DEFAULT_BABILONG_BATCH_SIZE,
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
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    dataset_loader = dataset_loader or load_hub_dataset
    generation_fn = generation_fn or generate_babilong_completions

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    results = {task: {} for task in selected_tasks}
    example_counts = {task: {} for task in selected_tasks}
    cell_eval_seconds = {task: {} for task in selected_tasks}

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
            local_indices = indices[rank::world_size]
            for batch_start in range(0, len(local_indices), batch_size):
                batch_indices = local_indices[batch_start : batch_start + batch_size]
                samples = [data[index] for index in batch_indices]
                prompts = [format_babilong_prompt(sample, task) for sample in samples]
                completions = generation_fn(model, tokenizer, prompts, max_new_tokens)
                if len(completions) != len(samples):
                    raise ValueError(
                        "BABILong generation_fn returned a different number of completions "
                        "than prompts"
                    )
                local_correct += sum(
                    answer_in_completion(sample["target"], completion)
                    for sample, completion in zip(samples, completions)
                )
                local_count += len(samples)

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
            elapsed = time.time() - started
            cell_eval_seconds[task][length] = elapsed
            print0(
                f"accuracy: {accuracy:.4f} "
                f"({int(correct_count)}/{int(evaluated_count)}) | "
                f"time: {elapsed:.2f}s"
            )

    aggregates = aggregate_babilong_scores(results)
    task_eval_seconds = {
        task: sum(length_times.values())
        for task, length_times in cell_eval_seconds.items()
    }
    length_eval_seconds = {
        length: sum(cell_eval_seconds[task][length] for task in selected_tasks)
        for length in selected_lengths
    }
    return {
        "results": results,
        "task_results": aggregates["task_results"],
        "length_results": aggregates["length_results"],
        "babilong_metric": aggregates["babilong_metric"],
        "example_counts": example_counts,
        "tasks": list(selected_tasks),
        "lengths": list(selected_lengths),
        "batch_size": batch_size,
        "eval_time": {
            "per_task_per_length_seconds": cell_eval_seconds,
            "per_task_seconds": task_eval_seconds,
            "per_length_seconds": length_eval_seconds,
            "total_seconds": sum(task_eval_seconds.values()),
        },
    }


__all__ = [
    "BABILONG_DATASET",
    "BABILONG_TASKS",
    "BABILONG_LENGTHS",
    "DEFAULT_BABILONG_TASKS",
    "DEFAULT_BABILONG_LENGTHS",
    "DEFAULT_BABILONG_BATCH_SIZE",
    "parse_babilong_tasks",
    "parse_babilong_lengths",
    "format_babilong_prompt",
    "answer_in_completion",
    "aggregate_babilong_scores",
    "generate_babilong_completions",
    "generate_babilong_completion",
    "evaluate_babilong",
]
