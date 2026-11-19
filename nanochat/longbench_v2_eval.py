"""LongBench-v2 direct-answer evaluation for nanochat GPT and recurrent ARMT.

Length filtering excludes examples, never truncates their documents. Numeric
lengths are measured from the context, not the dataset's short/medium/long label.
The response parser follows https://github.com/THUDM/LongBench/blob/main/pred.py.
"""

import json
import os
import random
import re
import time

import torch.distributed as dist

from nanochat.babilong_eval import (
    _ensure_rotary_capacity,
    _generate_armt_completions,
    _generate_gpt_completions,
)
from nanochat.common import get_dist_info, print0


LONGBENCH_V2_DATASET = "zai-org/LongBench-v2"
DEFAULT_LENGTH_BUCKETS = "16k,20k,32k,64k,128k"


def parse_length(value):
    """Parse a positive integer or a binary-k length, e.g. 16384 or 16k."""
    match = re.fullmatch(r"([0-9]+)([kK]?)", str(value).strip())
    if match is None:
        raise ValueError(f"Invalid length {value!r}; use a positive integer or e.g. 16k")
    length = int(match[1]) * (1024 if match[2] else 1)
    if length <= 0:
        raise ValueError("Lengths must be positive")
    return length


def parse_length_buckets(value=DEFAULT_LENGTH_BUCKETS):
    parts = value.split(",") if isinstance(value, str) else value
    edges = tuple(parse_length(part) for part in parts)
    if not edges or any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("Length bucket boundaries must be nonempty and strictly increasing")
    return edges


def _length_label(value):
    return f"{value // 1024}k" if value % 1024 == 0 else str(value)


def render_longbench_prompt(row):
    """Direct-answer (non-CoT) prompt using LongBench-v2's answer format."""
    choices = "\n".join(f"({letter}) {row[f'choice_{letter}'].strip()}" for letter in "ABCD")
    return (
        "Use the supplied document to choose the best answer.\n\n"
        f"<text>\n{row['context'].strip()}\n</text>\n\n"
        f"Question: {row['question'].strip()}\nChoices:\n{choices}\n\n"
        'Respond using this format: "The correct answer is (insert answer here)".'
    )


def encode_longbench_prompt(tokenizer, row):
    # render_for_completion truncates at 2048 tokens. Construct the identical
    # one-user-message framing directly, preserving the entire input.
    return [tokenizer.get_bos_token_id(), tokenizer.encode_special("<|user_start|>")] + (
        tokenizer.encode(render_longbench_prompt(row))
    ) + [tokenizer.encode_special("<|user_end|>"), tokenizer.encode_special("<|assistant_start|>")]


def extract_answer(response):
    """Upstream strict answer extraction; malformed responses count as wrong."""
    response = response.replace("*", "")
    for pattern in (r"The correct answer is \(([A-D])\)", r"The correct answer is ([A-D])"):
        match = re.search(pattern, response)
        if match:
            return match[1]
    return None


def generate_longbench_completions(model, tokenizer, prompt_tokens, max_new_tokens=128):
    if not prompt_tokens or max_new_tokens <= 0:
        raise ValueError("Generation requires nonempty prompts and positive max_new_tokens")
    # Grow outside inference_mode so persistent buffers remain usable in training.
    _ensure_rotary_capacity(model, max(map(len, prompt_tokens)) + max_new_tokens, "LongBench-v2")
    raw_model = getattr(model, "_orig_mod", model)
    generate = _generate_armt_completions if hasattr(raw_model, "armt_layers") else _generate_gpt_completions
    return generate(raw_model, tokenizer, prompt_tokens, max_new_tokens)


def _score(records):
    correct = sum(int(row["correct"]) for row in records)
    total = len(records)
    return {"accuracy": correct / total if total else None, "correct": correct, "total": total}


def aggregate_longbench_scores(records, bucket_edges):
    edges = parse_length_buckets(bucket_edges)
    buckets, cumulative = {}, {}
    lower = 0
    for upper in edges:
        label = f"<={_length_label(upper)}" if lower == 0 else f"{_length_label(lower)}-{_length_label(upper)}"
        buckets[label] = _score([r for r in records if lower < r["context_length"] <= upper])
        cumulative[f"<={_length_label(upper)}"] = _score([r for r in records if r["context_length"] <= upper])
        lower = upper
    buckets[f">{_length_label(lower)}"] = _score([r for r in records if r["context_length"] > lower])
    result = {"overall": _score(records), "length_buckets": buckets, "length_cumulative": cumulative}
    for field in ("domain", "difficulty", "dataset_length"):
        result[f"by_{field}"] = {
            value: _score([r for r in records if r[field] == value])
            for value in sorted({r[field] for r in records})
        }
    return result


def evaluate_longbench_v2(
    model, tokenizer, *, max_length=None, length_unit="tokens",
    length_buckets=DEFAULT_LENGTH_BUCKETS, batch_size=1, max_new_tokens=128,
    max_problems=None, seed=42, dataset_name=LONGBENCH_V2_DATASET,
    cache_dir=None, revision=None, dataset_loader=None, generation_fn=None,
):
    """Evaluate the HF train split (the released evaluation set), with DDP sharding.

    Overall/bucket accuracy is micro-averaged over selected examples. Limits are
    inclusive context-only lengths; framing/question/choices add prompt tokens.
    max_problems selects a seeded subset AFTER filtering. Empty buckets have
    accuracy=None, and selecting no examples is an error, not a zero score.
    """
    max_length = None if max_length is None else parse_length(max_length)
    edges = parse_length_buckets(length_buckets)
    if length_unit not in ("tokens", "words"):
        raise ValueError("length_unit must be tokens or words")
    if batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("batch_size and max_new_tokens must be positive")
    if max_problems is not None and max_problems <= 0:
        raise ValueError("max_problems must be positive or None")
    if dataset_loader is None:
        from datasets import load_dataset
        dataset_loader = load_dataset
    generation_fn = generation_fn or generate_longbench_completions
    ddp, rank, _, world_size = get_dist_info()
    started = time.perf_counter()
    dataset = dataset_loader(dataset_name, split="train", cache_dir=cache_dir, revision=revision)
    # Retain only indices/lengths, not all the potentially million-token texts.
    # Shard length computation too, then share the tiny numeric index.
    measured = []
    print0(f"LongBench-v2: measuring context lengths in {length_unit} (1k = 1024)...")
    for index in range(rank, len(dataset), world_size):
        context = dataset[index]["context"].strip()
        length = len(tokenizer.encode(context)) if length_unit == "tokens" else len(context.split())
        if length == 0:
            raise ValueError(f"LongBench-v2 example {index} has an empty context")
        measured.append((index, length))
    if ddp:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, measured)
        measured = [entry for shard in gathered for entry in shard]
    eligible = sorted((i, n) for i, n in measured if max_length is None or n <= max_length)
    eligible_count = len(eligible)
    if not eligible:
        raise ValueError("No LongBench-v2 examples match the length limit; increase --longbench-v2-max-length")
    if max_problems is not None and max_problems < len(eligible):
        eligible = sorted(random.Random(seed).sample(eligible, max_problems))
    # Group similar lengths to reduce padding in ARMT, deterministically on all ranks.
    local_examples = sorted(eligible, key=lambda item: (item[1], item[0]))[rank::world_size]
    print0(f"LongBench-v2: evaluating {len(eligible)}/{len(dataset)} examples; greedy generation, batch size {batch_size}")
    records = []
    raw_model = getattr(model, "_orig_mod", model)
    was_training = raw_model.training
    raw_model.eval()
    try:
        for start in range(0, len(local_examples), batch_size):
            batch = local_examples[start : start + batch_size]
            rows = [dataset[index] for index, _ in batch]
            prompts = [encode_longbench_prompt(tokenizer, row) for row in rows]
            outputs = generation_fn(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
            if len(outputs) != len(rows):
                raise ValueError("Generation returned the wrong number of LongBench-v2 completions")
            for (index, length), row, prompt, output in zip(batch, rows, prompts, outputs):
                prediction = extract_answer(output)
                records.append({
                    "index": index, "id": row.get("_id", str(index)),
                    "context_length": length, "prompt_tokens": len(prompt),
                    "domain": row.get("domain", "unknown"),
                    "difficulty": row.get("difficulty", "unknown"),
                    "dataset_length": row.get("length", "unknown"),
                    "generated_answer": output, "prediction": prediction,
                    "ground_truth_answer": row["answer"], "correct": prediction == row["answer"],
                })
            print0(f"LongBench-v2 rank 0 progress: {min(start + batch_size, len(local_examples))}/{len(local_examples)}")
    finally:
        raw_model.train(was_training)
    if ddp:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, records)
        records = [record for shard in gathered for record in shard]
    records.sort(key=lambda row: row["index"])
    result = aggregate_longbench_scores(records, edges)
    result.update({
        "longbench_v2_metric": result["overall"]["accuracy"],
        "examples": records, "eval_time_seconds": time.perf_counter() - started,
        "config": {
            "dataset": dataset_name, "revision": revision,
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None), "split": "train",
            "length_unit": length_unit, "length_scope": "context", "k": 1024,
            "max_length": max_length, "length_bucket_edges": list(edges),
            "seed": seed, "max_problems": max_problems, "batch_size": batch_size,
            "max_new_tokens": max_new_tokens, "decoding": "greedy", "prompt": "nanochat-direct-answer-v1",
            "dataset_count": len(dataset), "eligible_count": eligible_count,
            "filtered_out_count": len(dataset) - eligible_count,
        },
    })
    return result


def save_longbench_results(path, result):
    """Save aggregates, configuration, and per-example answers atomically."""
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temporary_path, path)
    return path
