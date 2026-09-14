"""Train nanochat GPT or ARMTGPT on the official BABILong training set.

Examples (run from the nanochat repository root):

    # Initialize an ARMTGPT and train it only on BABILong.
    python -m scripts.babilong_train --mode scratch --model-type armt \
        --depth 4 --max-seq-len 8192 --tasks qa1-qa5 --lengths 0k-8k

    # Load a pretrained GPT and continue training it on BABILong.
    python -m scripts.babilong_train --mode finetune --model-type gpt \
        --input-model-tag gpt_d4_with_bl --output-model-tag gpt_d4_babilong

    # Load a pretrained GPT, add ARMT, and finetune the resulting ARMTGPT.
    python -m scripts.babilong_train --mode finetune --model-type armt \
        --input-model-tag gpt_d4_with_bl --output-model-tag armt_d4_babilong

    # Continue a pretrained 2K ARMT checkpoint with 8K BABILong samples.
    # The recurrent segment size remains the checkpoint's original value.
    python -m scripts.babilong_train --mode finetune --model-type armt \
        --input-model-tag armt_d4_2k --max-seq-len 8192 \
        --output-model-tag armt_d4_babilong_8k
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
import wandb

from nanochat.armt import ARMTGPT, ARMTGPTConfig
from nanochat.babilong_data import BABILONG_TRAIN_DATASET, BABILongBatchStream
from nanochat.babilong_eval import (
    evaluate_babilong,
    parse_babilong_lengths,
    parse_babilong_lengths_train,
    parse_babilong_tasks,
)
from nanochat.babilong_generated_data import (
    BABILONG_QA_MARGIN,
    DEFAULT_BABILONG_BABI_PATH,
    DEFAULT_BABILONG_NOISE_DATASET,
    GeneratedBABILongBatchStream,
)
from nanochat.checkpoint_manager import (
    load_checkpoint,
    load_model,
    load_optimizer_state,
    save_checkpoint,
)
from nanochat.common import (
    COMPUTE_DTYPE,
    COMPUTE_DTYPE_REASON,
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    get_peak_flops,
    is_ddp_initialized,
    print0,
)
from nanochat.gpt import GPT, GPTConfig
from nanochat.loss_eval import evaluate_bpb
from nanochat.tokenizer import get_token_bytes, get_tokenizer
from scripts.base_eval import evaluate_core


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GPT/ARMTGPT on BABILong")
    parser.add_argument("--mode", choices=("scratch", "finetune"), required=True)
    parser.add_argument("--model-type", choices=("gpt", "armt"), required=True)
    parser.add_argument("--run", default="dummy", help="wandb name ('dummy' disables wandb)")
    parser.add_argument("--device-type", default="", help="cuda|cpu|mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require deterministic algorithms",
    )
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)

    # Scratch architecture and ARMT wrapping options.
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--aspect-ratio", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--window-pattern", default="SSSL")
    parser.add_argument("--num-mem-tokens", type=int, default=None)
    parser.add_argument("--d-mem", type=int, default=None)
    parser.add_argument("--segment-size", type=int, default=None)
    parser.add_argument(
        "--segment-position-mode", choices=("absolute", "reset"), default="absolute"
    )
    parser.add_argument("--armt-heads", type=int, default=1)
    parser.add_argument("--use-sink", action="store_true")
    parser.add_argument("--no-correction", action="store_true")

    # Parent checkpoint for finetuning. Output remains in base_checkpoints so
    # scripts.base_eval can evaluate it without a separate source type.
    parser.add_argument("--input-source", choices=("base", "sft", "rl"), default="base")
    parser.add_argument("--input-model-tag", default=None)
    parser.add_argument("--input-model-step", type=int, default=None)
    parser.add_argument("--load-optimizer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-model-tag", default=None)

    # Training data backend. ``precomputed`` retains the existing Hub dataset;
    # ``generated`` reproduces the original bAbI + PG-19 online pipeline for
    # train/validation only. Official generation benchmarks remain unchanged.
    parser.add_argument(
        "--train-data-backend",
        choices=("precomputed", "generated"),
        default="precomputed",
    )
    parser.add_argument("--train-dataset", default=BABILONG_TRAIN_DATASET)
    parser.add_argument(
        "--dataset-cache-dir",
        default=None,
        help="datasets cache directory (default: <NANOCHAT_BASE_DIR>/babilong_train_cache)",
    )
    parser.add_argument("--tasks", default="qa1-qa5")
    parser.add_argument("--lengths", default="0k-8k")
    parser.add_argument(
        "--max-seq-len",
        "--sequence-len",
        dest="max_seq_len",
        type=int,
        default=8192,
        help=(
            "training sample length; explicitly overrides a loaded "
            "checkpoint context"
        ),
    )
    parser.add_argument("--device-batch-size", type=int, default=1)
    parser.add_argument(
        "--total-batch-size",
        type=int,
        default=-1,
        help="tokens per optimizer step (-1 = one distributed micro-batch)",
    )
    parser.add_argument("--validation-examples-per-cell", type=int, default=100)
    parser.add_argument("--cell-batches", type=int, default=64)
    parser.add_argument("--oversize-policy", choices=("skip", "error"), default="skip")
    parser.add_argument("--generated-babi-path", default=DEFAULT_BABILONG_BABI_PATH)
    parser.add_argument(
        "--generated-noise-dataset", default=DEFAULT_BABILONG_NOISE_DATASET
    )
    parser.add_argument("--generated-noise-config", default=None)
    parser.add_argument("--generated-noise-train-split", default="train")
    parser.add_argument("--generated-noise-validation-split", default="test")
    parser.add_argument(
        "--generated-vary-lengths",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "sample discrete segment-size multiples through max-seq-len; "
            "otherwise always construct max-seq-len examples"
        ),
    )
    parser.add_argument(
        "--generated-mixed-length-ratio",
        type=float,
        default=0.0,
        help=(
            "reference mixed-length curriculum: probability of forcing the "
            "maximum length (otherwise sample a segment multiple uniformly)"
        ),
    )
    parser.add_argument("--generated-max-n-facts", type=int, default=None)
    parser.add_argument("--generated-task-start-pct", type=float, default=None)
    parser.add_argument("--generated-task-end-pct", type=float, default=None)
    parser.add_argument("--generated-qa-margin", type=int, default=BABILONG_QA_MARGIN)
    parser.add_argument("--generated-min-sentence-tokens", type=int, default=10)

    # Parent learning rates are inherited during finetuning. Weight decay and
    # the initial LR fraction use task-finetuning defaults unless overridden.
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--embedding-lr", type=float, default=None)
    parser.add_argument("--unembedding-lr", type=float, default=None)
    parser.add_argument("--matrix-lr", type=float, default=None)
    parser.add_argument("--scalar-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--init-lr-frac", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--warmdown-ratio", type=float, default=0.5)
    parser.add_argument("--final-lr-frac", type=float, default=0.0)

    # Standard validation, CORE, and BABILong evaluation. Positive intervals
    # also force the corresponding evaluation at the final training step.
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument(
        "--early-stopping-metric",
        choices=("none", "validation_loss", "babilong_metric"),
        default="validation_loss",
        help="criterion to monitor; --early-stopping-patience=-1 disables stopping",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=-1,
        metavar="STEPS",
        help=(
            "stop after this many training steps without monitored-metric "
            "improvement (-1 disables)"
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help=(
            "minimum monitored-metric change that resets patience; for BABILong, "
            "values greater than 1 are percentage points (5 means 0.05)"
        ),
    )
    parser.add_argument("--core-metric-every", type=int, default=2000)
    parser.add_argument("--core-metric-max-per-task", type=int, default=-1)
    parser.add_argument(
        "--babilong-eval-every",
        "--babilong-every",
        dest="babilong_eval_every",
        type=int,
        default=2000,
    )
    parser.add_argument("--eval-dataset", default="RMT-team/babilong")
    parser.add_argument("--eval-tasks", default=None, help="default: same as --tasks")
    parser.add_argument("--eval-lengths", default=None, help="default: same as --lengths")
    parser.add_argument("--eval-max-per-task", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-max-new-tokens", type=int, default=20, choices=range(10, 31))
    parser.add_argument("--save-every", type=int, default=-1)
    return parser


def _resolve_hyperparameters(args, parent_meta: dict | None) -> None:
    parent_user = {} if parent_meta is None else parent_meta.get("user_config", {})
    defaults = {
        "embedding_lr": 0.3,
        "unembedding_lr": 0.008,
        "matrix_lr": 0.02,
        "scalar_lr": 0.5,
        "weight_decay": 0.28 if args.mode == "scratch" else 0.0,
        "init_lr_frac": 1.0 if args.mode == "scratch" else 0.1,
    }
    inherited_names = {"embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr"}
    for name, fallback in defaults.items():
        if getattr(args, name) is not None:
            continue
        inherited = (
            parent_user.get(name)
            if parent_meta is not None and name in inherited_names
            else None
        )
        value = fallback if inherited is None else inherited
        setattr(args, name, value)
        source = "default" if inherited is None else "parent checkpoint"
        print0(f"Resolved {name}={value} from {source}")


def _scratch_model(args, tokenizer, device) -> GPT:
    if args.depth <= 0:
        raise ValueError("depth must be positive")
    base_dim = args.depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    common = dict(
        sequence_len=args.max_seq_len,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=args.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    if args.model_type == "gpt":
        config = GPTConfig(**common)
        model_cls = GPT
    else:
        config = ARMTGPTConfig(
            **common,
            num_mem_tokens=args.num_mem_tokens,
            d_mem=args.d_mem,
            segment_size=args.segment_size,
            segment_position_mode=args.segment_position_mode,
            correction=not args.no_correction,
            n_heads=args.armt_heads,
            use_sink=args.use_sink,
        )
        model_cls = ARMTGPT
    with torch.device("meta"):
        model = model_cls(config)
    model.to_empty(device=device)
    model.init_weights()
    return model


def _load_finetune_model(args, device) -> tuple[GPT, object, dict, bool]:
    model, tokenizer, meta = load_model(
        args.input_source,
        device,
        phase="train",
        model_tag=args.input_model_tag,
        step=args.input_model_step,
    )
    wrapped_gpt = False
    if args.model_type == "gpt" and isinstance(model, ARMTGPT):
        raise ValueError("Cannot finetune an ARMTGPT checkpoint as plain GPT")
    if args.model_type == "armt" and not isinstance(model, ARMTGPT):
        args.num_mem_tokens = 16 if args.num_mem_tokens is None else args.num_mem_tokens
        args.d_mem = 64 if args.d_mem is None else args.d_mem
        args.segment_size = 512 if args.segment_size is None else args.segment_size
        model = ARMTGPT.from_gpt(
            model,
            num_mem_tokens=args.num_mem_tokens,
            d_mem=args.d_mem,
            segment_size=args.segment_size,
            segment_position_mode=args.segment_position_mode,
            correction=not args.no_correction,
            n_heads=args.armt_heads,
            use_sink=args.use_sink,
        )
        wrapped_gpt = True
        print0("Wrapped the pretrained GPT checkpoint with initialized ARMT parameters")
    elif isinstance(model, ARMTGPT):
        for name, requested in (
            ("num_mem_tokens", args.num_mem_tokens),
            ("d_mem", args.d_mem),
            ("segment_size", args.segment_size),
        ):
            actual = getattr(model.config, name)
            if requested is not None and requested != actual:
                raise ValueError(
                    f"Loaded ARMTGPT has {name}={actual}, but {requested} was requested"
                )
            setattr(args, name, actual)
        # These shape/behavior settings belong to the loaded checkpoint and
        # cannot be changed without rebuilding or reinitializing the wrapper.
        args.segment_position_mode = model.config.segment_position_mode
        args.armt_heads = model.config.n_heads
        args.use_sink = model.config.use_sink
        args.no_correction = not model.config.correction
    return model, tokenizer, meta, wrapped_gpt


def _set_training_sequence_len(model: GPT, sequence_len: int) -> None:
    """Apply the data length independently of the checkpoint's saved context."""
    previous = model.config.sequence_len
    model.set_sequence_len(sequence_len)
    if sequence_len != previous:
        print0(
            "Changed runtime sequence length from checkpoint value "
            f"{previous:,} to {sequence_len:,}"
        )


def _lr_multiplier(args, step: int) -> float:
    if args.warmup_steps > 0 and step < args.warmup_steps:
        return (step + 1) / args.warmup_steps
    warmdown_steps = round(args.warmdown_ratio * args.num_iterations)
    if warmdown_steps <= 0 or step <= args.num_iterations - warmdown_steps:
        return 1.0
    progress = max(args.num_iterations - step, 0) / warmdown_steps
    return progress + (1.0 - progress) * args.final_lr_frac


def _muon_momentum(step: int) -> float:
    fraction = min(step / 300, 1.0)
    return 0.85 * (1.0 - fraction) + 0.95 * fraction


def _should_evaluate(
    step: int, last_step: bool, every: int, include_step_zero: bool = False
) -> bool:
    """Return whether a positive evaluation schedule fires, always including final."""
    periodic = every > 0 and (include_step_zero or step > 0) and step % every == 0
    return every > 0 and (last_step or periodic)


def _update_early_stopping(
    val_bpb: float,
    step: int,
    best_bpb: float,
    best_step: int | None,
    patience_steps: int,
    min_delta: float,
    maximize: bool = False,
) -> tuple[float, int, bool, bool]:
    """Update a monitored metric and decide whether step patience is exhausted."""
    if maximize:
        improved = best_step is None or val_bpb > best_bpb + min_delta
    else:
        improved = best_step is None or val_bpb < best_bpb - min_delta
    if improved:
        best_bpb = val_bpb
        best_step = step
    assert best_step is not None
    should_stop = (
        patience_steps >= 0
        and not improved
        and step - best_step >= patience_steps
    )
    return best_bpb, best_step, improved, should_stop


def _update_early_stopping_state(
    metric_value: float,
    step: int,
    best_value: float,
    best_step: int | None,
    patience_value: float,
    patience_step: int | None,
    patience_steps: int,
    min_delta: float,
    maximize: bool = False,
) -> tuple[float, int, float, int, bool, bool, bool]:
    """Track the absolute best separately from the patience reference."""

    def improved_over(value: float, reference: float, delta: float) -> bool:
        if maximize:
            return value > reference + delta
        return value < reference - delta

    is_best = best_step is None or improved_over(metric_value, best_value, 0.0)
    if is_best:
        best_value = metric_value
        best_step = step

    significant_improvement = (
        patience_step is None
        or improved_over(metric_value, patience_value, min_delta)
    )
    if significant_improvement:
        patience_value = metric_value
        patience_step = step

    assert best_step is not None
    assert patience_step is not None
    should_stop = (
        patience_steps >= 0
        and not significant_improvement
        and step - patience_step >= patience_steps
    )
    return (
        best_value,
        best_step,
        patience_value,
        patience_step,
        is_best,
        significant_improvement,
        should_stop,
    )


def _resolve_early_stopping_min_delta(metric: str, min_delta: float) -> float:
    """Normalize BABILong percentage-point tolerances to its [0, 1] scale."""
    if min_delta < 0:
        raise ValueError("--early-stopping-min-delta must be non-negative")
    if metric == "babilong_metric" and min_delta > 1.0:
        if min_delta > 100.0:
            raise ValueError(
                "--early-stopping-min-delta cannot exceed 100 percentage points "
                "for BABILong"
            )
        return min_delta / 100.0
    return min_delta


def _restore_training_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str,
    device: torch.device,
    rank: int,
) -> int:
    """Restore model and optimizer state and return its actual training step."""
    model_data, optimizer_data, meta = load_checkpoint(
        checkpoint_dir,
        0,
        device,
        load_optimizer=True,
        rank=rank,
    )
    model_data = {
        key.removeprefix("_orig_mod."): value for key, value in model_data.items()
    }
    model.load_state_dict(model_data, strict=True)
    if optimizer_data is None:
        raise RuntimeError(f"Best optimizer snapshot is missing in {checkpoint_dir}")
    optimizer.load_state_dict(optimizer_data)
    return int(meta["step"])


def _build_data_stream(
    args,
    tokenizer,
    device,
    tasks,
    lengths,
    split: str,
    rank: int,
    world_size: int,
):
    if args.train_data_backend == "precomputed":
        cell_batches = args.cell_batches
        if split == "validation":
            cell_batches = max(1, math.ceil(args.eval_batches / len(lengths)))
        return BABILongBatchStream(
            tokenizer,
            args.device_batch_size,
            args.max_seq_len,
            device,
            tasks=tasks,
            lengths=lengths,
            dataset_name=args.train_dataset,
            split=split,
            validation_examples=args.validation_examples_per_cell,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
            cell_batches=cell_batches,
            oversize_policy=args.oversize_policy,
            cache_dir=args.dataset_cache_dir,
        )

    generated_segment_size = args.segment_size
    if generated_segment_size is None:
        generated_segment_size = min(1024, args.max_seq_len)
    return GeneratedBABILongBatchStream(
        tokenizer,
        args.device_batch_size,
        args.max_seq_len,
        generated_segment_size,
        device,
        tasks=tasks,
        babi_path=args.generated_babi_path,
        split=split,
        noise_dataset_name=args.generated_noise_dataset,
        noise_config_name=args.generated_noise_config,
        noise_train_split=args.generated_noise_train_split,
        noise_validation_split=args.generated_noise_validation_split,
        vary_lengths=args.generated_vary_lengths and split == "train",
        mixed_length_ratio=args.generated_mixed_length_ratio,
        max_n_facts=args.generated_max_n_facts,
        task_start_pct=args.generated_task_start_pct,
        task_end_pct=args.generated_task_end_pct,
        qa_margin=args.generated_qa_margin,
        min_sentence_len=args.generated_min_sentence_tokens,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
        oversize_policy=args.oversize_policy,
        cache_dir=args.dataset_cache_dir,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.num_iterations <= 0:
        parser.error("--num-iterations must be positive")
    if args.max_seq_len <= 1:
        parser.error("--max-seq-len must be greater than one")
    if args.device_batch_size <= 0:
        parser.error("--device-batch-size must be positive")
    if args.total_batch_size == 0 or args.total_batch_size < -1:
        parser.error("--total-batch-size must be -1 or a positive integer")
    if args.eval_every > 0 and args.eval_batches <= 0:
        parser.error("--eval-batches must be positive when validation is enabled")
    if args.early_stopping_patience < -1:
        parser.error("--early-stopping-patience must be -1 or non-negative")
    requested_early_stopping_min_delta = args.early_stopping_min_delta
    try:
        effective_early_stopping_min_delta = _resolve_early_stopping_min_delta(
            args.early_stopping_metric, requested_early_stopping_min_delta
        )
    except ValueError as exc:
        parser.error(str(exc))
    early_stopping_enabled = (
        args.early_stopping_metric != "none"
        and args.early_stopping_patience >= 0
    )
    if (
        early_stopping_enabled
        and args.early_stopping_metric == "validation_loss"
        and args.eval_every <= 0
    ):
        parser.error(
            "--eval-every must be positive for validation-loss early stopping"
        )
    if (
        early_stopping_enabled
        and args.early_stopping_metric == "babilong_metric"
        and args.babilong_eval_every <= 0
    ):
        parser.error("--babilong-eval-every must be positive for BABILong early stopping")
    if not 0.0 <= args.warmdown_ratio <= 1.0:
        parser.error("--warmdown-ratio must be between zero and one")
    if not 0.0 <= args.generated_mixed_length_ratio <= 1.0:
        parser.error("--generated-mixed-length-ratio must be between zero and one")
    if args.generated_qa_margin <= 0:
        parser.error("--generated-qa-margin must be positive")
    if args.generated_min_sentence_tokens < 0:
        parser.error("--generated-min-sentence-tokens must be non-negative")
    placement = (
        args.generated_task_start_pct,
        args.generated_task_end_pct,
    )
    if (placement[0] is None) != (placement[1] is None):
        parser.error("generated task start/end percentages must be set together")
    if placement[0] is not None and not 0.0 <= placement[0] < placement[1] <= 1.0:
        parser.error(
            "generated task percentages must satisfy 0 <= start < end <= 1"
        )
    try:
        train_tasks = parse_babilong_tasks(args.tasks)
        train_lengths = parse_babilong_lengths_train(args.lengths)
        eval_tasks = parse_babilong_tasks(args.eval_tasks or train_tasks)
        eval_lengths = parse_babilong_lengths(args.eval_lengths or train_lengths)
    except ValueError as exc:
        parser.error(str(exc))
    if (
        early_stopping_enabled
        and args.early_stopping_metric == "babilong_metric"
        and (
            tuple(eval_tasks) != tuple(train_tasks)
            or (
                args.train_data_backend == "precomputed"
                and tuple(eval_lengths) != tuple(train_lengths)
            )
        )
    ):
        parser.error(
            "BABILong early stopping requires eval tasks to match training; "
            "precomputed training also requires matching lengths"
        )

    device_type = autodetect_device_type() if not args.device_type else args.device_type
    ddp, rank, local_rank, world_size, device = compute_init(
        device_type, seed=args.seed, deterministic=args.deterministic
    )
    master_process = rank == 0
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
    get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
    if device_type == "cuda":
        peak_flops = get_peak_flops(torch.cuda.get_device_name(local_rank))
    else:
        peak_flops = float("inf")
    print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
    if effective_early_stopping_min_delta != requested_early_stopping_min_delta:
        print0(
            "Interpreting BABILong early-stopping min delta "
            f"{requested_early_stopping_min_delta:g} as "
            f"{effective_early_stopping_min_delta:g} "
            f"({requested_early_stopping_min_delta:g} percentage points)"
        )

    parent_meta = None
    wrapped_gpt = False
    if args.mode == "scratch":
        args.num_mem_tokens = 16 if args.num_mem_tokens is None else args.num_mem_tokens
        args.d_mem = 64 if args.d_mem is None else args.d_mem
        args.segment_size = 512 if args.segment_size is None else args.segment_size
        tokenizer = get_tokenizer()
        model = _scratch_model(args, tokenizer, device)
    else:
        model, tokenizer, parent_meta, wrapped_gpt = _load_finetune_model(args, device)
    _resolve_hyperparameters(args, parent_meta)
    _set_training_sequence_len(model, args.max_seq_len)

    if args.output_model_tag is None:
        args.output_model_tag = f"babilong_{args.model_type}_{args.mode}"
    user_config = vars(args).copy()
    model_config = asdict(model.config)
    print0(f"Model config:\n{json.dumps(model_config, indent=2)}")
    checkpoint_dir = os.path.join(
        get_base_dir(), "base_checkpoints", args.output_model_tag
    )
    best_checkpoint_dir = os.path.join(checkpoint_dir, "best")

    use_dummy_wandb = args.run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
        project="nanochat-babilong", name=args.run, config=user_config
    )

    orig_model = model
    compile_dynamic = (
        args.train_data_backend == "generated" and args.generated_vary_lengths
    )
    model = torch.compile(model, dynamic=compile_dynamic) if args.compile else model
    num_flops_per_token = orig_model.estimate_flops()
    world_tokens_per_microbatch = args.device_batch_size * args.max_seq_len * world_size
    total_batch_size = (
        world_tokens_per_microbatch if args.total_batch_size == -1 else args.total_batch_size
    )
    if total_batch_size % world_tokens_per_microbatch:
        raise ValueError(
            f"total_batch_size ({total_batch_size}) must be divisible by the distributed "
            f"micro-batch ({world_tokens_per_microbatch})"
        )
    grad_accum_steps = total_batch_size // world_tokens_per_microbatch
    print0(
        f"Batch: {args.device_batch_size} x {args.max_seq_len} x {world_size} ranks x "
        f"{grad_accum_steps} accumulation = {total_batch_size:,} tokens"
    )

    optimizer = orig_model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        scalar_lr=args.scalar_lr,
        weight_decay=args.weight_decay,
    )
    if args.load_optimizer:
        if args.mode != "finetune":
            raise ValueError("--load-optimizer is only valid in finetune mode")
        if wrapped_gpt:
            raise ValueError("Cannot load a GPT optimizer after adding ARMT parameters")
        optimizer_data = load_optimizer_state(
            args.input_source,
            device,
            rank=rank,
            model_tag=args.input_model_tag,
            step=args.input_model_step,
        )
        if optimizer_data is not None:
            fresh_lrs = [group["lr"] for group in optimizer.param_groups]
            optimizer.load_state_dict(optimizer_data)
            for group, fresh_lr in zip(optimizer.param_groups, fresh_lrs):
                group["lr"] = fresh_lr
            print0("Loaded parent optimizer state and reset learning rates")
    for group in optimizer.param_groups:
        group["lr"] *= args.init_lr_frac
        group["initial_lr"] = group["lr"]

    scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
    train_stream = _build_data_stream(
        args,
        tokenizer,
        device,
        train_tasks,
        train_lengths,
        split="train",
        rank=rank,
        world_size=world_size,
    )

    def build_val_stream():
        return _build_data_stream(
            args,
            tokenizer,
            device,
            train_tasks,
            train_lengths,
            split="validation",
            rank=rank,
            world_size=world_size,
        )

    training_lengths = (
        list(train_lengths)
        if args.train_data_backend == "precomputed"
        else list(train_stream.sample_sizes)
    )
    training_dataset = (
        args.train_dataset
        if args.train_data_backend == "precomputed"
        else {
            "babi_path": os.path.abspath(os.path.expanduser(args.generated_babi_path)),
            "noise_dataset": args.generated_noise_dataset,
            "noise_config": args.generated_noise_config,
        }
    )
    token_bytes = get_token_bytes(device=device)
    x, y = next(train_stream)

    step = 0
    smooth_loss = 0.0
    total_training_time = 0.0
    total_training_tokens = 0
    best_validation_bpb = float("inf")
    best_validation_step = None
    validation_patience_bpb = float("inf")
    validation_patience_step = None
    best_babilong_metric = float("-inf")
    best_babilong_step = None
    babilong_patience_metric = float("-inf")
    babilong_patience_step = None
    early_stopped = False
    restored_best_weights = False
    restored_best_step = None
    latest_validation = None
    latest_core = None
    latest_babilong = None
    while True:
        last_step = early_stopped or step == args.num_iterations
        save_best_snapshot = False
        best_snapshot_value = None
        finalize_after_babilong_stop = False
        flops_so_far = num_flops_per_token * total_training_tokens

        if (
            _should_evaluate(
                step, last_step, args.eval_every, include_step_zero=True
            )
            and (
                latest_validation is None
                or latest_validation["evaluated_at_step"] != step
            )
        ):
            model.eval()
            eval_started = time.time()
            val_bpb = evaluate_bpb(
                model, build_val_stream(), args.eval_batches, token_bytes
            )
            (
                best_validation_bpb,
                best_validation_step,
                validation_patience_bpb,
                validation_patience_step,
                validation_is_best,
                validation_significant_improvement,
                patience_exhausted,
            ) = _update_early_stopping_state(
                val_bpb,
                step,
                best_validation_bpb,
                best_validation_step,
                validation_patience_bpb,
                validation_patience_step,
                (
                    args.early_stopping_patience
                    if early_stopping_enabled
                    and args.early_stopping_metric == "validation_loss"
                    else -1
                ),
                effective_early_stopping_min_delta,
            )
            steps_since_improvement = step - validation_patience_step
            if (
                validation_is_best
                and early_stopping_enabled
                and args.early_stopping_metric == "validation_loss"
            ):
                save_best_snapshot = True
                best_snapshot_value = val_bpb
            stop_for_patience = patience_exhausted and not last_step
            if stop_for_patience:
                early_stopped = True
                last_step = True
            latest_validation = {
                "evaluated_at_step": step,
                "scores": {"answer_bpb": val_bpb},
                "is_best": validation_is_best,
                "significant_improvement": validation_significant_improvement,
                "best_answer_bpb": best_validation_bpb,
                "best_step": best_validation_step,
                "patience_reference_bpb": validation_patience_bpb,
                "patience_reference_step": validation_patience_step,
                "steps_since_improvement": steps_since_improvement,
                "early_stopping_triggered": stop_for_patience,
                "batches": args.eval_batches,
                "tasks": list(train_tasks),
                "lengths": training_lengths,
                "eval_time_seconds": time.time() - eval_started,
            }
            print0(f"Step {step:05d} | BABILong validation answer bpb: {val_bpb:.6f}")
            wandb_run.log({
                "step": step,
                "validation/answer_bpb": val_bpb,
                "validation/best_answer_bpb": best_validation_bpb,
                "validation/steps_since_improvement": steps_since_improvement,
            })
            if stop_for_patience:
                print0(
                    f"Early stopping at step {step}: validation BPB has not "
                    f"improved for {steps_since_improvement} steps"
                )
            model.train()

        run_core_eval = (
            _should_evaluate(step, last_step, args.core_metric_every)
            and (
                latest_core is None
                or latest_core["evaluated_at_step"] != step
            )
        )
        if run_core_eval:
            model.eval()
            core_result = evaluate_core(
                orig_model,
                tokenizer,
                device,
                max_per_task=args.core_metric_max_per_task,
            )
            latest_core = {
                "evaluated_at_step": step,
                "per_task_scores": core_result["results"],
                "per_task_centered_scores": core_result["centered_results"],
                "aggregated_scores": {
                    "core_metric": core_result["core_metric"],
                },
                "eval_time": core_result["eval_time"],
                "seeds": core_result["seeds"],
                "max_per_task": args.core_metric_max_per_task,
            }
            print0(f"Step {step:05d} | CORE metric: {core_result['core_metric']:.4f}")
            wandb_run.log({
                "step": step,
                "core_metric": core_result["core_metric"],
                "core/centered_results": core_result["centered_results"],
            })
            model.train()

        run_generation_eval = (
            _should_evaluate(
                step,
                last_step,
                args.babilong_eval_every,
                include_step_zero=(
                    early_stopping_enabled
                    and args.early_stopping_metric == "babilong_metric"
                ),
            )
            and (
                latest_babilong is None
                or latest_babilong["evaluated_at_step"] != step
            )
        )
        if run_generation_eval:
            model.eval()
            result = evaluate_babilong(
                orig_model,
                tokenizer,
                device,
                tasks=eval_tasks,
                lengths=eval_lengths,
                max_per_task=args.eval_max_per_task,
                max_new_tokens=args.eval_max_new_tokens,
                dataset_name=args.eval_dataset,
                batch_size=args.eval_batch_size,
            )
            (
                best_babilong_metric,
                best_babilong_step,
                babilong_patience_metric,
                babilong_patience_step,
                babilong_is_best,
                babilong_significant_improvement,
                patience_exhausted,
            ) = _update_early_stopping_state(
                result["babilong_metric"],
                step,
                best_babilong_metric,
                best_babilong_step,
                babilong_patience_metric,
                babilong_patience_step,
                (
                    args.early_stopping_patience
                    if early_stopping_enabled
                    and args.early_stopping_metric == "babilong_metric"
                    else -1
                ),
                effective_early_stopping_min_delta,
                maximize=True,
            )
            babilong_steps_since_improvement = step - babilong_patience_step
            if (
                babilong_is_best
                and early_stopping_enabled
                and args.early_stopping_metric == "babilong_metric"
            ):
                save_best_snapshot = True
                best_snapshot_value = result["babilong_metric"]
            stop_for_babilong_patience = patience_exhausted and not last_step
            if stop_for_babilong_patience:
                early_stopped = True
                last_step = True
            latest_babilong = {
                "evaluated_at_step": step,
                "is_best": babilong_is_best,
                "significant_improvement": babilong_significant_improvement,
                "best_metric": best_babilong_metric,
                "best_step": best_babilong_step,
                "patience_reference_metric": babilong_patience_metric,
                "patience_reference_step": babilong_patience_step,
                "steps_since_improvement": babilong_steps_since_improvement,
                "early_stopping_triggered": stop_for_babilong_patience,
                "per_task_per_length_scores": result["results"],
                "aggregated_scores": {
                    "per_task": result["task_results"],
                    "per_length": result["length_results"],
                    "overall": result["babilong_metric"],
                },
                "eval_time": result["eval_time"],
                "seeds": result["seeds"],
                "example_counts": result["example_counts"],
                "tasks": result["tasks"],
                "lengths": result["lengths"],
                "batch_size": result["batch_size"],
                "max_per_task": args.eval_max_per_task,
                "max_new_tokens": args.eval_max_new_tokens,
                "dataset": args.eval_dataset,
            }
            print0(f"Step {step:05d} | BABILong metric: {result['babilong_metric']:.4f}")
            babilong_log = {
                f"babilong/{task}/{length}": score
                for task, length_scores in result["results"].items()
                for length, score in length_scores.items()
            }
            babilong_log.update({
                f"babilong_length/{length}": score
                for length, score in result["length_results"].items()
            })
            babilong_log.update({
                f"babilong_task/{task}": score
                for task, score in result["task_results"].items()
            })
            wandb_run.log({
                "step": step,
                "babilong_metric": result["babilong_metric"],
                "babilong/best_metric": best_babilong_metric,
                "babilong/steps_since_improvement": babilong_steps_since_improvement,
                **babilong_log,
            })
            if stop_for_babilong_patience:
                print0(
                    f"Early stopping at step {step}: BABILong metric has not "
                    f"improved for {babilong_steps_since_improvement} steps"
                )
            model.train()
            if stop_for_babilong_patience:
                finalize_after_babilong_stop = True

        if save_best_snapshot:
            best_snapshot_eval = {
                "training_step": step,
                "criterion": args.early_stopping_metric,
                "value": best_snapshot_value,
                "requested_min_delta": requested_early_stopping_min_delta,
                "effective_min_delta": effective_early_stopping_min_delta,
            }
            save_checkpoint(
                best_checkpoint_dir,
                0,
                orig_model.state_dict(),
                optimizer.state_dict(),
                {
                    # The file uses a fixed slot so only one large best snapshot
                    # is retained. This field records its actual training step.
                    "step": step,
                    "model_config": model_config,
                    "user_config": user_config,
                    "device_batch_size": args.device_batch_size,
                    "max_seq_len": args.max_seq_len,
                    "total_batch_size": total_batch_size,
                    "best_snapshot": best_snapshot_eval,
                },
                rank=rank,
                eval_data=best_snapshot_eval,
            )
            if is_ddp_initialized():
                dist.barrier()
            print0(
                f"Saved new best {args.early_stopping_metric} checkpoint "
                f"from step {step} to {best_checkpoint_dir}"
            )

        if finalize_after_babilong_stop:
            # Re-enter once to run any final evaluations skipped earlier this
            # step. The new best snapshot above must be saved before doing so.
            continue

        save_now = last_step or (
            step > 0 and args.save_every > 0 and step % args.save_every == 0
        )
        if save_now:
            selected_best_step_for_restore = (
                best_babilong_step
                if args.early_stopping_metric == "babilong_metric"
                else best_validation_step
            )
            if (
                last_step
                and early_stopping_enabled
                and selected_best_step_for_restore is not None
                and not restored_best_weights
            ):
                if is_ddp_initialized():
                    dist.barrier()
                restored_best_step = _restore_training_checkpoint(
                    orig_model,
                    optimizer,
                    best_checkpoint_dir,
                    device,
                    rank,
                )
                restored_best_weights = True
                print0(
                    f"Restored best {args.early_stopping_metric} weights "
                    f"from step {restored_best_step} for the final checkpoint"
                )
            peak_memory = int(get_max_memory())
            if early_stopped:
                stop_reason = "early_stopping"
            else:
                stop_reason = "iteration_limit" if last_step else None
            if args.early_stopping_metric == "babilong_metric":
                monitored_metric = "babilong/metric"
                monitored_mode = "max"
                selected_best_value = (
                    best_babilong_metric
                    if best_babilong_step is not None
                    else None
                )
                selected_best_step = best_babilong_step
                selected_patience_value = babilong_patience_metric
                selected_patience_step = babilong_patience_step
            elif args.early_stopping_metric == "validation_loss":
                monitored_metric = "validation/answer_bpb"
                monitored_mode = "min"
                selected_best_value = (
                    best_validation_bpb
                    if best_validation_step is not None
                    else None
                )
                selected_best_step = best_validation_step
                selected_patience_value = validation_patience_bpb
                selected_patience_step = validation_patience_step
            else:
                monitored_metric = None
                monitored_mode = None
                selected_best_value = None
                selected_best_step = None
                selected_patience_value = None
                selected_patience_step = None
            evaluation = {
                "checkpoint_step": step,
                "checkpoint_weights_step": restored_best_step if restored_best_weights else step,
                "reproducibility": {
                    "seed": args.seed,
                    "deterministic_algorithms": args.deterministic,
                    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                },
                "training": {
                    "mode": args.mode,
                    "data_backend": args.train_data_backend,
                    "dataset": training_dataset,
                    "tasks": list(train_tasks),
                    "lengths": training_lengths,
                    "requested_iterations": args.num_iterations,
                    "completed_iterations": step,
                    "stop_reason": stop_reason,
                    "total_time_seconds": total_training_time,
                    "total_tokens": total_training_tokens,
                    "total_flops": flops_so_far,
                    "peak_memory_bytes": peak_memory,
                    "peak_memory_mib": peak_memory / (1024 * 1024),
                    "data_stream": train_stream.stats(),
                },
                "early_stopping": {
                    "enabled": early_stopping_enabled,
                    "criterion": args.early_stopping_metric,
                    "patience_steps": args.early_stopping_patience,
                    "min_delta": effective_early_stopping_min_delta,
                    "requested_min_delta": requested_early_stopping_min_delta,
                    "mode": monitored_mode,
                    "monitored_metric": monitored_metric,
                    "best_value": selected_best_value,
                    "best_step": selected_best_step,
                    "patience_reference_value": selected_patience_value,
                    "patience_reference_step": selected_patience_step,
                    "steps_since_improvement": (
                        None
                        if selected_patience_step is None
                        else step - selected_patience_step
                    ),
                    "restored_best_weights": restored_best_weights,
                    "restored_best_step": restored_best_step,
                    "best_snapshot_directory": "best" if early_stopping_enabled else None,
                    "best_validation_bpb": (
                        best_validation_bpb
                        if best_validation_step is not None
                        else None
                    ),
                    "best_validation_step": best_validation_step,
                    "best_babilong_metric": (
                        best_babilong_metric
                        if best_babilong_step is not None
                        else None
                    ),
                    "best_babilong_step": best_babilong_step,
                    "stopped_early": early_stopped,
                    "stop_step": step if early_stopped else None,
                },
                "validation": latest_validation,
                "core": latest_core,
                "babilong": latest_babilong,
            }
            parent = None
            if args.mode == "finetune":
                parent = {
                    "source": args.input_source,
                    "model_tag": args.input_model_tag,
                    "step": parent_meta.get("step") if parent_meta else args.input_model_step,
                    "wrapped_gpt_with_armt": wrapped_gpt,
                }
            save_checkpoint(
                checkpoint_dir,
                step,
                orig_model.state_dict(),
                optimizer.state_dict(),
                {
                    "step": step,
                    "checkpoint_weights_step": restored_best_step if restored_best_weights else step,
                    "model_config": model_config,
                    "user_config": user_config,
                    "device_batch_size": args.device_batch_size,
                    "max_seq_len": args.max_seq_len,
                    "total_batch_size": total_batch_size,
                    "parent_checkpoint": parent,
                    "evaluation": evaluation,
                },
                rank=rank,
                eval_data=evaluation,
            )
        if last_step:
            break

        synchronize()
        started = time.time()
        step_tokens = 0
        for _ in range(grad_accum_steps):
            step_tokens += x.numel() * world_size
            loss = model(x, y)
            train_loss = loss.detach()
            scaled_loss = loss / grad_accum_steps
            if scaler is None:
                scaled_loss.backward()
            else:
                scaler.scale(scaled_loss).backward()
            x, y = next(train_stream)

        lr_multiplier = _lr_multiplier(args, step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lr_multiplier
            if group["kind"] == "muon":
                group["momentum"] = _muon_momentum(step)
        if scaler is None:
            optimizer.step()
        else:
            scaler.unscale_(optimizer)
            if is_ddp_initialized():
                for found_inf in scaler._found_inf_per_device(optimizer).values():
                    dist.all_reduce(found_inf, op=dist.ReduceOp.MAX)
            scaler.step(optimizer)
            scaler.update()
        model.zero_grad(set_to_none=True)
        synchronize()
        elapsed = time.time() - started
        if step > 10:
            total_training_time += elapsed
        total_training_tokens += step_tokens

        loss_value = train_loss.item()
        smooth_loss = 0.9 * smooth_loss + 0.1 * loss_value
        debiased_loss = smooth_loss / (1.0 - 0.9 ** (step + 1))
        tokens_per_second = int(step_tokens / elapsed)
        mfu = 100 * num_flops_per_token * step_tokens / (
            elapsed * peak_flops * world_size
        )
        print0(
            f"step {step:05d}/{args.num_iterations:05d} | loss: {debiased_loss:.6f} | "
            f"lrm: {lr_multiplier:.3f} | dt: {elapsed * 1000:.1f}ms | "
            f"tok/sec: {tokens_per_second:,} | mfu: {mfu:.2f}%"
        )
        if step % 20 == 0:
            wandb_run.log(
                {
                    "step": step,
                    "train/loss": debiased_loss,
                    "train/lrm": lr_multiplier,
                    "train/tokens_per_second": tokens_per_second,
                    "train/mfu": mfu,
                    "training_time": total_training_time,
                }
            )
        step += 1
        if step == 1:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif step % 5000 == 0:
            gc.collect()

    if early_stopped:
        if args.early_stopping_metric == "babilong_metric":
            print0(
                f"Best BABILong metric: {best_babilong_metric:.6f} at step {best_babilong_step}"
            )
        else:
            print0(f"Best validation BPB: {best_validation_bpb:.6f} at step {best_validation_step}")
    print0(f"Saved BABILong-trained checkpoint to {checkpoint_dir}")
    print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    print0(f"Total training time: {total_training_time / 60:.2f}m")
    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
