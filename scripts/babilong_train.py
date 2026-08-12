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
from nanochat.checkpoint_manager import load_model, load_optimizer_state, save_checkpoint
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

    # Official pre-generated BABILong training mixture.
    parser.add_argument("--train-dataset", default=BABILONG_TRAIN_DATASET)
    parser.add_argument(
        "--dataset-cache-dir",
        default=None,
        help="datasets cache directory (default: <NANOCHAT_BASE_DIR>/babilong_train_cache)",
    )
    parser.add_argument("--tasks", default="qa1-qa5")
    parser.add_argument("--lengths", default="0k-8k")
    parser.add_argument("--max-seq-len", type=int, default=8192)
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
        "--early-stopping-patience",
        type=int,
        default=-1,
        metavar="STEPS",
        help=(
            "stop after this many training steps without validation BPB "
            "improvement (-1 disables)"
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="minimum validation BPB decrease counted as an improvement",
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


def _ensure_rotary_capacity(model: GPT, real_tokens: int) -> None:
    required = real_tokens
    if isinstance(model, ARMTGPT):
        extras = model.config.num_mem_tokens + int(model.config.use_sink)
        if model.config.segment_position_mode == "reset":
            required = min(real_tokens, model.config.segment_size) + extras
        else:
            required = real_tokens + math.ceil(real_tokens / model.config.segment_size) * extras
    if required <= model.cos.size(1):
        return
    new_size = max(required, model.cos.size(1) * 2)
    head_dim = model.config.n_embd // model.config.n_head
    cos, sin = model._precompute_rotary_embeddings(new_size, head_dim)
    model.cos, model.sin = cos, sin
    model.rotary_seq_len = new_size
    print0(f"Extended rotary cache to {new_size:,} positions for training")


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
) -> tuple[float, int, bool, bool]:
    """Update best validation state and decide whether patience is exhausted."""
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
    if args.early_stopping_min_delta < 0:
        parser.error("--early-stopping-min-delta must be non-negative")
    if args.early_stopping_patience >= 0 and args.eval_every <= 0:
        parser.error(
            "--eval-every must be positive when early stopping is enabled"
        )
    if not 0.0 <= args.warmdown_ratio <= 1.0:
        parser.error("--warmdown-ratio must be between zero and one")
    try:
        train_tasks = parse_babilong_tasks(args.tasks)
        train_lengths = parse_babilong_lengths_train(args.lengths)
        eval_tasks = parse_babilong_tasks(args.eval_tasks or train_tasks)
        eval_lengths = parse_babilong_lengths(args.eval_lengths or train_lengths)
    except ValueError as exc:
        parser.error(str(exc))

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
    _ensure_rotary_capacity(model, args.max_seq_len)

    if args.output_model_tag is None:
        args.output_model_tag = f"babilong_{args.model_type}_{args.mode}"
    user_config = vars(args).copy()
    model_config = asdict(model.config)
    print0(f"Model config:\n{json.dumps(model_config, indent=2)}")
    checkpoint_dir = os.path.join(
        get_base_dir(), "base_checkpoints", args.output_model_tag
    )

    use_dummy_wandb = args.run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
        project="nanochat-babilong", name=args.run, config=user_config
    )

    orig_model = model
    model = torch.compile(model, dynamic=False) if args.compile else model
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
    train_stream = BABILongBatchStream(
        tokenizer,
        args.device_batch_size,
        args.max_seq_len,
        device,
        tasks=train_tasks,
        lengths=train_lengths,
        dataset_name=args.train_dataset,
        split="train",
        validation_examples=args.validation_examples_per_cell,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
        cell_batches=args.cell_batches,
        oversize_policy=args.oversize_policy,
        cache_dir=args.dataset_cache_dir,
    )

    def build_val_stream():
        return BABILongBatchStream(
            tokenizer,
            args.device_batch_size,
            args.max_seq_len,
            device,
            tasks=train_tasks,
            lengths=train_lengths,
            dataset_name=args.train_dataset,
            split="validation",
            validation_examples=args.validation_examples_per_cell,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
            cell_batches=max(1, math.ceil(args.eval_batches / len(train_lengths))),
            oversize_policy=args.oversize_policy,
            cache_dir=args.dataset_cache_dir,
        )

    token_bytes = get_token_bytes(device=device)
    x, y = next(train_stream)

    step = 0
    smooth_loss = 0.0
    total_training_time = 0.0
    best_validation_bpb = float("inf")
    best_validation_step = None
    early_stopped = False
    latest_validation = None
    latest_core = None
    latest_babilong = None
    while True:
        last_step = step == args.num_iterations
        flops_so_far = num_flops_per_token * total_batch_size * step

        if _should_evaluate(
            step, last_step, args.eval_every, include_step_zero=True
        ):
            model.eval()
            eval_started = time.time()
            val_bpb = evaluate_bpb(
                model, build_val_stream(), args.eval_batches, token_bytes
            )
            (
                best_validation_bpb,
                best_validation_step,
                validation_improved,
                patience_exhausted,
            ) = _update_early_stopping(
                val_bpb,
                step,
                best_validation_bpb,
                best_validation_step,
                args.early_stopping_patience,
                args.early_stopping_min_delta,
            )
            steps_since_improvement = step - best_validation_step
            stop_for_patience = patience_exhausted and not last_step
            if stop_for_patience:
                early_stopped = True
                last_step = True
            latest_validation = {
                "evaluated_at_step": step,
                "scores": {"answer_bpb": val_bpb},
                "is_best": validation_improved,
                "best_answer_bpb": best_validation_bpb,
                "best_step": best_validation_step,
                "steps_since_improvement": steps_since_improvement,
                "early_stopping_triggered": stop_for_patience,
                "batches": args.eval_batches,
                "tasks": list(train_tasks),
                "lengths": list(train_lengths),
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

        run_core_eval = _should_evaluate(step, last_step, args.core_metric_every)
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

        run_generation_eval = _should_evaluate(step, last_step, args.babilong_eval_every)
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
            latest_babilong = {
                "evaluated_at_step": step,
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
                **babilong_log,
            })
            model.train()

        save_now = last_step or (
            step > 0 and args.save_every > 0 and step % args.save_every == 0
        )
        if save_now:
            peak_memory = int(get_max_memory())
            if early_stopped:
                stop_reason = "early_stopping"
            else:
                stop_reason = "iteration_limit" if last_step else None
            evaluation = {
                "checkpoint_step": step,
                "reproducibility": {
                    "seed": args.seed,
                    "deterministic_algorithms": args.deterministic,
                    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                },
                "training": {
                    "mode": args.mode,
                    "dataset": args.train_dataset,
                    "tasks": list(train_tasks),
                    "lengths": list(train_lengths),
                    "requested_iterations": args.num_iterations,
                    "completed_iterations": step,
                    "stop_reason": stop_reason,
                    "total_time_seconds": total_training_time,
                    "total_flops": flops_so_far,
                    "peak_memory_bytes": peak_memory,
                    "peak_memory_mib": peak_memory / (1024 * 1024),
                    "data_stream": train_stream.stats(),
                },
                "early_stopping": {
                    "enabled": args.early_stopping_patience >= 0,
                    "patience_steps": args.early_stopping_patience,
                    "min_delta": args.early_stopping_min_delta,
                    "monitored_metric": "validation/answer_bpb",
                    "best_validation_bpb": (
                        best_validation_bpb
                        if best_validation_step is not None
                        else None
                    ),
                    "best_step": best_validation_step,
                    "steps_since_improvement": (
                        None
                        if best_validation_step is None
                        else step - best_validation_step
                    ),
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
        for _ in range(grad_accum_steps):
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

        loss_value = train_loss.item()
        smooth_loss = 0.9 * smooth_loss + 0.1 * loss_value
        debiased_loss = smooth_loss / (1.0 - 0.9 ** (step + 1))
        tokens_per_second = int(total_batch_size / elapsed)
        mfu = 100 * num_flops_per_token * total_batch_size / (
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
        print0(
            f"Best validation BPB: {best_validation_bpb:.6f} at step {best_validation_step}"
        )
    print0(f"Saved BABILong-trained checkpoint to {checkpoint_dir}")
    print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    print0(f"Total training time: {total_training_time / 60:.2f}m")
    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
