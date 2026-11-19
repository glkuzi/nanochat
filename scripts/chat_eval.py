"""
Evaluate the Chat model.
All the generic code lives here, and all the evaluation-specific
code lives in nanochat directory and is imported from here.

Example runs:
python -m scripts.chat_eval -i sft -a ARC-Easy --eval chat
python -m scripts.chat_eval -i sft --eval babilong --babilong-tasks qa1-qa5
python -m scripts.chat_eval -i sft --eval longbench-v2 --longbench-v2-max-length 32k
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -i sft -a ARC-Easy --eval chat
"""

import argparse
import csv
from functools import partial
import os
import re
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, get_dist_info, print0, autodetect_device_type
from nanochat.babilong_eval import (
    BABILONG_DATASET,
    DEFAULT_BABILONG_BATCH_SIZE,
    evaluate_babilong,
    parse_babilong_lengths,
    parse_babilong_tasks,
)
from nanochat.checkpoint_manager import load_model
from nanochat.common import get_base_dir
from nanochat.engine import Engine
from nanochat.longbench_v2_eval import (
    DEFAULT_LENGTH_BUCKETS,
    LONGBENCH_V2_DATASET,
    evaluate_longbench_v2,
    parse_length,
    parse_length_buckets,
    save_longbench_results,
)

from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# Generative evaluation loop (we go one problem at a time, sample, evaluate)

def run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # Run the evaluation
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]

        # Tokenize the prompt
        encoded_prompt = tokenizer.render_for_completion(conversation)
        # Get the completions
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # Decode the completions as text
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        # Evaluate success criteria
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        # Keep stats
        total += 1
        num_passed += int(passed)

        # Logging (overwrite the same line in the console)
        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    # Finish the in-place progress line with a newline before final summary
    print()

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    print0("=" * 50)
    print0(f"Final: {num_passed}/{total} ({100*num_passed/total:.2f}%)")

    # Return the accuracy
    return num_passed/total

# -----------------------------------------------------------------------------
# Categorical evaluation loop
# A lot easier because we don't have to sample. Therefore, we can actually go
# batches at a time and just check the logits for correct answer choices.

def _forward_chat_prompts(model, prompt_ids, prompt_lengths):
    """Use ARMT's ragged forward while retaining the normal GPT path."""
    if hasattr(model, "forward_padded"):
        return model.forward_padded(prompt_ids, prompt_lengths)
    return model(prompt_ids)


def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id() # use BOS as pad token is ok, these positions are ignored

    # We'll process batches of independent problems at a time because there is no sampling needed
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    # Run the evaluation
    letter_to_id_cache = {} # many letters will repeat often, let's save the tokenizer some work
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        # Prepare the batch of problems. They might all be of different length, so we pad/collate them.
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conversation) for conversation in conversations] # TODO: remake the way this works
        max_length = max(len(ids) for ids in prompt_ids)
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        answer_time_positions = (prompt_lengths - 1).tolist()
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        # Get the logits for the whole batch of conversations in parallel (efficiency win here)
        with torch.no_grad():
            logits = _forward_chat_prompts(model, prompt_ids, prompt_lengths)

        # Focus on the available answer on just the letters corresponding to choices
        # Note that this helps the evaluation a lot because it specifically narrows the focus to only the available letters
        # The much harder alternative would be to just generate from the Assistant and check if it responded with the correct
        # letter (e.g. A, B, C, D), but evaluations typically make the task easier in this way.
        for idx, conversation in enumerate(conversations):
            # get the token ids of all the available letters of this problem
            letters = conversation['letters']
            letter_ids = []
            for letter in letters:
                if not letter in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "Each letter must be a single token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            # focus logits just down to the answer position and the available letters of the answer
            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]
            # get the argmax letter (the predicted answer)
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            # evaluate the outcome
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    average = num_passed/total
    print0(f"Final: {num_passed}/{total} ({100*average:.2f}%)")
    return average

# -----------------------------------------------------------------------------

def run_chat_eval(task_name, model, tokenizer, engine,
                   batch_size=1, num_samples=1, max_new_tokens=512, temperature=0.0, top_k=50,
                   max_problems=None, longbench_v2_options=None):
    if task_name == "LongBench-v2":
        options = {"max_problems": max_problems, **(longbench_v2_options or {})}
        return run_chat_longbench_v2_eval(model, tokenizer, **options)["longbench_v2_metric"]
    # Create the evaluation object
    task_module = {
        'HumanEval': HumanEval,
        'MMLU': partial(MMLU, subset="all", split="test"),
        'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test"),
        'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test"),
        'GSM8K': partial(GSM8K, subset="main", split="test"),
    }[task_name]
    task_object = task_module()
    # Run the evaluation
    if task_object.eval_type == 'generative':
        acc = run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=max_problems)
    elif task_object.eval_type == 'categorical':
        acc = run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=max_problems)
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")
    return acc

# -----------------------------------------------------------------------------

def parse_eval_modes(value):
    """Parse and validate the independently aggregated evaluation groups."""
    modes = tuple(dict.fromkeys(part.strip().lower() for part in value.split(",") if part.strip()))
    valid_modes = {"chat", "babilong", "longbench-v2"}
    invalid_modes = set(modes) - valid_modes
    if not modes:
        raise ValueError("At least one evaluation mode is required")
    if invalid_modes:
        raise ValueError(
            f"Invalid evaluation modes: {sorted(invalid_modes)}. "
            f"Valid modes: {sorted(valid_modes)}"
        )
    return modes


BABILONG_CSV_FIELDS = (
    "input_context",
    "input_question",
    "generated_answer",
    "ground_truth_answer",
    "answers_matched",
)


def _safe_path_component(value):
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return component or "model"


def resolve_babilong_results_dir(
    requested_dir,
    base_dir,
    source,
    model_tag,
    step,
    meta,
    model_type,
):
    """Resolve an explicit directory or a stable checkpoint-specific default."""
    if requested_dir:
        return os.path.abspath(os.path.expanduser(requested_dir))
    user_config = meta.get("user_config", {})
    resolved_tag = (
        model_tag
        or user_config.get("output_model_tag")
        or user_config.get("model_tag")
        or model_type
    )
    resolved_step = step if step is not None else meta.get("step", "latest")
    return os.path.join(
        base_dir,
        "chat_eval_results",
        _safe_path_component(source),
        _safe_path_component(resolved_tag),
        f"step_{_safe_path_component(resolved_step)}",
        "babilong",
    )


def save_babilong_cell_csv(output_dir, task, length, records):
    """Atomically save one BABILong task-length cell as UTF-8 CSV."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{_safe_path_component(task)}_{_safe_path_component(length)}.csv"
    output_path = os.path.join(output_dir, filename)
    temporary_path = f"{output_path}.tmp"
    with open(
        temporary_path, "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=BABILONG_CSV_FIELDS,
        )
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary_path, output_path)
    print0(f"Saved BABILong examples: {output_path}")
    return output_path


def run_chat_babilong_eval(
    model,
    tokenizer,
    device,
    tasks=None,
    lengths=None,
    max_per_task=-1,
    max_new_tokens=20,
    batch_size=DEFAULT_BABILONG_BATCH_SIZE,
    dataset_name=BABILONG_DATASET,
    results_dir=None,
):
    """Evaluate BABILong for either GPT or ARMT and print all aggregates."""
    example_callback = (
        partial(save_babilong_cell_csv, results_dir)
        if results_dir is not None
        else None
    )
    result = evaluate_babilong(
        model,
        tokenizer,
        device,
        tasks=tasks,
        lengths=lengths,
        max_per_task=max_per_task,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        dataset_name=dataset_name,
        example_callback=example_callback,
    )
    print0("\nBABILong task aggregates:")
    for task, score in result["task_results"].items():
        print0(f"  {task}: {score:.4f}")
    print0("BABILong length aggregates:")
    for length, score in result["length_results"].items():
        print0(f"  {length}: {score:.4f}")
    print0(f"BABILong metric: {result['babilong_metric']:.4f}")
    if results_dir is not None:
        print0(f"BABILong example CSV directory: {results_dir}")
    return result


def add_longbench_v2_arguments(parser):
    group = parser.add_argument_group("LongBench-v2 (greedy direct-answer evaluation)")
    group.add_argument('--longbench-v2-max-length', type=parse_length, default=None,
                       help='Inclusive context-length limit, e.g. 16k or 32768 (1k=1024; default: no limit). Longer examples are excluded, not truncated.')
    group.add_argument('--longbench-v2-length-buckets', type=parse_length_buckets, default=DEFAULT_LENGTH_BUCKETS,
                       help='Increasing comma-separated boundaries for disjoint and cumulative scores (default: %(default)s)')
    group.add_argument('--longbench-v2-length-unit', choices=('tokens', 'words'), default='tokens',
                       help='Measure context using model tokenizer tokens or whitespace-separated words (default: tokens)')
    group.add_argument('--longbench-v2-batch-size', type=int, default=1,
                       help='Per-device generation batch size (default: 1 for long-context memory use)')
    group.add_argument('--longbench-v2-max-new-tokens', type=int, default=128,
                       help='Greedy answer generation budget (default: 128)')
    group.add_argument('--longbench-v2-dataset', default=LONGBENCH_V2_DATASET,
                       help='Hugging Face dataset repository (default: %(default)s)')
    group.add_argument('--longbench-v2-cache-dir', default=None, help='Optional Hugging Face datasets cache directory')
    group.add_argument('--longbench-v2-revision', default=None, help='Optional dataset commit/revision for reproducible runs')
    group.add_argument('--longbench-v2-results-file', default=None,
                       help='JSON output path (default: checkpoint-specific chat_eval_results directory)')


def run_chat_longbench_v2_eval(model, tokenizer, results_path=None, **kwargs):
    result = evaluate_longbench_v2(model, tokenizer, **kwargs)
    overall = result['overall']
    print0(f"\nLongBench-v2 accuracy: {100 * overall['accuracy']:.2f}% ({overall['correct']}/{overall['total']})")
    for key, title in (("length_buckets", "length buckets"), ("length_cumulative", "cumulative lengths"),
                       ("by_dataset_length", "dataset length labels"), ("by_difficulty", "difficulty"),
                       ("by_domain", "domains")):
        print0(f"LongBench-v2 {title}:")
        for label, score in result[key].items():
            accuracy = "n/a" if score['accuracy'] is None else f"{100 * score['accuracy']:.2f}%"
            print0(f"  {label}: {accuracy} ({score['correct']}/{score['total']})")
    print0(f"LongBench-v2 eval time: {result['eval_time_seconds']:.2f}s")
    if results_path is not None and get_dist_info()[1] == 0:
        output_path = save_longbench_results(results_path, result)
        print0(f"Saved LongBench-v2 results: {output_path}")
    return result

# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--source', choices=('base', 'sft', 'rl'), default='sft',
        help="Checkpoint source: base (pretrained), sft, or rl",
    )
    parser.add_argument(
        '--model-type', '--model_type', choices=('auto', 'gpt', 'armt'), default='auto',
        help="Expected architecture; auto detects it from checkpoint metadata",
    )
    parser.add_argument(
        '--eval', default='chat,babilong',
        help='Comma-separated evaluations: chat,babilong,longbench-v2 (default: chat,babilong)',
    )
    parser.add_argument('-a', '--task-name', type=str, default=None, help="Chat task name, including LongBench-v2. Default = the five ChatCORE tasks. Use | to split multiple tasks.")
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch size for categorical evaluation')
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
    parser.add_argument('-x', '--max-problems', type=int, default=None, help='Max problems to evaluate')
    parser.add_argument('--babilong-tasks', type=str, default='qa1-qa5', help='BABILong tasks, e.g. qa1-qa5 or qa1,qa3 (default: qa1-qa5)')
    parser.add_argument('--babilong-lengths', type=str, default='0k-32k', help='BABILong context lengths, e.g. 0k-32k or 0k,4k,32k (default: 0k-32k)')
    parser.add_argument('--babilong-max-per-task', type=int, default=-1, help='Max examples per BABILong task/length (-1 = all 100)')
    parser.add_argument('--babilong-max-new-tokens', type=int, default=20, choices=range(10, 31), metavar='N', help='Greedy BABILong generation budget from 10 to 30 tokens (default: 20)')
    parser.add_argument('--babilong-batch-size', type=int, default=DEFAULT_BABILONG_BATCH_SIZE, help='Per-device BABILong generation batch size (default: 8)')
    parser.add_argument('--babilong-dataset', type=str, default=BABILONG_DATASET, help='Hugging Face BABILong dataset repository')
    parser.add_argument(
        '--babilong-results-dir', type=str, default=None,
        help='Directory for per-task-length BABILong CSV files (default: checkpoint-specific cache directory)',
    )
    parser.add_argument('--seed', type=int, default=42, help='Global Python/NumPy/PyTorch seed (default: 42)')
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True, help='Require deterministic algorithms (default: enabled)')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
    add_longbench_v2_arguments(parser)
    args = parser.parse_args()
    try:
        eval_modes = parse_eval_modes(args.eval)
        babilong_tasks = parse_babilong_tasks(args.babilong_tasks)
        babilong_lengths = parse_babilong_lengths(args.babilong_lengths)
        if args.longbench_v2_batch_size <= 0 or args.longbench_v2_max_new_tokens <= 0:
            raise ValueError('LongBench-v2 batch size and max-new-tokens must be positive')
    except ValueError as exc:
        parser.error(str(exc))

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(
        device_type, seed=args.seed, deterministic=args.deterministic
    )

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    loaded_model_type = "armt" if hasattr(model.config, "num_mem_tokens") else "gpt"
    if args.model_type != "auto" and args.model_type != loaded_model_type:
        parser.error(
            f"Requested {args.model_type}, but checkpoint contains {loaded_model_type}"
        )
    print0(f"Loaded {loaded_model_type.upper()} from {args.source} checkpoints")
    engine = Engine(model, tokenizer)

    # Get the tasks to evaluate on
    all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval']
    baseline_accuracies = {
        'ARC-Easy': 0.25, # multiple choice 1 of 4 => 25%
        'ARC-Challenge': 0.25, # multiple choice 1 of 4 => 25%
        'MMLU': 0.25, # multiple choice 1 of 4 => 25%
        'GSM8K': 0.0, # open-ended => 0%
        'HumanEval': 0.0, # open-ended => 0%
    }
    task_names = all_tasks if args.task_name is None else args.task_name.split('|')
    run_longbench = 'longbench-v2' in eval_modes or ('chat' in eval_modes and 'LongBench-v2' in task_names)

    # Run all the task evaluations sequentially
    results = {}
    if 'chat' in eval_modes:
        for task_name in task_names:
            if task_name == 'LongBench-v2':
                continue  # Run once below, with its own options and aggregates.
            acc = run_chat_eval(
                task_name,
                model, tokenizer, engine,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                max_problems=args.max_problems,
            )
            results[task_name] = acc
            print0(f"{task_name} accuracy: {100 * acc:.2f}%")

    # calculate the ChatCORE metric if we can (similar to CORE, it's the mean centered accuracy)
    # this way, ChatCORE ranges from 0 (at random baseline) to 1 (peak performance)
    all_tasks_were_evaluated = all(task_name in results for task_name in all_tasks)
    if all_tasks_were_evaluated:
        centered_mean = 0
        for task_name in all_tasks:
            acc = results[task_name]
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        chatcore_metric = centered_mean / len(all_tasks)
        print0(f"ChatCORE metric: {chatcore_metric:.4f}")

    # BABILong is reported separately and never contributes to ChatCORE.
    if 'babilong' in eval_modes:
        babilong_results_dir = resolve_babilong_results_dir(
            args.babilong_results_dir,
            get_base_dir(),
            args.source,
            args.model_tag,
            args.step,
            meta,
            loaded_model_type,
        )
        print0("\n" + "=" * 50)
        print0("BABILong Evaluation")
        print0("=" * 50)
        print0(f"Tasks: {', '.join(babilong_tasks)}")
        print0(f"Lengths: {', '.join(babilong_lengths)}")
        run_chat_babilong_eval(
            model,
            tokenizer,
            device,
            tasks=babilong_tasks,
            lengths=babilong_lengths,
            max_per_task=args.babilong_max_per_task,
            max_new_tokens=args.babilong_max_new_tokens,
            batch_size=args.babilong_batch_size,
            dataset_name=args.babilong_dataset,
            results_dir=babilong_results_dir,
        )

    # LongBench-v2 is independent of ChatCORE and BABILong; opt in explicitly.
    if run_longbench:
        results_path = args.longbench_v2_results_file
        if results_path is None:
            checkpoint_results_dir = os.path.dirname(resolve_babilong_results_dir(
                None, get_base_dir(), args.source, args.model_tag, args.step, meta, loaded_model_type,
            ))
            limit = args.longbench_v2_max_length or 'all'
            count = args.max_problems if args.max_problems is not None else 'all'
            results_path = os.path.join(
                checkpoint_results_dir, 'longbench-v2',
                f'results_{args.longbench_v2_length_unit}_max_{limit}_seed_{args.seed}_n_{count}.json',
            )
        run_chat_longbench_v2_eval(
            model, tokenizer, results_path=results_path,
            max_length=args.longbench_v2_max_length,
            length_unit=args.longbench_v2_length_unit,
            length_buckets=args.longbench_v2_length_buckets,
            batch_size=args.longbench_v2_batch_size,
            max_new_tokens=args.longbench_v2_max_new_tokens,
            max_problems=args.max_problems, seed=args.seed,
            dataset_name=args.longbench_v2_dataset,
            cache_dir=args.longbench_v2_cache_dir,
            revision=args.longbench_v2_revision,
        )

    compute_cleanup()
