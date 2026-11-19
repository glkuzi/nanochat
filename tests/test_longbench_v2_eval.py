"""Offline tests for LongBench-v2 filtering, generation, scoring and CLI routing."""

import argparse
import json
import os

import pytest
import torch

from nanochat import longbench_v2_eval as lb
from scripts import chat_eval


class Tokenizer:
    def encode(self, text):
        return [10 + ord(char) % 100 for char in text]

    def get_bos_token_id(self):
        return 1

    def encode_special(self, token):
        return {"<|user_start|>": 2, "<|user_end|>": 3,
                "<|assistant_start|>": 4, "<|assistant_end|>": 5}[token]

    def decode(self, tokens):
        return ",".join(map(str, tokens))


def example(length, index=0, answer="A"):
    return {"_id": str(index), "context": "x" * length, "question": "Which one?",
            **{f"choice_{letter}": f"choice {letter}" for letter in "ABCD"},
            "answer": answer, "length": "short", "difficulty": "easy", "domain": "test"}


@pytest.mark.parametrize("value,expected", [("16k", 16384), ("20K", 20480), (" 32768 ", 32768), (100, 100)])
def test_length_parsing(value, expected):
    assert lb.parse_length(value) == expected


@pytest.mark.parametrize("value", ["0k", "-1", "1.5k", "16kb", "", None])
def test_invalid_length(value):
    with pytest.raises(ValueError):
        lb.parse_length(value)


@pytest.mark.parametrize("value", ["", "16k,16k", "32k,16k", "16k,", ()])
def test_invalid_bucket_order(value):
    with pytest.raises(ValueError):
        lb.parse_length_buckets(value)


@pytest.mark.parametrize("output,answer", [
    ("The correct answer is (B).", "B"),
    ("**The correct answer is C**", "C"),
    ("A distracting letter. The correct answer is (D)", "D"),
    ("B", None), ("The correct answer is (b)", None), ("", None),
])
def test_upstream_strict_answer_extraction(output, answer):
    assert lb.extract_answer(output) == answer


def test_prompt_preserves_long_context_and_chat_framing():
    tokenizer = Tokenizer()
    row = example(5000)
    tokens = lb.encode_longbench_prompt(tokenizer, row)
    assert tokens[:2] == [1, 2]
    assert tokens[-2:] == [3, 4]
    assert tokens[2:-2] == tokenizer.encode(lb.render_longbench_prompt(row))
    assert len(tokens) > 5000
    assert row["context"] in lb.render_longbench_prompt(row)
    row["answer"] = "D"
    assert tokens == lb.encode_longbench_prompt(tokenizer, row)  # no label leakage


def test_inclusive_filter_buckets_and_micro_averaging(tmp_path):
    rows = [example(n, i, "A" if i in (0, 2) else "B")
            for i, n in enumerate((16384, 16385, 20480, 20481, 32768, 32769))]
    calls = []

    def load(name, **kwargs):
        assert name == lb.LONGBENCH_V2_DATASET
        assert kwargs == {"split": "train", "cache_dir": str(tmp_path), "revision": "pinned"}
        return rows

    model = torch.nn.Linear(1, 1).train()

    def generate(m, tokenizer, prompts, max_new_tokens):
        assert not m.training
        assert max_new_tokens == 30
        calls.append(len(prompts))
        return ["The correct answer is (A)"] * len(prompts)

    result = lb.evaluate_longbench_v2(
        model, Tokenizer(), max_length="32k", length_buckets="16k,20k,32k,64k",
        batch_size=2, max_new_tokens=30, dataset_loader=load, generation_fn=generate,
        cache_dir=str(tmp_path), revision="pinned",
    )
    assert model.training
    assert calls == [2, 2, 1]
    assert result["overall"] == {"accuracy": 2 / 5, "correct": 2, "total": 5}
    assert result["config"]["filtered_out_count"] == 1
    assert result["length_buckets"]["<=16k"]["total"] == 1
    assert result["length_buckets"]["16k-20k"]["accuracy"] == 0.5
    assert result["length_buckets"]["20k-32k"]["total"] == 2
    assert result["length_buckets"]["32k-64k"]["accuracy"] is None
    assert result["length_cumulative"]["<=20k"]["accuracy"] == 2 / 3
    assert sum(score["total"] for score in result["length_buckets"].values()) == 5
    assert [r["context_length"] for r in result["examples"]] == [16384, 16385, 20480, 20481, 32768]
    path = tmp_path / "results" / "scores.json"
    lb.save_longbench_results(str(path), result)
    assert json.loads(path.read_text()) == result


def test_words_filter_and_seeded_subset_after_filtering():
    rows = [dict(example(1, i), context="one two " * (i + 1)) for i in range(12)]
    kwargs = dict(length_unit="words", max_length=16, max_problems=3, seed=7,
                  dataset_loader=lambda *a, **k: rows,
                  generation_fn=lambda m, t, p, **k: ["The correct answer is A"] * len(p))
    first = lb.evaluate_longbench_v2(torch.nn.Linear(1, 1), Tokenizer(), **kwargs)
    second = lb.evaluate_longbench_v2(torch.nn.Linear(1, 1), Tokenizer(), **kwargs)
    assert first["examples"] == second["examples"]
    assert first["overall"]["total"] == 3
    assert first["config"]["eligible_count"] == 8
    assert all(r["context_length"] <= 16 for r in first["examples"])


def test_empty_selection_fails_instead_of_reporting_zero():
    with pytest.raises(ValueError, match="No LongBench-v2 examples"):
        lb.evaluate_longbench_v2(torch.nn.Linear(1, 1), Tokenizer(), max_length=1,
                                dataset_loader=lambda *a, **k: [example(2)])


def test_generation_failure_restores_model_mode():
    model = torch.nn.Linear(1, 1).train()
    with pytest.raises(ValueError, match="wrong number"):
        lb.evaluate_longbench_v2(model, Tokenizer(), dataset_loader=lambda *a, **k: [example(2)],
                                generation_fn=lambda *a, **k: [])
    assert model.training


@pytest.mark.parametrize("armt", [False, True])
def test_generation_with_real_tiny_models_matches_single_examples(armt):
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.armt import ARMTGPT, ARMTGPTConfig

    torch.manual_seed(42)
    kwargs = dict(sequence_len=8, vocab_size=128, n_layer=2, n_head=4,
                  n_kv_head=2, n_embd=32, window_pattern="L")
    if armt:
        config = ARMTGPTConfig(**kwargs, segment_size=4, num_mem_tokens=2,
                               d_mem=8, n_heads=2, memory_dtype="float32")
        cls = ARMTGPT
    else:
        config, cls = GPTConfig(**kwargs), GPT
    with torch.device("meta"):
        model = cls(config, pad_vocab_size_to=1)
    model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    # Longer than the checkpoint's sequence_len and multiple recurrent segments.
    prompts = [[10, 11, 12] * 5, [13, 14, 15] * 3]
    batched = lb.generate_longbench_completions(model, Tokenizer(), prompts, max_new_tokens=2)
    single = [lb.generate_longbench_completions(model, Tokenizer(), [p], max_new_tokens=2)[0] for p in prompts]
    assert batched == single


def test_cli_arguments_and_task_route(monkeypatch, tmp_path):
    parser = argparse.ArgumentParser()
    chat_eval.add_longbench_v2_arguments(parser)
    defaults = parser.parse_args([])
    assert defaults.longbench_v2_length_buckets == (16384, 20480, 32768, 65536, 131072)
    assert defaults.longbench_v2_batch_size == 1
    args = parser.parse_args(["--longbench-v2-max-length", "32k", "--longbench-v2-length-buckets", "16k,20k"])
    assert args.longbench_v2_max_length == 32768
    assert args.longbench_v2_length_buckets == (16384, 20480)
    assert chat_eval.parse_eval_modes("chat,longbench-v2,babilong,longbench-v2") == ("chat", "longbench-v2", "babilong")
    path = tmp_path / "results.json"
    score = chat_eval.run_chat_eval(
        "LongBench-v2", torch.nn.Linear(1, 1), Tokenizer(), None, max_problems=1,
        longbench_v2_options={"results_path": str(path), "dataset_loader": lambda *a, **k: [example(2)],
                             "generation_fn": lambda m, t, p, **k: ["The correct answer is A"] * len(p)},
    )
    assert score == 1.0
    assert json.loads(path.read_text())["overall"]["total"] == 1


def _distributed_worker(rank, init_path, output_dir):
    import torch.distributed as dist

    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
    dist.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=2)
    try:
        for count in (1, 5):
            result = lb.evaluate_longbench_v2(
                torch.nn.Linear(1, 1), Tokenizer(), max_problems=count,
                dataset_loader=lambda *a, **k: [example(i + 1, i) for i in range(5)],
                generation_fn=lambda m, t, p, **k: ["The correct answer is A"] * len(p),
            )
            lb.save_longbench_results(os.path.join(output_dir, f"rank{rank}_n{count}.json"), result)
    finally:
        dist.destroy_process_group()


def test_distributed_aggregation_with_uneven_and_empty_shards(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_distributed_worker, args=(str(tmp_path / "init"), str(tmp_path)), nprocs=2, join=True)
    for count in (1, 5):
        results = [json.loads((tmp_path / f"rank{rank}_n{count}.json").read_text()) for rank in (0, 1)]
        assert results[0]["examples"] == results[1]["examples"]
        assert results[0]["overall"] == {"accuracy": 1.0, "correct": count, "total": count}
        assert len({row["index"] for row in results[0]["examples"]}) == count
