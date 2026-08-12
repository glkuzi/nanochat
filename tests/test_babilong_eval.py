"""Tests for BABILong selection, scoring, and macro aggregation."""

import torch

from nanochat.babilong_eval import (
    aggregate_babilong_scores,
    answer_in_completion,
    evaluate_babilong,
    generate_babilong_completions,
    parse_babilong_lengths,
    parse_babilong_tasks,
)


def test_armt_generation_uses_one_mixed_length_batch():
    class Tokenizer:
        def get_bos_token_id(self):
            return 15

        def encode_special(self, token):
            assert token == "<|assistant_end|>"
            return 14

        def __call__(self, prompt, prepend=None):
            return [prepend] + list(range(1, len(prompt) + 1))

        def decode(self, tokens):
            return ",".join(map(str, tokens))

    class MockARMT:
        armt_layers = ()

        def __init__(self):
            self.batch_shapes = []

        def get_device(self):
            return torch.device("cpu")

        def forward_padded(self, ids, sequence_lengths):
            self.batch_shapes.append((tuple(ids.shape), sequence_lengths.tolist()))
            logits = torch.zeros(*ids.shape, 16)
            logits[..., 14] = 1
            return logits

    model = MockARMT()
    completions = generate_babilong_completions(
        model,
        Tokenizer(),
        ["a", "abcd", "ab"],
        max_new_tokens=10,
    )

    assert completions == ["", "", ""]
    assert model.batch_shapes == [((3, 5), [2, 5, 3])]


def test_selection_ranges_and_defaults():
    assert parse_babilong_tasks() == ("qa1", "qa2", "qa3", "qa4", "qa5")
    assert parse_babilong_tasks("qa1-qa3,qa5") == ("qa1", "qa2", "qa3", "qa5")
    assert parse_babilong_lengths() == ("0k", "1k", "2k", "4k", "8k", "16k", "32k")
    assert parse_babilong_lengths("0k,4k,32k") == ("0k", "4k", "32k")


def test_substring_metric_and_aggregates():
    assert answer_in_completion("Kitchen", "The answer is KITCHEN.")
    assert not answer_in_completion("garden", "The answer is the office.")
    scores = {
        "qa1": {"0k": 1.0, "1k": 0.5},
        "qa2": {"0k": 0.0, "1k": 0.5},
    }
    aggregates = aggregate_babilong_scores(scores)
    assert aggregates["task_results"] == {"qa1": 0.75, "qa2": 0.25}
    assert aggregates["length_results"] == {"0k": 0.5, "1k": 0.5}
    assert aggregates["babilong_metric"] == 0.5


def test_evaluate_babilong_returns_all_aggregation_levels():
    rows = [
        {"input": f"Person {i} moved to the kitchen.", "question": f"Where is person {i}?", "target": "kitchen"}
        for i in range(10)
    ]
    observed_batch_sizes = []

    def dataset_loader(repo_id, subset, split):
        assert repo_id == "test/babilong"
        assert subset in {"0k", "1k"}
        assert split in {"qa1", "qa2"}
        return rows

    def generation_fn(model, tokenizer, prompts, max_new_tokens):
        assert all("<context>" in prompt for prompt in prompts)
        assert max_new_tokens == 20
        observed_batch_sizes.append(len(prompts))
        return ["The answer is KITCHEN."] * len(prompts)

    result = evaluate_babilong(
        model=object(),
        tokenizer=object(),
        device=torch.device("cpu"),
        tasks="qa1-qa2",
        lengths="0k-1k",
        dataset_name="test/babilong",
        dataset_loader=dataset_loader,
        generation_fn=generation_fn,
    )

    assert result["results"] == {
        "qa1": {"0k": 1.0, "1k": 1.0},
        "qa2": {"0k": 1.0, "1k": 1.0},
    }
    assert result["task_results"] == {"qa1": 1.0, "qa2": 1.0}
    assert result["length_results"] == {"0k": 1.0, "1k": 1.0}
    assert result["babilong_metric"] == 1.0
    assert result["example_counts"]["qa1"]["0k"] == 10
    assert result["batch_size"] == 8
    assert observed_batch_sizes == [8, 2] * 4
    timing = result["eval_time"]
    assert set(timing["per_task_per_length_seconds"]) == {"qa1", "qa2"}
    assert set(timing["per_task_seconds"]) == {"qa1", "qa2"}
    assert set(timing["per_length_seconds"]) == {"0k", "1k"}
    assert all(value >= 0 for value in timing["per_task_seconds"].values())
    assert timing["total_seconds"] == sum(timing["per_task_seconds"].values())
    for length in ("0k", "1k"):
        assert timing["per_length_seconds"][length] == sum(
            timing["per_task_per_length_seconds"][task][length]
            for task in ("qa1", "qa2")
        )


def test_evaluate_babilong_rejects_invalid_batch_size():
    try:
        evaluate_babilong(
            model=object(),
            tokenizer=object(),
            device=torch.device("cpu"),
            batch_size=0,
        )
    except ValueError as exc:
        assert str(exc) == "batch_size must be a positive integer"
    else:
        raise AssertionError("expected invalid batch size to fail")
