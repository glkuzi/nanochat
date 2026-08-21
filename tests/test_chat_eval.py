"""Focused tests for GPT/ARMT chat evaluation forwarding."""

import csv
import torch

from scripts import chat_eval as chat_eval_module
from scripts.chat_eval import (
    _forward_chat_prompts,
    parse_eval_modes,
    run_chat_babilong_eval,
)


class PaddedModel:
    def __init__(self):
        self.received_lengths = None

    def forward_padded(self, prompt_ids, prompt_lengths):
        self.received_lengths = prompt_lengths
        return prompt_ids.float().unsqueeze(-1)

    def __call__(self, prompt_ids):
        raise AssertionError("ARMT evaluation must use forward_padded")


class GPTModel:
    def __init__(self):
        self.called = False

    def __call__(self, prompt_ids):
        self.called = True
        return prompt_ids.float().unsqueeze(-1)


def test_armt_chat_eval_uses_real_prompt_lengths():
    model = PaddedModel()
    prompt_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    prompt_lengths = torch.tensor([3, 2])
    logits = _forward_chat_prompts(model, prompt_ids, prompt_lengths)

    torch.testing.assert_close(model.received_lengths, prompt_lengths)
    assert logits.shape == (2, 3, 1)


def test_gpt_chat_eval_keeps_standard_forward_path():
    model = GPTModel()
    prompt_ids = torch.tensor([[1, 2, 3]])
    prompt_lengths = torch.tensor([3])
    logits = _forward_chat_prompts(model, prompt_ids, prompt_lengths)

    assert model.called
    assert logits.shape == (1, 3, 1)


def test_chat_eval_modes_are_independent():
    assert parse_eval_modes("chat,babilong") == ("chat", "babilong")
    assert parse_eval_modes("babilong") == ("babilong",)

    try:
        parse_eval_modes("core")
    except ValueError as exc:
        assert "Invalid evaluation modes" in str(exc)
    else:
        raise AssertionError("expected an unknown evaluation mode to fail")


def test_chat_babilong_eval_passes_gpt_and_armt_to_shared_evaluator(monkeypatch):
    calls = []
    expected = {
        "results": {"qa1": {"0k": 0.5}},
        "task_results": {"qa1": 0.5},
        "length_results": {"0k": 0.5},
        "babilong_metric": 0.5,
    }

    def fake_evaluate(model, tokenizer, device, **kwargs):
        calls.append((model, tokenizer, device, kwargs))
        return expected

    monkeypatch.setattr(chat_eval_module, "evaluate_babilong", fake_evaluate)
    tokenizer = object()
    device = torch.device("cpu")
    models = (GPTModel(), PaddedModel())
    for model in models:
        result = run_chat_babilong_eval(
            model,
            tokenizer,
            device,
            tasks=("qa1",),
            lengths=("0k",),
            max_per_task=3,
            max_new_tokens=10,
            batch_size=8,
            dataset_name="test/babilong",
        )
        assert result is expected

    assert [call[0] for call in calls] == list(models)
    assert all(call[1] is tokenizer and call[2] == device for call in calls)
    assert all(
        call[3]
        == {
            "tasks": ("qa1",),
            "lengths": ("0k",),
            "max_per_task": 3,
            "max_new_tokens": 10,
            "batch_size": 8,
            "dataset_name": "test/babilong",
            "example_callback": None,
        }
        for call in calls
    )


def test_chat_babilong_eval_saves_one_csv_per_cell(monkeypatch, tmp_path):
    expected = {
        "results": {"qa1": {"2k": 1.0}},
        "task_results": {"qa1": 1.0},
        "length_results": {"2k": 1.0},
        "babilong_metric": 1.0,
    }
    record = {
        "input_context": "Mary moved to the kitchen.\nJohn stayed outside.",
        "input_question": "Where is Mary?",
        "generated_answer": "The answer is kitchen.",
        "ground_truth_answer": "kitchen",
        "answers_matched": True,
    }

    def fake_evaluate(model, tokenizer, device, **kwargs):
        kwargs["example_callback"]("qa1", "2k", [record])
        return expected

    monkeypatch.setattr(chat_eval_module, "evaluate_babilong", fake_evaluate)
    result = run_chat_babilong_eval(
        GPTModel(),
        object(),
        torch.device("cpu"),
        tasks=("qa1",),
        lengths=("2k",),
        results_dir=str(tmp_path),
    )

    assert result is expected
    with open(tmp_path / "qa1_2k.csv", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == chat_eval_module.BABILONG_CSV_FIELDS
        assert list(reader) == [{**record, "answers_matched": "True"}]


def test_default_babilong_results_dir_is_checkpoint_specific():
    path = chat_eval_module.resolve_babilong_results_dir(
        None,
        "/cache/nanochat",
        "sft",
        None,
        None,
        {"step": 17, "user_config": {"output_model_tag": "armt/d12"}},
        "armt",
    )

    assert path == "/cache/nanochat/chat_eval_results/sft/armt_d12/step_17/babilong"
