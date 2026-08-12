"""Checkpoint persistence tests."""

import json

import torch

from nanochat.checkpoint_manager import save_checkpoint


def test_save_checkpoint_writes_evaluation_file(tmp_path):
    evaluation = {
        "checkpoint_step": 12,
        "validation": {
            "evaluated_at_step": 10,
            "scores": {"bpb": 1.25, "minimum_bpb": 1.2},
        },
        "core": {
            "evaluated_at_step": 12,
            "per_task_scores": {"task_a": 0.75},
            "per_task_centered_scores": {"task_a": 0.5},
            "aggregated_scores": {"core_metric": 0.5},
        },
        "babilong": {
            "evaluated_at_step": 12,
            "per_task_per_length_scores": {"qa1": {"0k": 1.0}},
            "aggregated_scores": {
                "per_task": {"qa1": 1.0},
                "per_length": {"0k": 1.0},
                "overall": 1.0,
            },
        },
    }
    metadata = {"step": 12, "evaluation": evaluation}

    save_checkpoint(
        tmp_path,
        step=12,
        model_data={"weight": torch.tensor([1.0])},
        optimizer_data=None,
        meta_data=metadata,
        eval_data=evaluation,
    )

    with open(tmp_path / "eval_000012.json", encoding="utf-8") as f:
        assert json.load(f) == evaluation
    with open(tmp_path / "meta_000012.json", encoding="utf-8") as f:
        assert json.load(f)["evaluation"] == evaluation
    assert (tmp_path / "model_000012.pt").is_file()
