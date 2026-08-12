"""Checkpoint persistence tests."""

import json

import torch

from nanochat.checkpoint_manager import save_checkpoint


def test_save_checkpoint_writes_evaluation_file(tmp_path):
    evaluation = {
        "checkpoint_step": 12,
        "training": {
            "total_time_seconds": 123.5,
            "total_flops": 1.0e15,
            "peak_memory_bytes": 1073741824,
            "peak_memory_mib": 1024.0,
        },
        "validation": {
            "evaluated_at_step": 10,
            "scores": {"bpb": 1.25, "minimum_bpb": 1.2},
        },
        "core": {
            "evaluated_at_step": 12,
            "per_task_scores": {"task_a": 0.75},
            "per_task_centered_scores": {"task_a": 0.5},
            "aggregated_scores": {"core_metric": 0.5},
            "eval_time": {
                "per_task_seconds": {"task_a": 2.5},
                "total_seconds": 2.5,
            },
        },
        "babilong": {
            "evaluated_at_step": 12,
            "per_task_per_length_scores": {"qa1": {"0k": 1.0}},
            "aggregated_scores": {
                "per_task": {"qa1": 1.0},
                "per_length": {"0k": 1.0},
                "overall": 1.0,
            },
            "eval_time": {
                "per_task_per_length_seconds": {"qa1": {"0k": 3.0}},
                "per_task_seconds": {"qa1": 3.0},
                "per_length_seconds": {"0k": 3.0},
                "total_seconds": 3.0,
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
