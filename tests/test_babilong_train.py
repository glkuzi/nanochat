"""Tests for BABILong trainer evaluation scheduling and defaults."""

import pytest

from scripts.babilong_train import (
    _should_evaluate,
    _update_early_stopping,
    build_parser,
    main,
)


def test_standard_evaluations_are_enabled_by_default():
    args = build_parser().parse_args([
        "--mode",
        "scratch",
        "--model-type",
        "gpt",
    ])

    assert args.train_data_backend == "precomputed"
    assert args.generated_vary_lengths is False
    assert args.generated_mixed_length_ratio == 0.0
    assert args.generated_noise_dataset == "pg19"
    assert args.generated_qa_margin == 20
    assert args.eval_every == 250
    assert args.early_stopping_metric == "validation_loss"
    assert args.early_stopping_patience == -1
    assert args.early_stopping_min_delta == 0.0
    assert args.core_metric_every == 2000
    assert args.babilong_eval_every == 2000
    assert args.core_metric_max_per_task == -1


def test_generated_training_backend_options_are_exposed():
    args = build_parser().parse_args([
        "--mode",
        "scratch",
        "--model-type",
        "armt",
        "--train-data-backend",
        "generated",
        "--generated-vary-lengths",
        "--generated-mixed-length-ratio",
        "0.25",
        "--generated-task-start-pct",
        "0.1",
        "--generated-task-end-pct",
        "0.9",
    ])
    assert args.train_data_backend == "generated"
    assert args.generated_vary_lengths is True
    assert args.generated_mixed_length_ratio == 0.25


def test_babilong_every_alias_is_supported():
    args = build_parser().parse_args([
        "--mode",
        "scratch",
        "--model-type",
        "gpt",
        "--babilong-every",
        "17",
    ])
    assert args.babilong_eval_every == 17


def test_positive_schedules_always_include_final_step():
    assert _should_evaluate(0, False, 100, include_step_zero=True)
    assert not _should_evaluate(0, False, 100)
    assert _should_evaluate(2000, False, 2000)
    assert _should_evaluate(137, True, 2000)
    assert not _should_evaluate(137, True, -1)


def test_early_stopping_uses_training_step_patience_and_min_delta():
    best_bpb, best_step, improved, should_stop = _update_early_stopping(
        val_bpb=2.0,
        step=0,
        best_bpb=float("inf"),
        best_step=None,
        patience_steps=500,
        min_delta=0.01,
    )
    assert (best_bpb, best_step, improved, should_stop) == (2.0, 0, True, False)

    unchanged = _update_early_stopping(
        val_bpb=1.995,
        step=250,
        best_bpb=best_bpb,
        best_step=best_step,
        patience_steps=500,
        min_delta=0.01,
    )
    assert unchanged == (2.0, 0, False, False)

    exhausted = _update_early_stopping(
        val_bpb=2.1,
        step=500,
        best_bpb=best_bpb,
        best_step=best_step,
        patience_steps=500,
        min_delta=0.01,
    )
    assert exhausted == (2.0, 0, False, True)

    disabled = _update_early_stopping(
        val_bpb=2.1,
        step=1000,
        best_bpb=best_bpb,
        best_step=best_step,
        patience_steps=-1,
        min_delta=0.01,
    )
    assert disabled == (2.0, 0, False, False)


def test_early_stopping_can_maximize_babilong_metric():
    best_score, best_step, improved, should_stop = _update_early_stopping(
        val_bpb=0.40,
        step=0,
        best_bpb=float("-inf"),
        best_step=None,
        patience_steps=500,
        min_delta=0.01,
        maximize=True,
    )
    assert (best_score, best_step, improved, should_stop) == (
        0.40,
        0,
        True,
        False,
    )

    below_delta = _update_early_stopping(
        val_bpb=0.405,
        step=250,
        best_bpb=best_score,
        best_step=best_step,
        patience_steps=500,
        min_delta=0.01,
        maximize=True,
    )
    assert below_delta == (0.40, 0, False, False)

    exhausted = _update_early_stopping(
        val_bpb=0.39,
        step=500,
        best_bpb=best_score,
        best_step=best_step,
        patience_steps=500,
        min_delta=0.01,
        maximize=True,
    )
    assert exhausted == (0.40, 0, False, True)


def test_babilong_early_stopping_requires_positive_eval_schedule(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "babilong_train.py",
            "--mode",
            "scratch",
            "--model-type",
            "gpt",
            "--early-stopping-metric",
            "babilong_metric",
            "--early-stopping-patience",
            "500",
            "--babilong-eval-every",
            "-1",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2


def test_babilong_early_stopping_uses_training_tasks_and_lengths(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "babilong_train.py",
            "--mode",
            "scratch",
            "--model-type",
            "gpt",
            "--tasks",
            "qa1-qa5",
            "--lengths",
            "0k-8k",
            "--eval-tasks",
            "qa1",
            "--early-stopping-metric",
            "babilong_metric",
            "--early-stopping-patience",
            "500",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
