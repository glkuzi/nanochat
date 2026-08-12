"""Tests for BABILong trainer evaluation scheduling and defaults."""

from scripts.babilong_train import (
    _should_evaluate,
    _update_early_stopping,
    build_parser,
)


def test_standard_evaluations_are_enabled_by_default():
    args = build_parser().parse_args([
        "--mode",
        "scratch",
        "--model-type",
        "gpt",
    ])

    assert args.eval_every == 250
    assert args.early_stopping_patience == -1
    assert args.early_stopping_min_delta == 0.0
    assert args.core_metric_every == 2000
    assert args.babilong_eval_every == 2000
    assert args.core_metric_max_per_task == -1


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
