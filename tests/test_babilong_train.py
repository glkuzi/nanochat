"""Tests for BABILong trainer evaluation scheduling and defaults."""

from scripts.babilong_train import _should_evaluate, build_parser


def test_standard_evaluations_are_enabled_by_default():
    args = build_parser().parse_args([
        "--mode",
        "scratch",
        "--model-type",
        "gpt",
    ])

    assert args.eval_every == 250
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
