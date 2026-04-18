from __future__ import annotations

import pytest

from trade_assist.policy import get_policy_adapter
from trade_assist.ta.constants import COL_CLOSE
from trade_assist.tuning import tune_policy


def test_tune_policy_random_returns_trials(ohlcv_factory):
    df = ohlcv_factory(periods=420, start_price=100.0, step=0.2, volume=1_000_000.0)
    adapter = get_policy_adapter("v1")
    base_policy = adapter.default_policy_payload()

    result = tune_policy(
        adapter=adapter,
        ohlcv_map={"AAA": df},
        index_close=df[COL_CLOSE],
        initial_cash=100_000.0,
        initial_positions={"AAA": 0.0},
        base_policy_payload=base_policy,
        split_cfg={
            "warmup_days": 20,
            "walk_forward": {
                "train_days": 120,
                "validation_days": 60,
                "step_days": 60,
                "min_windows": 2,
            },
        },
        tuning_cfg={
            "method": "random",
            "trials": 4,
            "seed": 7,
            "search_space": {
                "max_weight": {"type": "float", "low": 0.25, "high": 0.75},
                "min_hold_days": {"type": "int", "low": 1, "high": 5},
            },
        },
        objective_cfg={},
    )

    assert len(result.trials) == 4
    assert len(result.windows) >= 2
    assert isinstance(result.best_policy_payload, dict)
    assert "max_weight" in result.best_policy_payload
    assert all(
        0 <= trial.failed_window_count <= trial.window_count for trial in result.trials
    )
    assert all(
        len(trial.failed_windows) == trial.failed_window_count
        for trial in result.trials
    )
    assert all(isinstance(trial.failed_criteria, tuple) for trial in result.trials)


def test_tune_policy_grid_enforces_max_grid_points(ohlcv_factory):
    df = ohlcv_factory(periods=420, start_price=100.0, step=0.2, volume=1_000_000.0)
    adapter = get_policy_adapter("v1")

    with pytest.raises(ValueError):
        tune_policy(
            adapter=adapter,
            ohlcv_map={"AAA": df},
            index_close=df[COL_CLOSE],
            initial_cash=100_000.0,
            initial_positions={"AAA": 0.0},
            base_policy_payload=adapter.default_policy_payload(),
            split_cfg={
                "warmup_days": 20,
                "walk_forward": {
                    "train_days": 120,
                    "validation_days": 60,
                    "step_days": 60,
                },
            },
            tuning_cfg={
                "method": "grid",
                "max_grid_points": 4,
                "search_space": {
                    "min_hold_days": {"type": "int", "low": 1, "high": 5},
                    "max_weight": {"type": "float", "low": 0.2, "high": 0.8, "num": 5},
                },
            },
            objective_cfg={},
        )


def test_tune_policy_reports_progress(ohlcv_factory):
    df = ohlcv_factory(periods=420, start_price=100.0, step=0.2, volume=1_000_000.0)
    adapter = get_policy_adapter("v1")
    progress_calls = []

    result = tune_policy(
        adapter=adapter,
        ohlcv_map={"AAA": df},
        index_close=df[COL_CLOSE],
        initial_cash=100_000.0,
        initial_positions={"AAA": 0.0},
        base_policy_payload=adapter.default_policy_payload(),
        split_cfg={
            "warmup_days": 20,
            "walk_forward": {
                "train_days": 120,
                "validation_days": 60,
                "step_days": 60,
                "min_windows": 2,
            },
        },
        tuning_cfg={
            "method": "random",
            "trials": 3,
            "seed": 11,
            "search_space": {
                "max_weight": {"type": "float", "low": 0.25, "high": 0.75},
            },
        },
        objective_cfg={},
        progress_callback=lambda trial_id, total_trials, trial: progress_calls.append(
            (
                trial_id,
                total_trials,
                trial.trial_id,
                trial.failed_window_count,
                trial.window_count,
                tuple(trial.failed_windows),
                tuple(trial.failed_criteria),
            )
        ),
    )

    assert len(result.trials) == 3
    assert [call[:3] for call in progress_calls] == [(1, 3, 1), (2, 3, 2), (3, 3, 3)]
    assert all(
        0 <= failed <= total
        and len(failed_windows) == failed
        and isinstance(failed_criteria, tuple)
        for _, _, _, failed, total, failed_windows, failed_criteria in progress_calls
    )


def test_tune_policy_parallel_workers_runs_trials(ohlcv_factory):
    df = ohlcv_factory(periods=420, start_price=100.0, step=0.2, volume=1_000_000.0)
    adapter = get_policy_adapter("v1")
    progress_calls = []

    result = tune_policy(
        adapter=adapter,
        ohlcv_map={"AAA": df},
        index_close=df[COL_CLOSE],
        initial_cash=100_000.0,
        initial_positions={"AAA": 0.0},
        base_policy_payload=adapter.default_policy_payload(),
        split_cfg={
            "warmup_days": 20,
            "walk_forward": {
                "train_days": 120,
                "validation_days": 60,
                "step_days": 60,
                "min_windows": 2,
            },
        },
        tuning_cfg={
            "method": "random",
            "trials": 2,
            "workers": 2,
            "seed": 19,
            "search_space": {
                "max_weight": {"type": "float", "low": 0.25, "high": 0.75},
            },
        },
        objective_cfg={},
        progress_callback=lambda completed_count, total_trials, trial: (
            progress_calls.append((completed_count, total_trials, trial.trial_id))
        ),
    )

    assert len(result.trials) == 2
    assert sorted(trial.trial_id for trial in result.trials) == [1, 2]
    assert [completed for completed, _, _ in progress_calls] == [1, 2]
    assert sorted(trial_id for _, _, trial_id in progress_calls) == [1, 2]
