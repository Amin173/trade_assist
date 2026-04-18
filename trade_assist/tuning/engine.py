from __future__ import annotations

import copy
import itertools
import math
import os
import random
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..policy.adapter import PolicyAdapter
from ..policy.metrics import compute_performance_stats
from ..policy.registry import get_policy_adapter
from .models import TrialEvaluation, TuningResult, TuningWindow

KEY_TYPE = "type"
TYPE_FLOAT = "float"
TYPE_INT = "int"
TYPE_BOOL = "bool"
TYPE_CATEGORICAL = "categorical"

SEARCH_METHOD_RANDOM = "random"
SEARCH_METHOD_GRID = "grid"

DEFAULT_TRIALS = 100
DEFAULT_SEED = 42
DEFAULT_MAX_GRID_POINTS = 10_000
DEFAULT_TUNE_WORKERS = 1

DEFAULT_WARMUP_DAYS = 252
DEFAULT_TRAIN_DAYS = 756
DEFAULT_VALIDATION_DAYS = 126
DEFAULT_STEP_DAYS = 63
DEFAULT_MIN_WINDOWS = 1

DEFAULT_OBJECTIVE_WEIGHTS = {
    "cagr_pct": 1.0,
    "max_drawdown_pct": 1.0,
    "annualized_vol_pct": -0.1,
    "sharpe": 10.0,
    "sortino": 5.0,
    "trade_count": 0.0,
}


@dataclass
class _TrialWorkerState:
    adapter: PolicyAdapter
    ohlcv_map: dict[str, pd.DataFrame]
    index_close: pd.Series | pd.DataFrame
    initial_cash: float
    initial_positions: dict[str, float] | None
    base_policy_payload: dict[str, Any]
    objective_cfg: dict[str, Any]
    windows: list[TuningWindow]


_TRIAL_WORKER_STATE: _TrialWorkerState | None = None


@dataclass(frozen=True)
class _ConstraintFailure:
    code: str
    label: str
    detail: str


class TuningInterrupted(RuntimeError):
    def __init__(
        self,
        partial_trials: list[TrialEvaluation],
        total_trials: int,
        message: str = "Tuning interrupted by user.",
    ) -> None:
        super().__init__(message)
        self.partial_trials = list(partial_trials)
        self.total_trials = int(total_trials)


def _set_nested(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cur = payload
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _build_walk_forward_windows(
    dates: pd.DatetimeIndex,
    split_cfg: dict[str, Any],
) -> list[TuningWindow]:
    if len(dates) == 0:
        return []

    warmup_days = int(split_cfg.get("warmup_days", DEFAULT_WARMUP_DAYS))
    walk = split_cfg.get("walk_forward", {}) or {}
    train_days = int(walk.get("train_days", DEFAULT_TRAIN_DAYS))
    validation_days = int(walk.get("validation_days", DEFAULT_VALIDATION_DAYS))
    step_days = int(walk.get("step_days", DEFAULT_STEP_DAYS))
    min_windows = int(walk.get("min_windows", DEFAULT_MIN_WINDOWS))

    if train_days <= 0 or validation_days <= 0 or step_days <= 0:
        raise ValueError(
            "walk_forward train_days/validation_days/step_days must be > 0"
        )

    windows: list[TuningWindow] = []
    val_start_idx = warmup_days + train_days
    n = len(dates)

    while val_start_idx + validation_days <= n:
        train_start_idx = val_start_idx - train_days
        train_end_idx = val_start_idx - 1
        val_end_idx = val_start_idx + validation_days - 1
        slice_start_idx = max(0, train_start_idx - warmup_days)

        windows.append(
            TuningWindow(
                train_start=dates[train_start_idx],
                train_end=dates[train_end_idx],
                validation_start=dates[val_start_idx],
                validation_end=dates[val_end_idx],
                slice_start=dates[slice_start_idx],
                slice_end=dates[val_end_idx],
            )
        )
        val_start_idx += step_days

    if len(windows) < min_windows:
        raise ValueError(
            "Not enough data for walk-forward windows. "
            f"Generated {len(windows)}, need at least {min_windows}."
        )
    return windows


def _slice_ohlcv_map(
    ohlcv_map: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for ticker, df in ohlcv_map.items():
        sub = df.loc[start:end]
        if sub.empty:
            raise ValueError(
                f"No market data for {ticker} in slice {start.date()}..{end.date()}"
            )
        out[ticker] = sub
    return out


def _slice_index_close(
    index_close: pd.Series | pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series | pd.DataFrame:
    sub = index_close.loc[start:end]
    if len(sub) == 0:
        raise ValueError(f"No regime market data in slice {start.date()}..{end.date()}")
    return sub


def _extract_trade_count_for_validation(
    rebalance_log: pd.DataFrame,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> int:
    if (
        rebalance_log.empty
        or "date" not in rebalance_log.columns
        or "trade_count" not in rebalance_log.columns
    ):
        return 0
    reb = rebalance_log.copy()
    reb["date"] = pd.to_datetime(reb["date"], errors="coerce")
    reb = reb.dropna(subset=["date"])
    mask = (reb["date"] >= validation_start) & (reb["date"] <= validation_end)
    return int(reb.loc[mask, "trade_count"].sum())


def _score_stats(
    stats: dict[str, float],
    trade_count: int,
    objective_cfg: dict[str, Any],
) -> tuple[float, bool, tuple[_ConstraintFailure, ...]]:
    constraints = objective_cfg.get("constraints", {}) or {}
    min_trade_count = constraints.get("min_trade_count")
    max_drawdown_floor = constraints.get("max_drawdown_floor_pct")

    failures: list[_ConstraintFailure] = []
    if min_trade_count is not None and trade_count < int(min_trade_count):
        failures.append(
            _ConstraintFailure(
                code="min_trade_count",
                label=f"trade_count>={int(min_trade_count)}",
                detail=f"observed {trade_count}",
            )
        )
    if max_drawdown_floor is not None and float(stats["max_drawdown_pct"]) < float(
        max_drawdown_floor
    ):
        failures.append(
            _ConstraintFailure(
                code="max_drawdown_floor_pct",
                label=f"max_drawdown>={float(max_drawdown_floor):.2f}%",
                detail=f"observed {float(stats['max_drawdown_pct']):.2f}%",
            )
        )
    feasible = len(failures) == 0

    weights = dict(DEFAULT_OBJECTIVE_WEIGHTS)
    weights.update(objective_cfg.get("weights", {}) or {})

    score = 0.0
    score += float(weights.get("cagr_pct", 0.0)) * float(stats["cagr_pct"])
    score += float(weights.get("max_drawdown_pct", 0.0)) * float(
        stats["max_drawdown_pct"]
    )
    score += float(weights.get("annualized_vol_pct", 0.0)) * float(
        stats["annualized_vol_pct"]
    )
    score += float(weights.get("sharpe", 0.0)) * float(stats["sharpe"])
    score += float(weights.get("sortino", 0.0)) * float(stats["sortino"])
    score += float(weights.get("trade_count", 0.0)) * float(trade_count)

    if not feasible:
        score -= abs(score) + 1_000_000.0
    return score, feasible, tuple(failures)


def _format_validation_window_label(window: TuningWindow) -> str:
    return (
        f"{window.validation_start.date().isoformat()}.."
        f"{window.validation_end.date().isoformat()}"
    )


def _summarize_constraint_failures(
    failures: list[_ConstraintFailure],
) -> tuple[str, ...]:
    if not failures:
        return ()

    counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    details: dict[str, str] = {}
    order: list[str] = []
    for failure in failures:
        if failure.code not in counts:
            order.append(failure.code)
            counts[failure.code] = 0
            labels[failure.code] = failure.label
            details[failure.code] = failure.detail
        counts[failure.code] += 1

    return tuple(
        f"{labels[code]} x{counts[code]} ({details[code]})" for code in order
    )


def _sample_random_value(rng: random.Random, spec: dict[str, Any]) -> Any:
    spec_type = str(spec[KEY_TYPE]).lower()
    if spec_type == TYPE_BOOL:
        values = spec.get("values", [False, True])
        return rng.choice(list(values))

    if spec_type == TYPE_CATEGORICAL:
        values = spec.get("values")
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError("categorical search spec requires a non-empty values list")
        return rng.choice(values)

    if spec_type == TYPE_INT:
        low_i = int(spec["low"])
        high_i = int(spec["high"])
        step_i = int(spec.get("step", 1))
        if step_i <= 0:
            raise ValueError("int search spec step must be > 0")
        values = list(range(low_i, high_i + 1, step_i))
        if not values:
            raise ValueError(f"Empty int candidate set for spec: {spec}")
        return rng.choice(values)

    if spec_type == TYPE_FLOAT:
        low_f = float(spec["low"])
        high_f = float(spec["high"])
        if high_f < low_f:
            raise ValueError("float search spec requires high >= low")
        step_any = spec.get("step")
        if step_any is not None:
            step_f = float(step_any)
            if step_f <= 0:
                raise ValueError("float search spec step must be > 0")
            n = int(math.floor((high_f - low_f) / step_f))
            values = [low_f + i * step_f for i in range(n + 1)]
            if not values:
                raise ValueError(f"Empty float stepped candidate set for spec: {spec}")
            return rng.choice(values)
        if bool(spec.get("log", False)):
            if low_f <= 0 or high_f <= 0:
                raise ValueError("float log search spec requires low/high > 0")
            return float(math.exp(rng.uniform(math.log(low_f), math.log(high_f))))
        return float(rng.uniform(low_f, high_f))

    raise ValueError(f"Unsupported search spec type: {spec_type}")


def _grid_values(spec: dict[str, Any]) -> list[Any]:
    spec_type = str(spec[KEY_TYPE]).lower()
    if spec_type == TYPE_BOOL:
        return list(spec.get("values", [False, True]))
    if spec_type == TYPE_CATEGORICAL:
        values = spec.get("values")
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError("categorical search spec requires a non-empty values list")
        return values
    if spec_type == TYPE_INT:
        low_i = int(spec["low"])
        high_i = int(spec["high"])
        step_i = int(spec.get("step", 1))
        if step_i <= 0:
            raise ValueError("int search spec step must be > 0")
        return list(range(low_i, high_i + 1, step_i))
    if spec_type == TYPE_FLOAT:
        low_f = float(spec["low"])
        high_f = float(spec["high"])
        if high_f < low_f:
            raise ValueError("float search spec requires high >= low")
        step_any = spec.get("step")
        if step_any is not None:
            step_f = float(step_any)
            if step_f <= 0:
                raise ValueError("float search spec step must be > 0")
            n = int(math.floor((high_f - low_f) / step_f))
            return [low_f + i * step_f for i in range(n + 1)]
        num = int(spec.get("num", 5))
        if num <= 1:
            return [low_f]
        return [float(x) for x in np.linspace(low_f, high_f, num=num)]
    raise ValueError(f"Unsupported search spec type: {spec_type}")


def _iter_overrides(
    search_space: dict[str, Any], tuning_cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    method = str(tuning_cfg.get("method", SEARCH_METHOD_RANDOM)).lower()
    if method not in {SEARCH_METHOD_RANDOM, SEARCH_METHOD_GRID}:
        raise ValueError("tuning.method must be 'random' or 'grid'")

    items = list(search_space.items())
    if not items:
        raise ValueError("tuning.search_space must include at least one parameter")

    if method == SEARCH_METHOD_RANDOM:
        trials = int(tuning_cfg.get("trials", DEFAULT_TRIALS))
        seed = int(tuning_cfg.get("seed", DEFAULT_SEED))
        rng = random.Random(seed)
        out: list[dict[str, Any]] = []
        for _ in range(trials):
            sample: dict[str, Any] = {}
            for path, spec in items:
                sample[path] = _sample_random_value(rng, spec)
            out.append(sample)
        return out

    value_lists: list[list[Any]] = []
    keys: list[str] = []
    for path, spec in items:
        keys.append(path)
        vals = _grid_values(spec)
        if len(vals) == 0:
            raise ValueError(f"Grid search produced no candidates for '{path}'")
        value_lists.append(vals)

    total = 1
    for vals in value_lists:
        total *= len(vals)
    max_grid_points = int(tuning_cfg.get("max_grid_points", DEFAULT_MAX_GRID_POINTS))
    if total > max_grid_points:
        raise ValueError(
            "Grid search candidate count "
            f"{total} exceeds max_grid_points={max_grid_points}."
        )

    out = []
    for combo in itertools.product(*value_lists):
        out.append(dict(zip(keys, combo, strict=False)))
    return out


def _build_policy_payload(
    base_policy_payload: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    payload = copy.deepcopy(base_policy_payload)
    for path, value in overrides.items():
        _set_nested(payload, path, value)
    return payload


def _resolve_tuning_workers(tuning_cfg: dict[str, Any], total_trials: int) -> int:
    if total_trials <= 0:
        return 1

    if "workers" not in tuning_cfg:
        requested = DEFAULT_TUNE_WORKERS
    else:
        workers_raw = tuning_cfg["workers"]
        if workers_raw is None:
            requested = os.cpu_count() or 1
        else:
            requested = int(workers_raw)

    return max(1, min(int(requested), total_trials))


def _evaluate_trial(
    adapter: PolicyAdapter,
    ohlcv_map: dict[str, pd.DataFrame],
    index_close: pd.Series | pd.DataFrame,
    initial_cash: float,
    initial_positions: dict[str, float] | None,
    base_policy_payload: dict[str, Any],
    objective_cfg: dict[str, Any],
    windows: list[TuningWindow],
    trial_id: int,
    overrides: dict[str, Any],
) -> TrialEvaluation:
    payload = _build_policy_payload(base_policy_payload, overrides)
    policy_config = adapter.build_policy_config(payload)

    window_scores: list[float] = []
    feasible_flags: list[bool] = []
    cagr_vals: list[float] = []
    mdd_vals: list[float] = []
    vol_vals: list[float] = []
    sharpe_vals: list[float] = []
    sortino_vals: list[float] = []
    trade_counts: list[int] = []
    failed_windows: list[str] = []
    failed_criteria_details: list[_ConstraintFailure] = []

    for window in windows:
        sub_ohlcv = _slice_ohlcv_map(ohlcv_map, window.slice_start, window.slice_end)
        sub_index_close = _slice_index_close(
            index_close, window.slice_start, window.slice_end
        )
        backtest = adapter.run_backtest(
            ohlcv_map=sub_ohlcv,
            index_close=sub_index_close,
            policy_config=policy_config,
            initial_cash=initial_cash,
            initial_positions=initial_positions,
        )

        val_eq = backtest.equity_curve.loc[
            window.validation_start : window.validation_end
        ]
        stats = compute_performance_stats(val_eq)
        trade_count = _extract_trade_count_for_validation(
            rebalance_log=backtest.rebalance_log,
            validation_start=window.validation_start,
            validation_end=window.validation_end,
        )
        score, feasible, constraint_failures = _score_stats(
            stats, trade_count, objective_cfg
        )

        window_scores.append(score)
        feasible_flags.append(feasible)
        if not feasible:
            failed_windows.append(_format_validation_window_label(window))
            failed_criteria_details.extend(constraint_failures)
        cagr_vals.append(float(stats["cagr_pct"]))
        mdd_vals.append(float(stats["max_drawdown_pct"]))
        vol_vals.append(float(stats["annualized_vol_pct"]))
        sharpe_vals.append(float(stats["sharpe"]))
        sortino_vals.append(float(stats["sortino"]))
        trade_counts.append(trade_count)

    return TrialEvaluation(
        trial_id=trial_id,
        score=float(np.mean(window_scores)) if window_scores else float("-inf"),
        feasible=all(feasible_flags) if feasible_flags else False,
        window_count=len(windows),
        failed_window_count=int(sum(1 for flag in feasible_flags if not flag)),
        failed_windows=tuple(failed_windows),
        failed_criteria=_summarize_constraint_failures(failed_criteria_details),
        mean_cagr_pct=float(np.mean(cagr_vals)) if cagr_vals else 0.0,
        mean_max_drawdown_pct=float(np.mean(mdd_vals)) if mdd_vals else 0.0,
        mean_annualized_vol_pct=float(np.mean(vol_vals)) if vol_vals else 0.0,
        mean_sharpe=float(np.mean(sharpe_vals)) if sharpe_vals else 0.0,
        mean_sortino=float(np.mean(sortino_vals)) if sortino_vals else 0.0,
        mean_trade_count=float(np.mean(trade_counts)) if trade_counts else 0.0,
        policy_payload=payload,
        overrides=overrides,
    )


def _initialize_trial_worker(
    policy_type: str,
    ohlcv_map: dict[str, pd.DataFrame],
    index_close: pd.Series | pd.DataFrame,
    initial_cash: float,
    initial_positions: dict[str, float] | None,
    base_policy_payload: dict[str, Any],
    objective_cfg: dict[str, Any],
    windows: list[TuningWindow],
) -> None:
    global _TRIAL_WORKER_STATE
    _TRIAL_WORKER_STATE = _TrialWorkerState(
        adapter=get_policy_adapter(policy_type),
        ohlcv_map=ohlcv_map,
        index_close=index_close,
        initial_cash=initial_cash,
        initial_positions=initial_positions,
        base_policy_payload=base_policy_payload,
        objective_cfg=objective_cfg,
        windows=windows,
    )


def _evaluate_trial_in_worker(
    trial_id: int, overrides: dict[str, Any]
) -> TrialEvaluation:
    if _TRIAL_WORKER_STATE is None:
        raise RuntimeError("Trial worker state was not initialized")
    return _evaluate_trial(
        adapter=_TRIAL_WORKER_STATE.adapter,
        ohlcv_map=_TRIAL_WORKER_STATE.ohlcv_map,
        index_close=_TRIAL_WORKER_STATE.index_close,
        initial_cash=_TRIAL_WORKER_STATE.initial_cash,
        initial_positions=_TRIAL_WORKER_STATE.initial_positions,
        base_policy_payload=_TRIAL_WORKER_STATE.base_policy_payload,
        objective_cfg=_TRIAL_WORKER_STATE.objective_cfg,
        windows=_TRIAL_WORKER_STATE.windows,
        trial_id=trial_id,
        overrides=overrides,
    )


def _shutdown_executor_now(executor: Any) -> None:
    if hasattr(executor, "terminate_workers"):
        executor.terminate_workers()
        return
    if hasattr(executor, "kill_workers"):
        executor.kill_workers()
        return
    executor.shutdown(wait=False, cancel_futures=True)


def tune_policy(
    adapter: PolicyAdapter,
    ohlcv_map: dict[str, pd.DataFrame],
    index_close: pd.Series | pd.DataFrame,
    initial_cash: float,
    initial_positions: dict[str, float] | None,
    base_policy_payload: dict[str, Any],
    split_cfg: dict[str, Any] | None,
    tuning_cfg: dict[str, Any],
    objective_cfg: dict[str, Any] | None = None,
    progress_callback: Callable[[int, int, TrialEvaluation], None] | None = None,
) -> TuningResult:
    if not ohlcv_map:
        raise ValueError("ohlcv_map must include at least one ticker")

    objective = objective_cfg or {}
    split = split_cfg or {}
    dates = next(iter(ohlcv_map.values())).index
    windows = _build_walk_forward_windows(dates, split)

    search_space = tuning_cfg.get("search_space", {})
    if not isinstance(search_space, dict):
        raise ValueError("tuning.search_space must be an object")
    override_list = _iter_overrides(search_space, tuning_cfg)
    total_trials = len(override_list)
    worker_count = _resolve_tuning_workers(tuning_cfg, total_trials)

    trials: list[TrialEvaluation] = []
    if worker_count == 1:
        try:
            for completed_count, (trial_id, overrides) in enumerate(
                enumerate(override_list, start=1), start=1
            ):
                trial = _evaluate_trial(
                    adapter=adapter,
                    ohlcv_map=ohlcv_map,
                    index_close=index_close,
                    initial_cash=initial_cash,
                    initial_positions=initial_positions,
                    base_policy_payload=base_policy_payload,
                    objective_cfg=objective,
                    windows=windows,
                    trial_id=trial_id,
                    overrides=overrides,
                )
                trials.append(trial)
                if progress_callback is not None:
                    progress_callback(completed_count, total_trials, trial)
        except KeyboardInterrupt as exc:
            raise TuningInterrupted(
                partial_trials=trials,
                total_trials=total_trials,
            ) from None
    else:
        try:
            executor: Any = ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_initialize_trial_worker,
                initargs=(
                    adapter.policy_type,
                    ohlcv_map,
                    index_close,
                    initial_cash,
                    initial_positions,
                    base_policy_payload,
                    objective,
                    windows,
                ),
            )
            interrupted = False
            try:
                future_to_trial_id = {
                    executor.submit(
                        _evaluate_trial_in_worker, trial_id, overrides
                    ): trial_id
                    for trial_id, overrides in enumerate(override_list, start=1)
                }
                completed_count = 0
                for future in as_completed(future_to_trial_id):
                    trial = future.result()
                    trials.append(trial)
                    completed_count += 1
                    if progress_callback is not None:
                        progress_callback(completed_count, total_trials, trial)
            except KeyboardInterrupt as exc:
                interrupted = True
                for future in future_to_trial_id:
                    future.cancel()
                _shutdown_executor_now(executor)
                raise TuningInterrupted(
                    partial_trials=trials,
                    total_trials=total_trials,
                ) from None
            finally:
                if not interrupted:
                    executor.shutdown(wait=True, cancel_futures=False)
        except (NotImplementedError, PermissionError, OSError):
            executor = ThreadPoolExecutor(max_workers=worker_count)
            interrupted = False
            try:
                future_to_trial_id = {
                    executor.submit(
                        _evaluate_trial,
                        adapter,
                        ohlcv_map,
                        index_close,
                        initial_cash,
                        initial_positions,
                        base_policy_payload,
                        objective,
                        windows,
                        trial_id,
                        overrides,
                    ): trial_id
                    for trial_id, overrides in enumerate(override_list, start=1)
                }
                completed_count = 0
                for future in as_completed(future_to_trial_id):
                    trial = future.result()
                    trials.append(trial)
                    completed_count += 1
                    if progress_callback is not None:
                        progress_callback(completed_count, total_trials, trial)
            except KeyboardInterrupt as exc:
                interrupted = True
                for future in future_to_trial_id:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise TuningInterrupted(
                    partial_trials=trials,
                    total_trials=total_trials,
                ) from None
            finally:
                if not interrupted:
                    executor.shutdown(wait=True, cancel_futures=False)

    if not trials:
        raise ValueError("No trials were generated")

    best_trial = sorted(
        trials, key=lambda t: (t.score, t.feasible, -t.trial_id), reverse=True
    )[0]

    final_test_stats: dict[str, float] | None = None
    final_test_cfg = split.get("final_test", {}) or {}
    if final_test_cfg.get("start") and final_test_cfg.get("end"):
        start = pd.Timestamp(str(final_test_cfg["start"]))
        end = pd.Timestamp(str(final_test_cfg["end"]))
        warmup_days = int(split.get("warmup_days", DEFAULT_WARMUP_DAYS))

        idx = dates
        start_pos = int(idx.searchsorted(start))
        if start_pos >= len(idx):
            raise ValueError(f"final_test.start {start.date()} is after available data")
        warm_pos = max(0, start_pos - warmup_days)
        warm_start = idx[warm_pos]

        sub_ohlcv = _slice_ohlcv_map(ohlcv_map, warm_start, end)
        sub_index_close = _slice_index_close(index_close, warm_start, end)
        best_config = adapter.build_policy_config(best_trial.policy_payload)
        backtest = adapter.run_backtest(
            ohlcv_map=sub_ohlcv,
            index_close=sub_index_close,
            policy_config=best_config,
            initial_cash=initial_cash,
            initial_positions=initial_positions,
        )
        eq_test = backtest.equity_curve.loc[start:end]
        final_test_stats = compute_performance_stats(eq_test)

    return TuningResult(
        trials=trials,
        best_trial=best_trial,
        best_policy_payload=best_trial.policy_payload,
        windows=windows,
        final_test_stats=final_test_stats,
    )
