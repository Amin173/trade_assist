from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class TuningWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    slice_start: pd.Timestamp
    slice_end: pd.Timestamp


@dataclass
class TrialEvaluation:
    trial_id: int
    score: float
    feasible: bool
    window_count: int
    failed_window_count: int
    failed_windows: tuple[str, ...]
    failed_criteria: tuple[str, ...]
    mean_cagr_pct: float
    mean_max_drawdown_pct: float
    mean_annualized_vol_pct: float
    mean_sharpe: float
    mean_sortino: float
    mean_trade_count: float
    policy_payload: dict[str, Any]
    overrides: dict[str, Any]

    def to_row(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "score": self.score,
            "feasible": self.feasible,
            "window_count": self.window_count,
            "failed_window_count": self.failed_window_count,
            "failed_windows": "; ".join(self.failed_windows),
            "failed_criteria": "; ".join(self.failed_criteria),
            "mean_cagr_pct": self.mean_cagr_pct,
            "mean_max_drawdown_pct": self.mean_max_drawdown_pct,
            "mean_annualized_vol_pct": self.mean_annualized_vol_pct,
            "mean_sharpe": self.mean_sharpe,
            "mean_sortino": self.mean_sortino,
            "mean_trade_count": self.mean_trade_count,
            "overrides": self.overrides,
        }


@dataclass
class TuningResult:
    trials: list[TrialEvaluation]
    best_trial: TrialEvaluation
    best_policy_payload: dict[str, Any]
    windows: list[TuningWindow]
    final_test_stats: dict[str, float] | None = None

    def trials_dataframe(self) -> pd.DataFrame:
        rows = [trial.to_row() for trial in self.trials]
        if not rows:
            return pd.DataFrame(
                columns=[
                    "trial_id",
                    "score",
                    "feasible",
                    "window_count",
                    "failed_window_count",
                    "failed_windows",
                    "failed_criteria",
                    "mean_cagr_pct",
                    "mean_max_drawdown_pct",
                    "mean_annualized_vol_pct",
                    "mean_sharpe",
                    "mean_sortino",
                    "mean_trade_count",
                    "overrides",
                ]
            )
        return pd.DataFrame(rows).sort_values(
            ["score", "trial_id"], ascending=[False, True]
        )
