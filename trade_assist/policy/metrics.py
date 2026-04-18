from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import ANNUAL_TRADING_DAYS

DEFAULT_DAYS_PER_YEAR = 365.25
PERCENT_SCALE = 100.0


def compute_performance_stats(equity_curve: pd.Series) -> dict[str, float]:
    eq = equity_curve.dropna()
    if len(eq) < 2:
        return {
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "annualized_vol_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar": 0.0,
            "win_rate_pct": 0.0,
            "best_day_pct": 0.0,
            "worst_day_pct": 0.0,
        }

    returns = eq.pct_change().dropna()
    if len(returns) == 0:
        returns = pd.Series([0.0], index=eq.index[:1])

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    duration_days = max((eq.index[-1] - eq.index[0]).days, 1)
    years = duration_days / DEFAULT_DAYS_PER_YEAR
    cagr = (
        float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0)
        if eq.iloc[0] > 0
        else 0.0
    )

    annualized_vol = float(returns.std() * np.sqrt(ANNUAL_TRADING_DAYS))
    annualized_mean = float(returns.mean() * ANNUAL_TRADING_DAYS)
    sharpe = float(annualized_mean / annualized_vol) if annualized_vol > 0 else 0.0

    downside = returns[returns < 0]
    downside_vol = (
        float(downside.std() * np.sqrt(ANNUAL_TRADING_DAYS))
        if len(downside) > 1
        else 0.0
    )
    sortino = float(annualized_mean / downside_vol) if downside_vol > 0 else 0.0

    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    max_drawdown = float(drawdown.min())
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 else 0.0

    win_rate = float((returns > 0).mean())
    best_day = float(returns.max())
    worst_day = float(returns.min())

    return {
        "total_return_pct": total_return * PERCENT_SCALE,
        "cagr_pct": cagr * PERCENT_SCALE,
        "annualized_vol_pct": annualized_vol * PERCENT_SCALE,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_drawdown * PERCENT_SCALE,
        "calmar": calmar,
        "win_rate_pct": win_rate * PERCENT_SCALE,
        "best_day_pct": best_day * PERCENT_SCALE,
        "worst_day_pct": worst_day * PERCENT_SCALE,
    }
