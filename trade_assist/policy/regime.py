from __future__ import annotations

import pandas as pd

from ..ta.indicators import ema, realized_volatility
from .config import RegimeConfig
from .constants import (
    EMA_FAST_WINDOW,
    EMA_SLOW_WINDOW,
    REGIME_SLOPE_DIFF,
    REGIME_VOL_CEILING,
    VOL_WINDOW,
)


def _to_close_matrix(index_close: pd.Series | pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    if isinstance(index_close, pd.DataFrame):
        if index_close.empty or index_close.shape[1] == 0:
            raise ValueError("index_close DataFrame must include at least one column")
        anchor = index_close.iloc[:, 0]
        matrix = index_close
    else:
        anchor = index_close
        matrix = pd.DataFrame({"anchor": index_close})
    return anchor, matrix


def compute_regime(
    index_close: pd.Series | pd.DataFrame,
    config: RegimeConfig | None = None,
) -> pd.Series:
    cfg = config or RegimeConfig()
    anchor_close, close_matrix = _to_close_matrix(index_close)

    e200 = ema(anchor_close, EMA_SLOW_WINDOW)
    e50 = ema(anchor_close, EMA_FAST_WINDOW)
    slope50 = e50.diff(REGIME_SLOPE_DIFF)
    vol20 = realized_volatility(anchor_close, VOL_WINDOW)

    anchor_trend = anchor_close > e200
    trend_alignment = e50 > e200
    slope_positive = slope50 > 0
    low_vol = vol20 < REGIME_VOL_CEILING

    roll_max = anchor_close.rolling(cfg.drawdown_lookback, min_periods=1).max()
    drawdown = anchor_close / roll_max - 1.0
    drawdown_ok = drawdown >= cfg.drawdown_floor

    signals: dict[str, pd.Series] = {
        "anchor_trend": anchor_trend,
        "trend_alignment": trend_alignment,
        "slope_positive": slope_positive,
        "low_vol": low_vol,
        "drawdown_ok": drawdown_ok,
    }

    if cfg.use_breadth and close_matrix.shape[1] > 1:
        above_slow = close_matrix.gt(close_matrix.apply(lambda s: ema(s, EMA_SLOW_WINDOW)))
        breadth_frac = above_slow.mean(axis=1)
        signals["breadth_ok"] = breadth_frac >= cfg.breadth_min_frac

    signal_df = pd.DataFrame(signals).fillna(False)
    min_conf = max(1, min(cfg.min_confirmations, signal_df.shape[1]))
    risk_on = signal_df.sum(axis=1) >= min_conf
    if cfg.require_anchor_trend:
        risk_on &= anchor_trend.fillna(False)
    return risk_on.astype(int)
