from __future__ import annotations

import pandas as pd

from ..ta.indicators import ema, realized_volatility
from .constants import (
    EMA_FAST_WINDOW,
    EMA_SLOW_WINDOW,
    REGIME_SLOPE_DIFF,
    REGIME_VOL_CEILING,
    VOL_WINDOW,
)


def compute_regime(index_close: pd.Series) -> pd.Series:
    e200 = ema(index_close, EMA_SLOW_WINDOW)
    e50 = ema(index_close, EMA_FAST_WINDOW)
    slope50 = e50.diff(REGIME_SLOPE_DIFF)
    vol20 = realized_volatility(index_close, VOL_WINDOW)
    risk_on = (index_close > e200) & (slope50 > 0) & (vol20 < REGIME_VOL_CEILING)
    return risk_on.astype(int)
