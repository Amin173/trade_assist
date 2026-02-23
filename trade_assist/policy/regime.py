from __future__ import annotations

import pandas as pd

from ..ta.indicators import ema, realized_volatility


def compute_regime(index_close: pd.Series) -> pd.Series:
    e200 = ema(index_close, 200)
    e50 = ema(index_close, 50)
    slope50 = e50.diff(10)
    vol20 = realized_volatility(index_close, 20)
    risk_on = (index_close > e200) & (slope50 > 0) & (vol20 < 0.35)
    return risk_on.astype(int)
