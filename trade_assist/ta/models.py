from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TickerTA:
    ticker: str
    df: pd.DataFrame
    levels_support: list[float]
    levels_resistance: list[float]
    mdd: float
    mdd_peak: pd.Timestamp
    mdd_trough: pd.Timestamp
