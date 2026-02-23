from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    final_holdings: pd.Series
    final_cash: float
    regime: pd.Series
    rebalance_log: pd.DataFrame
    account_history: pd.DataFrame
