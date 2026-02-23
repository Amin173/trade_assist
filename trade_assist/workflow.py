from __future__ import annotations

import pandas as pd

from .policy import BacktestResult, PolicyConfig, run_policy
from .policy.constants import DEFAULT_INITIAL_CASH
from .ta.constants import (
    COL_CLOSE,
    DEFAULT_INDEX_TICKER,
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
)
from .ta.data import fetch_ohlcv_map


def backtest_from_tickers(
    tickers: list[str],
    index_ticker: str = DEFAULT_INDEX_TICKER,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    config: PolicyConfig | None = None,
    initial_equity: float = DEFAULT_INITIAL_CASH,
    initial_positions: dict[str, float] | None = None,
) -> BacktestResult:
    ohlcv_map = fetch_ohlcv_map(tickers=tickers, period=period, interval=interval)
    index_df = fetch_ohlcv_map([index_ticker], period=period, interval=interval)[
        index_ticker
    ]
    index_close: pd.Series = index_df[COL_CLOSE]
    return run_policy(
        ohlcv_map=ohlcv_map,
        index_close=index_close,
        config=config,
        initial_cash=initial_equity,
        initial_positions=initial_positions,
    )
