from __future__ import annotations

import pandas as pd

from .policy import BacktestResult, PolicyConfig, run_policy
from .ta.data import fetch_ohlcv_map


def backtest_from_tickers(
    tickers: list[str],
    index_ticker: str = "SPY",
    period: str = "5y",
    interval: str = "1d",
    config: PolicyConfig | None = None,
    initial_equity: float = 100_000.0,
    initial_positions: dict[str, float] | None = None,
) -> BacktestResult:
    ohlcv_map = fetch_ohlcv_map(tickers=tickers, period=period, interval=interval)
    index_df = fetch_ohlcv_map([index_ticker], period=period, interval=interval)[index_ticker]
    index_close: pd.Series = index_df["Close"]
    return run_policy(
        ohlcv_map=ohlcv_map,
        index_close=index_close,
        config=config,
        initial_cash=initial_equity,
        initial_positions=initial_positions,
    )
