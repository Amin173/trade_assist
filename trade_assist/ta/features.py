from __future__ import annotations

from .constants import (
    ATR_WINDOW,
    COL_CLOSE,
    COL_HIGH,
    COL_LOW,
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    LOOKBACK_SESSIONS,
    PIVOT_LEFT,
    PIVOT_RIGHT,
    RSI_WINDOW,
    SMA_WINDOWS,
    TOP_LEVEL_COUNT,
    TOP_LEVEL_TOL,
    VOL_WINDOW,
)
from .data import fetch_ohlcv
from .indicators import annualized_volatility, atr, macd, max_drawdown, rsi, sma
from .levels import pivot_points, top_levels
from .models import TickerTA


def build_features(
    ticker: str, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL
) -> TickerTA:
    df = fetch_ohlcv(ticker, period=period, interval=interval)

    for w in SMA_WINDOWS:
        df[f"SMA{w}"] = sma(df[COL_CLOSE], w)

    df["RSI14"] = rsi(df[COL_CLOSE], RSI_WINDOW)
    df["MACD"], df["MACDSignal"], df["MACDHist"] = macd(df[COL_CLOSE])
    df["ATR14"] = atr(df[COL_HIGH], df[COL_LOW], df[COL_CLOSE], ATR_WINDOW)
    df["Vol20"] = annualized_volatility(df[COL_CLOSE], VOL_WINDOW)

    lookback = min(len(df), LOOKBACK_SESSIONS)
    sub = df.iloc[-lookback:].copy()

    piv_high = pivot_points(
        sub[COL_HIGH], left=PIVOT_LEFT, right=PIVOT_RIGHT, mode="high"
    )
    piv_low = pivot_points(sub[COL_LOW], left=PIVOT_LEFT, right=PIVOT_RIGHT, mode="low")

    resistance = top_levels(piv_high, k=TOP_LEVEL_COUNT, tol=TOP_LEVEL_TOL)
    support = top_levels(piv_low, k=TOP_LEVEL_COUNT, tol=TOP_LEVEL_TOL)

    mdd, peak, trough = max_drawdown(df[COL_CLOSE])

    return TickerTA(
        ticker=ticker,
        df=df,
        levels_support=support,
        levels_resistance=resistance,
        mdd=mdd,
        mdd_peak=peak,
        mdd_trough=trough,
    )
