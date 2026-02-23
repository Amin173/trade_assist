from __future__ import annotations

from .data import fetch_ohlcv
from .indicators import annualized_volatility, atr, macd, max_drawdown, rsi, sma
from .levels import pivot_points, top_levels
from .models import TickerTA


def build_features(ticker: str, period: str = "3y", interval: str = "1d") -> TickerTA:
    df = fetch_ohlcv(ticker, period=period, interval=interval)

    for w in [20, 50, 100, 200]:
        df[f"SMA{w}"] = sma(df["Close"], w)

    df["RSI14"] = rsi(df["Close"], 14)
    df["MACD"], df["MACDSignal"], df["MACDHist"] = macd(df["Close"])
    df["ATR14"] = atr(df["High"], df["Low"], df["Close"], 14)
    df["Vol20"] = annualized_volatility(df["Close"], 20)

    lookback = min(len(df), 252)
    sub = df.iloc[-lookback:].copy()

    piv_high = pivot_points(sub["High"], left=3, right=3, mode="high")
    piv_low = pivot_points(sub["Low"], left=3, right=3, mode="low")

    resistance = top_levels(piv_high, k=6, tol=0.010)
    support = top_levels(piv_low, k=6, tol=0.010)

    mdd, peak, trough = max_drawdown(df["Close"])

    return TickerTA(
        ticker=ticker,
        df=df,
        levels_support=support,
        levels_resistance=resistance,
        mdd=mdd,
        mdd_peak=peak,
        mdd_trough=trough,
    )
