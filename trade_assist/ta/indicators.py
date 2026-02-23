from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .constants import ATR_WINDOW, EPSILON, RSI_WINDOW, TRADING_DAYS_PER_YEAR, VOL_WINDOW


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = ATR_WINDOW) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def annualized_volatility(
    close: pd.Series,
    window: int = VOL_WINDOW,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    returns = close.pct_change()
    return returns.rolling(window).std() * np.sqrt(trading_days)


def realized_volatility(
    close: pd.Series,
    window: int = VOL_WINDOW,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    returns = close.pct_change()
    return returns.rolling(window).std() * np.sqrt(trading_days)


def max_drawdown(close: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    cummax = close.cummax()
    drawdown = close / (cummax + EPSILON) - 1.0
    trough = drawdown.idxmin()
    peak = close.loc[:trough].idxmax()
    return float(drawdown.loc[trough]), peak, trough
