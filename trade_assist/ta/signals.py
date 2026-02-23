from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import COL_CLOSE, EPSILON, RSI_OVERBOUGHT, RSI_OVERSOLD
from .models import TickerTA


def summarize_signals(ta: TickerTA) -> dict[str, str]:
    df = ta.df
    last = df.iloc[-1]

    trend = "neutral"
    if last[COL_CLOSE] > last["SMA200"] and last["SMA50"] > last["SMA200"]:
        trend = "bullish"
    elif last[COL_CLOSE] < last["SMA200"] and last["SMA50"] < last["SMA200"]:
        trend = "bearish"

    rsi_state = "neutral"
    if last["RSI14"] >= RSI_OVERBOUGHT:
        rsi_state = "overbought"
    elif last["RSI14"] <= RSI_OVERSOLD:
        rsi_state = "oversold"

    macd_state = "bullish" if last["MACD"] > last["MACDSignal"] else "bearish"

    dist_50 = (
        (last[COL_CLOSE] / last["SMA50"] - 1.0) if pd.notna(last["SMA50"]) else np.nan
    )
    dist_200 = (
        (last[COL_CLOSE] / last["SMA200"] - 1.0) if pd.notna(last["SMA200"]) else np.nan
    )

    return {
        "trend_regime": trend,
        "rsi14": f"{last['RSI14']:.1f} ({rsi_state})",
        "macd": macd_state,
        "dist_to_sma50": f"{dist_50 * 100:.2f}%" if np.isfinite(dist_50) else "n/a",
        "dist_to_sma200": f"{dist_200 * 100:.2f}%" if np.isfinite(dist_200) else "n/a",
        "atr14_pct": (
            f"{(last['ATR14'] / max(float(last[COL_CLOSE]), EPSILON)) * 100:.2f}%"
        ),
        "vol20_ann": f"{last['Vol20'] * 100:.2f}%",
        "max_drawdown": (
            f"{ta.mdd * 100:.2f}% ({ta.mdd_peak.date()} -> {ta.mdd_trough.date()})"
        ),
    }
