from __future__ import annotations

import matplotlib.pyplot as plt

from .constants import COL_CLOSE, COL_VOLUME, LOOKBACK_SESSIONS, RSI_OVERBOUGHT, RSI_OVERSOLD
from .models import TickerTA


def plot_ticker(ta: TickerTA, last_n: int = LOOKBACK_SESSIONS) -> None:
    df = ta.df.iloc[-last_n:].copy()

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[3.5, 1.4, 1.4], hspace=0.15)

    ax = fig.add_subplot(gs[0])
    ax.plot(df.index, df[COL_CLOSE], label=COL_CLOSE)
    ax.plot(df.index, df["SMA20"], label="SMA20")
    ax.plot(df.index, df["SMA50"], label="SMA50")
    ax.plot(df.index, df["SMA200"], label="SMA200")

    for level in ta.levels_support:
        ax.axhline(level, linewidth=1, linestyle="--")
    for level in ta.levels_resistance:
        ax.axhline(level, linewidth=1, linestyle="--")

    ax.set_title(f"{ta.ticker} - Price + MAs + Pivot S/R (last {last_n} sessions)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)

    ax_vol = ax.twinx()
    ax_vol.fill_between(df.index, 0, df[COL_VOLUME], alpha=0.2)
    ax_vol.set_yticks([])

    ax2 = fig.add_subplot(gs[1], sharex=ax)
    ax2.plot(df.index, df["RSI14"], label="RSI14")
    ax2.axhline(RSI_OVERBOUGHT, linestyle="--", linewidth=1)
    ax2.axhline(RSI_OVERSOLD, linestyle="--", linewidth=1)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(gs[2], sharex=ax)
    ax3.plot(df.index, df["MACD"], label="MACD")
    ax3.plot(df.index, df["MACDSignal"], label="Signal")
    ax3.bar(df.index, df["MACDHist"], width=1.0, alpha=0.3, label="Hist")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.25)

    plt.show()
