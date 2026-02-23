from __future__ import annotations

from .features import build_features
from .models import TickerTA
from .plotting import plot_ticker
from .signals import summarize_signals


def run_ta(tickers: list[str], plot: bool = True) -> dict[str, TickerTA]:
    results: dict[str, TickerTA] = {}

    for ticker in tickers:
        ta = build_features(ticker)
        results[ticker] = ta

        summary = summarize_signals(ta)
        print("\n" + "=" * 80)
        print(f"{ticker} SIGNAL SUMMARY")
        for k, v in summary.items():
            print(f"- {k}: {v}")

        print(f"- support_levels (pivot clusters): {[round(x, 2) for x in ta.levels_support]}")
        print(f"- resistance_levels (pivot clusters): {[round(x, 2) for x in ta.levels_resistance]}")

        if plot:
            plot_ticker(ta, last_n=252)

    return results
