from __future__ import annotations

from .policy import PolicyConfig
from .ta.constants import DEFAULT_INDEX_TICKER, DEFAULT_PERIOD
from .ta.runner import run_ta
from .workflow import backtest_from_tickers


def main() -> None:
    tickers = ["VRT", "ANET", "CDNS"]

    # Measurement layer (TA)
    run_ta(tickers=tickers, plot=True)

    # Decision layer (policy)
    config = PolicyConfig.from_dict(
        {
            "rebalance_freq": "W-FRI",
            "target_vol_risk_on": 0.22,
            "gross_cap_risk_on": 1.0,
            "gross_cap_risk_off": 0.3,
            "max_weight": 0.45,
            "min_hold_days": 10,
            "score_weights": {
                "c_over_ema200": 0.35,
                "ema50_over_ema200": 0.25,
                "mom_252_21": 0.25,
                "vol_adjusted_momentum": 0.15,
                "breakout55": 0.10,
            },
        }
    )

    result = backtest_from_tickers(
        tickers=tickers,
        index_ticker=DEFAULT_INDEX_TICKER,
        period=DEFAULT_PERIOD,
        config=config,
    )
    print("\nFinal equity:", round(float(result.equity_curve.iloc[-1]), 2))
    print("Final cash:", round(result.final_cash, 2))
    print("Final holdings (shares):")
    print(result.final_holdings.round(4))


if __name__ == "__main__":
    main()
