from __future__ import annotations

import pandas as pd

from trade_assist.policy import PolicyConfig, run_policy
from trade_assist.ta.constants import COL_OPEN, COL_VOLUME


def test_backtest_liquidity_cap_limits_buy_notional(ohlcv_factory):
    df = ohlcv_factory(periods=320, start_price=100.0, step=0.4, volume=1_000.0)
    policy = PolicyConfig.from_dict(
        {
            "rebalance_freq": "B",
            "min_hold_days": 0,
            "gross_cap_risk_on": 1.0,
            "gross_cap_risk_off": 1.0,
            "liquidity": {"min_adv_dollars": 0, "max_trade_adv_fraction": 0.01},
            "risk_exit": {
                "stop_loss_pct": 0.0,
                "trailing_stop_pct": 0.0,
                "cooldown_days": 0,
            },
        }
    )

    out = run_policy(
        ohlcv_map={"AAA": df},
        index_close=df["Close"],
        config=policy,
        initial_cash=1_000_000.0,
    )

    buys = out.rebalance_log[out.rebalance_log["buy_notional"] > 0]
    assert not buys.empty

    first = buys.iloc[0]
    day = pd.Timestamp(first["date"])
    adv_for_day = float((df[COL_VOLUME].shift(-1) * df[COL_OPEN].shift(-1)).loc[day])
    expected_cap = adv_for_day * 0.01
    assert float(first["buy_notional"]) <= expected_cap + 1e-6
    assert int(first["liquidity_capped_count"]) >= 1


def test_backtest_stop_loss_exit_and_cooldown_prevent_reentry(ohlcv_factory):
    # Keep enough history before the shock so momentum/zscore features are available
    # and the policy can hold a live position when the stop-loss check runs.
    df = ohlcv_factory(
        periods=380, start_price=100.0, step=0.3, volume=5_000_000.0
    ).copy()
    # Inject a sharp drop that should trigger stop-loss from initial entry basis.
    shock_day = df.index[350]
    df.loc[shock_day:, "Close"] = df.loc[shock_day:, "Close"] * 0.45
    df.loc[shock_day:, "Open"] = df.loc[shock_day:, "Close"]
    df.loc[shock_day:, "High"] = df.loc[shock_day:, "Close"] * 1.01
    df.loc[shock_day:, "Low"] = df.loc[shock_day:, "Close"] * 0.99

    policy = PolicyConfig.from_dict(
        {
            "rebalance_freq": "B",
            "min_hold_days": 0,
            "gross_cap_risk_on": 1.0,
            "gross_cap_risk_off": 1.0,
            "liquidity": {"min_adv_dollars": 0, "max_trade_adv_fraction": 1.0},
            "risk_exit": {
                "stop_loss_pct": 0.1,
                "trailing_stop_pct": 0.0,
                "cooldown_days": 1000,
            },
        }
    )

    out = run_policy(
        ohlcv_map={"AAA": df},
        index_close=df["Close"],
        config=policy,
        initial_cash=0.0,
        initial_positions={"AAA": 100.0},
    )

    assert (out.rebalance_log["risk_exit_count"] > 0).any()
    assert float(out.final_holdings["AAA"]) == 0.0
