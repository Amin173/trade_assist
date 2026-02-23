from __future__ import annotations

import numpy as np

from trade_assist.policy import PolicyConfig, run_policy
from trade_assist.policy.constants import EVENT_RISK_EXIT
from trade_assist.ta.constants import COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN


def test_backtest_allows_sells_when_min_adv_blocks_buys(ohlcv_factory):
    df = ohlcv_factory(periods=320, start_price=120.0, step=0.2, volume=100.0)
    policy = PolicyConfig.from_dict(
        {
            "rebalance_freq": "B",
            "min_hold_days": 0,
            "gross_cap_risk_on": 0.0,
            "gross_cap_risk_off": 0.0,
            "liquidity": {"min_adv_dollars": 1_000_000_000.0, "max_trade_adv_fraction": 1.0},
            "risk_exit": {"stop_loss_pct": 0.0, "trailing_stop_pct": 0.0, "cooldown_days": 0},
        }
    )

    out = run_policy(
        ohlcv_map={"AAA": df},
        index_close=df[COL_CLOSE],
        config=policy,
        initial_cash=0.0,
        initial_positions={"AAA": 50.0},
    )

    first = out.rebalance_log.iloc[0]
    assert float(first["sell_notional"]) > 0.0
    assert float(out.final_holdings["AAA"]) == 0.0


def test_backtest_max_trade_adv_fraction_zero_blocks_all_trading(ohlcv_factory):
    initial_cash = 250_000.0
    df = ohlcv_factory(periods=340, start_price=80.0, step=0.3, volume=2_000_000.0)
    policy = PolicyConfig.from_dict(
        {
            "rebalance_freq": "B",
            "min_hold_days": 0,
            "liquidity": {"min_adv_dollars": 0.0, "max_trade_adv_fraction": 0.0},
            "risk_exit": {"stop_loss_pct": 0.0, "trailing_stop_pct": 0.0, "cooldown_days": 0},
        }
    )

    out = run_policy(
        ohlcv_map={"AAA": df},
        index_close=df[COL_CLOSE],
        config=policy,
        initial_cash=initial_cash,
    )

    assert int(out.rebalance_log["trade_count"].sum()) == 0
    assert float(out.final_cash) == initial_cash
    assert float(out.final_holdings["AAA"]) == 0.0


def test_backtest_trailing_stop_forces_exit_even_without_rebalance(ohlcv_factory):
    df = ohlcv_factory(periods=120, start_price=100.0, step=1.0, volume=5_000_000.0).copy()
    peak_ix = 85
    peak_price = float(df.iloc[peak_ix][COL_CLOSE])
    # Force a >20% drawdown from local peak to trigger trailing stop.
    df.loc[df.index[peak_ix + 1] :, COL_CLOSE] = peak_price * 0.75
    df.loc[df.index[peak_ix + 1] :, COL_OPEN] = df.loc[df.index[peak_ix + 1] :, COL_CLOSE]
    df.loc[df.index[peak_ix + 1] :, COL_HIGH] = df.loc[df.index[peak_ix + 1] :, COL_CLOSE] * 1.01
    df.loc[df.index[peak_ix + 1] :, COL_LOW] = df.loc[df.index[peak_ix + 1] :, COL_CLOSE] * 0.99

    policy = PolicyConfig.from_dict(
        {
            "rebalance_freq": "YE",
            "min_hold_days": 0,
            "liquidity": {"min_adv_dollars": 0.0, "max_trade_adv_fraction": 1.0},
            "risk_exit": {"stop_loss_pct": 0.0, "trailing_stop_pct": 0.2, "cooldown_days": 10},
        }
    )

    out = run_policy(
        ohlcv_map={"AAA": df},
        index_close=df[COL_CLOSE],
        config=policy,
        initial_cash=0.0,
        initial_positions={"AAA": 100.0},
    )

    exits = out.rebalance_log[out.rebalance_log["risk_exit_count"] > 0]
    assert not exits.empty
    assert set(exits["event"]) == {EVENT_RISK_EXIT}
    assert float(out.final_holdings["AAA"]) == 0.0
