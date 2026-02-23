from __future__ import annotations

from trade_assist.policy import PolicyConfig
from trade_assist.policy.recommendations import recommend_positions
from trade_assist.ta.constants import COL_CLOSE


def test_recommendation_applies_liquidity_constraints(ohlcv_factory):
    df = ohlcv_factory(periods=320, start_price=100.0, step=0.3, volume=50_000.0)
    ohlcv_map = {"AAA": df, "BBB": df * [1.02, 1.02, 1.02, 1.02, 1.0]}
    index_close = df[COL_CLOSE]

    policy = PolicyConfig.from_dict(
        {
            "gross_cap_risk_on": 1.0,
            "gross_cap_risk_off": 1.0,
            "liquidity": {"min_adv_dollars": 100_000_000.0, "max_trade_adv_fraction": 0.01},
        }
    )

    recs, _, _ = recommend_positions(
        ohlcv_map=ohlcv_map,
        index_close=index_close,
        current_positions={"AAA": 0.0, "BBB": 0.0},
        current_cash=1_000_000.0,
        config=policy,
        min_trade_shares=0.01,
    )

    reasons = [r.reason for r in recs]
    assert any("ADV below minimum" in reason for reason in reasons)
