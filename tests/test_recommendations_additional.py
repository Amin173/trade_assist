from __future__ import annotations

import pandas as pd

from trade_assist.policy import PolicyConfig
from trade_assist.policy.constants import ACTION_BUY, ACTION_HOLD
import trade_assist.policy.recommendations as recmod
from trade_assist.ta.constants import COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN, COL_VOLUME


def _single_day_ohlcv(price: float = 100.0, volume: float = 1000.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
    return pd.DataFrame(
        {
            COL_OPEN: [price],
            COL_HIGH: [price * 1.01],
            COL_LOW: [price * 0.99],
            COL_CLOSE: [price],
            COL_VOLUME: [volume],
        },
        index=idx,
    )


def test_recommend_positions_threshold_boundary(monkeypatch):
    df = _single_day_ohlcv(price=100.0, volume=100_000.0)
    ohlcv_map = {"AAA": df}
    idx = df.index

    def fake_latest_target_weights(*_args, **_kwargs):
        target_w = pd.Series({"AAA": 1.0})
        feats = {"AAA": pd.DataFrame({"c_over_ema200": [1.0]}, index=idx)}
        scores = pd.DataFrame({"AAA": [0.5]}, index=idx)
        return target_w, 1, feats, scores

    monkeypatch.setattr(recmod, "_latest_target_weights", fake_latest_target_weights)
    cfg = PolicyConfig.from_dict({"liquidity": {"min_adv_dollars": 0.0, "max_trade_adv_fraction": 1.0}})

    # target_shares = equity / price = 100 / 100 = 1.0
    recs_equal, _, _ = recmod.recommend_positions(
        ohlcv_map=ohlcv_map,
        index_close=df[COL_CLOSE],
        current_positions={"AAA": 0.0},
        current_cash=100.0,
        config=cfg,
        min_trade_shares=1.0,
    )
    assert recs_equal[0].action == ACTION_BUY

    recs_above, _, _ = recmod.recommend_positions(
        ohlcv_map=ohlcv_map,
        index_close=df[COL_CLOSE],
        current_positions={"AAA": 0.0},
        current_cash=100.0,
        config=cfg,
        min_trade_shares=1.1,
    )
    assert recs_above[0].action == ACTION_HOLD


def test_recommend_positions_liquidity_cap_reason_and_target(monkeypatch):
    df = _single_day_ohlcv(price=100.0, volume=100.0)  # ADV dollars = 10,000
    ohlcv_map = {"AAA": df}
    idx = df.index

    def fake_latest_target_weights(*_args, **_kwargs):
        target_w = pd.Series({"AAA": 1.0})
        feats = {"AAA": pd.DataFrame({"c_over_ema200": [1.0]}, index=idx)}
        scores = pd.DataFrame({"AAA": [0.9]}, index=idx)
        return target_w, 1, feats, scores

    monkeypatch.setattr(recmod, "_latest_target_weights", fake_latest_target_weights)
    cfg = PolicyConfig.from_dict({"liquidity": {"min_adv_dollars": 0.0, "max_trade_adv_fraction": 0.1}})

    # equity=10000 -> raw target shares=100; cap=10% * $10,000 = $1,000 -> 10 shares
    recs, _, _ = recmod.recommend_positions(
        ohlcv_map=ohlcv_map,
        index_close=df[COL_CLOSE],
        current_positions={"AAA": 0.0},
        current_cash=10_000.0,
        config=cfg,
        min_trade_shares=0.01,
    )

    rec = recs[0]
    assert rec.target_shares == 10.0
    assert "Trade capped by liquidity" in rec.reason
