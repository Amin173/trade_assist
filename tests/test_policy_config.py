from __future__ import annotations

from trade_assist.policy import PolicyConfig


def test_policy_config_from_dict_parses_nested_sections():
    cfg = PolicyConfig.from_dict(
        {
            "rebalance_freq": "B",
            "min_trade_shares": 2.0,
            "liquidity": {
                "min_adv_dollars": 1_000_000.0,
                "max_trade_adv_fraction": 0.05,
            },
            "risk_exit": {
                "stop_loss_pct": 0.08,
                "trailing_stop_pct": 0.12,
                "cooldown_days": 7,
            },
            "score_weights": {"breakout55": 0.33},
        }
    )

    assert cfg.rebalance_freq == "B"
    assert cfg.min_trade_shares == 2.0
    assert cfg.liquidity.min_adv_dollars == 1_000_000.0
    assert cfg.liquidity.max_trade_adv_fraction == 0.05
    assert cfg.risk_exit.stop_loss_pct == 0.08
    assert cfg.risk_exit.trailing_stop_pct == 0.12
    assert cfg.risk_exit.cooldown_days == 7
    assert cfg.score_weights.breakout55 == 0.33
