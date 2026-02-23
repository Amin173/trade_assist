from __future__ import annotations

import pytest

from trade_assist.config_validation import validate_config, validate_policy


def test_validate_backtest_config_rejects_missing_required_sections():
    with pytest.raises(ValueError):
        validate_config({}, command="backtest")


def test_validate_policy_accepts_known_keys():
    payload = {
        "rebalance_freq": "W-FRI",
        "liquidity": {"min_adv_dollars": 0, "max_trade_adv_fraction": 0.1},
        "risk_exit": {"stop_loss_pct": 0.1, "trailing_stop_pct": 0.2, "cooldown_days": 5},
    }
    validate_policy(payload)
