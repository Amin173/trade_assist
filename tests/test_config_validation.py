from __future__ import annotations

import pytest

from trade_assist.config_validation import (
    ConfigValidationError,
    validate_config,
    validate_policy,
)


def test_validate_backtest_config_rejects_missing_required_sections():
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config({}, command="backtest")
    assert str(exc_info.value) == (
        "Invalid backtest config: missing required field 'data' at top level."
    )


def test_validate_policy_accepts_known_keys():
    payload = {
        "rebalance_freq": "W-FRI",
        "min_trade_shares": 1.0,
        "liquidity": {"min_adv_dollars": 0, "max_trade_adv_fraction": 0.1},
        "risk_exit": {
            "stop_loss_pct": 0.1,
            "trailing_stop_pct": 0.2,
            "cooldown_days": 5,
        },
        "regime": {
            "use_breadth": True,
            "breadth_min_frac": 0.6,
            "min_confirmations": 4,
        },
    }
    validate_policy(payload)


def test_validate_backtest_config_accepts_plot_trade_markers():
    payload = {
        "data": {"tickers": ["AAA"]},
        "portfolio": {"cash": 1000},
        "output": {"plot_equity_curve": True, "plot_trade_markers": True},
    }
    validate_config(payload, command="backtest")


def test_validate_tune_config_accepts_minimal_sections():
    payload = {
        "data": {"tickers": ["AAA"]},
        "portfolio": {"cash": 1000},
        "tuning": {
            "method": "random",
            "trials": 5,
            "search_space": {"max_weight": {"type": "float", "low": 0.2, "high": 0.6}},
        },
    }
    validate_config(payload, command="tune")


def test_validate_tune_config_accepts_auto_workers():
    payload = {
        "data": {"tickers": ["AAA"]},
        "portfolio": {"cash": 1000},
        "tuning": {
            "method": "random",
            "trials": 5,
            "workers": None,
            "search_space": {"max_weight": {"type": "float", "low": 0.2, "high": 0.6}},
        },
    }
    validate_config(payload, command="tune")
