from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .constants import (
    DEFAULT_COOLDOWN_DAYS,
    DEFAULT_COVARIANCE_LOOKBACK,
    DEFAULT_GROSS_CAP_RISK_OFF,
    DEFAULT_GROSS_CAP_RISK_ON,
    DEFAULT_MAX_TRADE_ADV_FRACTION,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_MIN_ADV_DOLLARS,
    DEFAULT_MIN_HOLD_DAYS,
    DEFAULT_REBALANCE_FREQ,
    DEFAULT_RISK_OFF_TARGET_VOL_MULTIPLIER,
    DEFAULT_REGIME_BREADTH_MIN_FRAC,
    DEFAULT_REGIME_DRAWDOWN_FLOOR,
    DEFAULT_REGIME_DRAWDOWN_LOOKBACK,
    DEFAULT_REGIME_MIN_CONFIRMATIONS,
    DEFAULT_REGIME_REQUIRE_ANCHOR_TREND,
    DEFAULT_REGIME_USE_BREADTH,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TARGET_VOL_RISK_ON,
    DEFAULT_TRAILING_STOP_PCT,
    DEFAULT_WEIGHT_BREAKOUT55,
    DEFAULT_WEIGHT_C_OVER_EMA200,
    DEFAULT_WEIGHT_EMA50_OVER_EMA200,
    DEFAULT_WEIGHT_MOM_252_21,
    DEFAULT_WEIGHT_VOL_ADJUSTED_MOMENTUM,
)


@dataclass
class ScoreWeights:
    c_over_ema200: float = DEFAULT_WEIGHT_C_OVER_EMA200
    ema50_over_ema200: float = DEFAULT_WEIGHT_EMA50_OVER_EMA200
    mom_252_21: float = DEFAULT_WEIGHT_MOM_252_21
    vol_adjusted_momentum: float = DEFAULT_WEIGHT_VOL_ADJUSTED_MOMENTUM
    breakout55: float = DEFAULT_WEIGHT_BREAKOUT55


@dataclass
class LiquidityConfig:
    min_adv_dollars: float = DEFAULT_MIN_ADV_DOLLARS
    max_trade_adv_fraction: float = DEFAULT_MAX_TRADE_ADV_FRACTION


@dataclass
class RiskExitConfig:
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT
    trailing_stop_pct: float = DEFAULT_TRAILING_STOP_PCT
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS


@dataclass
class RegimeConfig:
    use_breadth: bool = DEFAULT_REGIME_USE_BREADTH
    breadth_min_frac: float = DEFAULT_REGIME_BREADTH_MIN_FRAC
    min_confirmations: int = DEFAULT_REGIME_MIN_CONFIRMATIONS
    require_anchor_trend: bool = DEFAULT_REGIME_REQUIRE_ANCHOR_TREND
    drawdown_lookback: int = DEFAULT_REGIME_DRAWDOWN_LOOKBACK
    drawdown_floor: float = DEFAULT_REGIME_DRAWDOWN_FLOOR


@dataclass
class PolicyConfig:
    rebalance_freq: str = DEFAULT_REBALANCE_FREQ
    target_vol_risk_on: float = DEFAULT_TARGET_VOL_RISK_ON
    gross_cap_risk_on: float = DEFAULT_GROSS_CAP_RISK_ON
    gross_cap_risk_off: float = DEFAULT_GROSS_CAP_RISK_OFF
    max_weight: float = DEFAULT_MAX_WEIGHT
    min_hold_days: int = DEFAULT_MIN_HOLD_DAYS
    risk_off_target_vol_multiplier: float = DEFAULT_RISK_OFF_TARGET_VOL_MULTIPLIER
    covariance_lookback: int = DEFAULT_COVARIANCE_LOOKBACK
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    risk_exit: RiskExitConfig = field(default_factory=RiskExitConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PolicyConfig":
        data = dict(payload)
        score_weights_payload = data.pop("score_weights", {}) or {}
        liquidity_payload = data.pop("liquidity", {}) or {}
        risk_exit_payload = data.pop("risk_exit", {}) or {}
        regime_payload = data.pop("regime", {}) or {}
        return cls(
            score_weights=ScoreWeights(**score_weights_payload),
            liquidity=LiquidityConfig(**liquidity_payload),
            risk_exit=RiskExitConfig(**risk_exit_payload),
            regime=RegimeConfig(**regime_payload),
            **data,
        )
