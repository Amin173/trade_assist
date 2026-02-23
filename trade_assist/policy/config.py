from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoreWeights:
    c_over_ema200: float = 0.35
    ema50_over_ema200: float = 0.25
    mom_252_21: float = 0.25
    vol_adjusted_momentum: float = 0.15
    breakout55: float = 0.10


@dataclass
class PolicyConfig:
    rebalance_freq: str = "W-FRI"
    target_vol_risk_on: float = 0.22
    gross_cap_risk_on: float = 1.00
    gross_cap_risk_off: float = 0.30
    max_weight: float = 0.45
    min_hold_days: int = 10
    risk_off_target_vol_multiplier: float = 0.5
    covariance_lookback: int = 252
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PolicyConfig":
        data = dict(payload)
        score_weights_payload = data.pop("score_weights", {}) or {}
        return cls(score_weights=ScoreWeights(**score_weights_payload), **data)
