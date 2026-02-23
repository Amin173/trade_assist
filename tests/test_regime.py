from __future__ import annotations

import numpy as np
import pandas as pd

from trade_assist.policy import RegimeConfig
from trade_assist.policy.regime import compute_regime


def test_regime_breadth_signal_can_gate_risk_on():
    idx = pd.bdate_range("2022-01-03", periods=420)
    anchor = pd.Series(np.linspace(100.0, 200.0, len(idx)), index=idx)
    qqq = pd.Series(np.linspace(90.0, 180.0, len(idx)), index=idx)
    iwm = pd.Series(np.linspace(180.0, 90.0, len(idx)), index=idx)
    closes = pd.concat({"SPY": anchor, "QQQ": qqq, "IWM": iwm}, axis=1)

    no_breadth_cfg = RegimeConfig(
        use_breadth=False, min_confirmations=5, require_anchor_trend=True
    )
    with_breadth_cfg = RegimeConfig(
        use_breadth=True,
        breadth_min_frac=0.8,
        min_confirmations=6,
        require_anchor_trend=True,
    )

    regime_no_breadth = compute_regime(closes, config=no_breadth_cfg)
    regime_with_breadth = compute_regime(closes, config=with_breadth_cfg)

    assert int(regime_no_breadth.iloc[-1]) == 1
    assert int(regime_with_breadth.iloc[-1]) == 0


def test_regime_anchor_requirement_overrides_other_signals():
    idx = pd.bdate_range("2022-01-03", periods=420)
    anchor = pd.Series(np.linspace(100.0, 90.0, len(idx)), index=idx)

    relaxed_cfg = RegimeConfig(
        min_confirmations=1, require_anchor_trend=False, use_breadth=False
    )
    strict_cfg = RegimeConfig(
        min_confirmations=1, require_anchor_trend=True, use_breadth=False
    )

    regime_relaxed = compute_regime(anchor, config=relaxed_cfg)
    regime_strict = compute_regime(anchor, config=strict_cfg)

    assert int(regime_relaxed.iloc[-1]) == 1
    assert int(regime_strict.iloc[-1]) == 0
