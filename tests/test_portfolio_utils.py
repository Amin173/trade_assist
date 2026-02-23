from __future__ import annotations

import numpy as np
import pandas as pd

from trade_assist.policy.portfolio import (
    estimate_slippage_bps,
    softmax_weights,
    vol_target_scale,
)


def test_softmax_weights_returns_zero_series_for_all_nan():
    scores = pd.Series({"AAA": np.nan, "BBB": np.nan})
    out = softmax_weights(scores)
    assert (out == 0.0).all()


def test_vol_target_scale_returns_zero_when_no_common_assets():
    weights = pd.Series({"AAA": 1.0})
    cov = pd.DataFrame([[0.04]], index=["BBB"], columns=["BBB"])
    assert vol_target_scale(weights, cov, target_vol=0.2) == 0.0


def test_estimate_slippage_bps_fallback_for_non_positive_adv():
    bps = estimate_slippage_bps(order_dollars=10_000.0, adv_dollars=0.0)
    assert bps == 50.0
