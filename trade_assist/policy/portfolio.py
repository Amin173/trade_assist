from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import (
    EPSILON,
    SLIPPAGE_BASE_BPS,
    SLIPPAGE_FALLBACK_BPS,
    SLIPPAGE_IMPACT_COEFF,
    SLIPPAGE_IMPACT_EXP,
)


def softmax_weights(scores: pd.Series) -> pd.Series:
    x = scores.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return pd.Series(0.0, index=scores.index)
    y = np.exp(x - x.max())
    w = y / y.sum()
    out = pd.Series(0.0, index=scores.index)
    out.loc[w.index] = w
    return out


def vol_target_scale(weights: pd.Series, cov: pd.DataFrame, target_vol: float) -> float:
    common = [asset for asset in weights.index if asset in cov.index and asset in cov.columns]
    if not common:
        return 0.0

    w = weights.reindex(common).fillna(0.0).to_numpy(dtype=float)
    sigma = cov.reindex(index=common, columns=common).fillna(0.0).to_numpy(dtype=float)
    port_var = float(np.einsum("i,ij,j->", w, sigma, w))
    port_vol = np.sqrt(max(port_var, EPSILON))
    return target_vol / port_vol


def estimate_slippage_bps(order_dollars: float, adv_dollars: float) -> float:
    base = SLIPPAGE_BASE_BPS
    if adv_dollars <= 0:
        return SLIPPAGE_FALLBACK_BPS
    frac = abs(order_dollars) / adv_dollars
    impact = SLIPPAGE_IMPACT_COEFF * (frac ** SLIPPAGE_IMPACT_EXP)
    return base + impact
