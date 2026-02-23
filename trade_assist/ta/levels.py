from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def pivot_points(
    series: pd.Series, left: int = 3, right: int = 3, mode: str = "high"
) -> pd.Series:
    """
    Identify pivot highs or lows.
    mode = 'high' -> pivot high if value is strictly greater than neighbors.
    mode = 'low'  -> pivot low if value is strictly lower than neighbors.
    """
    pivots = pd.Series(index=series.index, dtype=float)

    for i in range(left, len(series) - right):
        window = series.iloc[i - left : i + right + 1]
        center = series.iloc[i]
        if mode == "high":
            if center == window.max() and (window == center).sum() == 1:
                pivots.iloc[i] = center
        elif mode == "low":
            if center == window.min() and (window == center).sum() == 1:
                pivots.iloc[i] = center
        else:
            raise ValueError("mode must be 'high' or 'low'")

    return pivots


def top_levels(pivots: pd.Series, k: int = 6, tol: float = 0.008) -> List[float]:
    vals = pivots.dropna().values
    if len(vals) == 0:
        return []

    vals_sorted = np.sort(vals)
    clusters: List[List[float]] = []

    for v in vals_sorted:
        placed = False
        for cluster in clusters:
            center = float(np.mean(cluster))
            if abs(v - center) / center <= tol:
                cluster.append(float(v))
                placed = True
                break
        if not placed:
            clusters.append([float(v)])

    clusters.sort(key=lambda c: (len(c), np.mean(c)), reverse=True)
    levels = [float(np.mean(c)) for c in clusters[:k]]
    return sorted(levels)
