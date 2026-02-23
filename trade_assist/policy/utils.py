from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import EPSILON, SHRINKAGE_LAMBDA, SHRINKAGE_MIN_SAMPLES, ZSCORE_MIN_PERIODS, ZSCORE_WINDOW


def zscore(series: pd.Series) -> pd.Series:
    mean = series.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).mean()
    std = series.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_MIN_PERIODS).std()
    return (series - mean) / (std + EPSILON)


def ledoit_wolf_shrinkage_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Simple shrinkage toward diagonal (heuristic implementation).
    """
    X = returns.dropna()
    if len(X) < SHRINKAGE_MIN_SAMPLES:
        return X.cov()
    sample = X.cov()
    diag = np.diag(np.diag(sample.values))
    lam = SHRINKAGE_LAMBDA
    shrunk = (1 - lam) * sample.values + lam * diag
    return pd.DataFrame(shrunk, index=sample.index, columns=sample.columns)
