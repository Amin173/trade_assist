from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(series: pd.Series) -> pd.Series:
    mean = series.rolling(252, min_periods=60).mean()
    std = series.rolling(252, min_periods=60).std()
    return (series - mean) / (std + 1e-12)


def ledoit_wolf_shrinkage_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Simple shrinkage toward diagonal (heuristic implementation).
    """
    X = returns.dropna()
    if len(X) < 60:
        return X.cov()
    sample = X.cov()
    diag = np.diag(np.diag(sample.values))
    lam = 0.15
    shrunk = (1 - lam) * sample.values + lam * diag
    return pd.DataFrame(shrunk, index=sample.index, columns=sample.columns)
