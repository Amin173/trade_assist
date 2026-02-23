from __future__ import annotations

import pandas as pd

from ..ta.indicators import atr, ema, realized_volatility
from .config import ScoreWeights
from .utils import zscore


def build_asset_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    close = ohlcv["Close"]
    feats = pd.DataFrame(index=ohlcv.index)
    feats["mom_252_21"] = close.pct_change(252) - close.pct_change(21)
    feats["c_over_ema200"] = close / ema(close, 200) - 1
    feats["ema50_over_ema200"] = ema(close, 50) / ema(close, 200) - 1
    feats["vol20"] = realized_volatility(close, 20)
    feats["atr14"] = atr(ohlcv["High"], ohlcv["Low"], close, 14)
    feats["breakout55"] = (close >= close.rolling(55).max().shift(1)).astype(int)
    return feats


def score_assets(feature_map: dict[str, pd.DataFrame], weights: ScoreWeights) -> pd.DataFrame:
    tickers = list(feature_map.keys())
    idx = feature_map[tickers[0]].index
    score = pd.DataFrame(index=idx, columns=tickers, dtype=float)

    for ticker in tickers:
        f = feature_map[ticker]
        score[ticker] = (
            weights.c_over_ema200 * zscore(f["c_over_ema200"])
            + weights.ema50_over_ema200 * zscore(f["ema50_over_ema200"])
            + weights.mom_252_21 * zscore(f["mom_252_21"])
            + weights.vol_adjusted_momentum * zscore(f["mom_252_21"] / (f["vol20"] + 1e-9))
            + weights.breakout55 * f["breakout55"]
        )

    return score
