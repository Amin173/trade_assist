from __future__ import annotations

import pandas as pd

from ..ta.constants import COL_CLOSE, COL_HIGH, COL_LOW
from ..ta.indicators import atr, ema, realized_volatility
from .constants import (
    ATR_WINDOW,
    BREAKOUT_WINDOW,
    EMA_FAST_WINDOW,
    EMA_SLOW_WINDOW,
    MOM_LONG_WINDOW,
    MOM_SHORT_WINDOW,
    VOL_ADJUST_EPSILON,
    VOL_WINDOW,
)
from .config import ScoreWeights
from .utils import zscore


def build_asset_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    close = ohlcv[COL_CLOSE]
    feats = pd.DataFrame(index=ohlcv.index)
    feats["mom_252_21"] = close.pct_change(MOM_LONG_WINDOW) - close.pct_change(
        MOM_SHORT_WINDOW
    )
    feats["c_over_ema200"] = close / ema(close, EMA_SLOW_WINDOW) - 1
    feats["ema50_over_ema200"] = (
        ema(close, EMA_FAST_WINDOW) / ema(close, EMA_SLOW_WINDOW) - 1
    )
    feats["vol20"] = realized_volatility(close, VOL_WINDOW)
    feats["atr14"] = atr(ohlcv[COL_HIGH], ohlcv[COL_LOW], close, ATR_WINDOW)
    feats["breakout55"] = (
        close >= close.rolling(BREAKOUT_WINDOW).max().shift(1)
    ).astype(int)
    return feats


def score_assets(
    feature_map: dict[str, pd.DataFrame], weights: ScoreWeights
) -> pd.DataFrame:
    tickers = list(feature_map.keys())
    idx = feature_map[tickers[0]].index
    score = pd.DataFrame(index=idx, columns=tickers, dtype=float)

    for ticker in tickers:
        f = feature_map[ticker]
        score[ticker] = (
            weights.c_over_ema200 * zscore(f["c_over_ema200"])
            + weights.ema50_over_ema200 * zscore(f["ema50_over_ema200"])
            + weights.mom_252_21 * zscore(f["mom_252_21"])
            + weights.vol_adjusted_momentum
            * zscore(f["mom_252_21"] / (f["vol20"] + VOL_ADJUST_EPSILON))
            + weights.breakout55 * f["breakout55"]
        )

    return score
