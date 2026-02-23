from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PolicyConfig
from .features import build_asset_features, score_assets
from .portfolio import softmax_weights, vol_target_scale
from .regime import compute_regime
from .utils import ledoit_wolf_shrinkage_cov


@dataclass
class Recommendation:
    ticker: str
    action: str
    current_shares: float
    target_shares: float
    delta_shares: float
    reason: str


def _latest_target_weights(
    ohlcv_map: dict[str, pd.DataFrame],
    index_close: pd.Series,
    config: PolicyConfig,
) -> tuple[pd.Series, int, dict[str, pd.DataFrame], pd.DataFrame]:
    tickers = list(ohlcv_map.keys())
    feats = {ticker: build_asset_features(ohlcv_map[ticker]) for ticker in tickers}
    scores = score_assets(feats, config.score_weights)

    close_df = pd.concat({ticker: ohlcv_map[ticker]["Close"] for ticker in tickers}, axis=1)
    ret_df = close_df.pct_change()
    day = close_df.index[-1]

    regime = compute_regime(index_close).reindex(close_df.index).fillna(0).astype(int)
    risk_on = int(regime.loc[day])
    gross_cap = config.gross_cap_risk_on if risk_on else config.gross_cap_risk_off
    target_vol = (
        config.target_vol_risk_on
        if risk_on
        else config.target_vol_risk_on * config.risk_off_target_vol_multiplier
    )

    eligible = []
    for ticker in tickers:
        c_over_ema200 = feats[ticker].loc[day, "c_over_ema200"]
        if np.isfinite(c_over_ema200) and c_over_ema200 > 0:
            eligible.append(ticker)

    target_w = pd.Series(0.0, index=tickers)
    if not eligible:
        return target_w, risk_on, feats, scores

    s = scores.loc[day, eligible].dropna()
    if len(s) == 0:
        return target_w, risk_on, feats, scores

    w = softmax_weights(s)
    w = w.clip(upper=config.max_weight)
    if w.sum() > 0:
        w = w / w.sum()

    window = ret_df.loc[:, eligible].tail(config.covariance_lookback)
    cov = ledoit_wolf_shrinkage_cov(window)
    if cov.isna().values.any() or cov.empty:
        cov = window.cov().fillna(0.0)

    scale = vol_target_scale(w, cov, target_vol)
    w = w * min(scale, 1.0)
    if w.sum() > gross_cap:
        w = w * (gross_cap / w.sum())

    target_w.loc[w.index] = w
    return target_w, risk_on, feats, scores


def recommend_positions(
    ohlcv_map: dict[str, pd.DataFrame],
    index_close: pd.Series,
    current_positions: dict[str, float],
    current_cash: float,
    config: PolicyConfig,
    min_trade_shares: float = 1.0,
) -> tuple[list[Recommendation], pd.Series, int]:
    tickers = list(ohlcv_map.keys())
    close_df = pd.concat({ticker: ohlcv_map[ticker]["Close"] for ticker in tickers}, axis=1)
    latest_day = close_df.index[-1]
    latest_prices = close_df.loc[latest_day].reindex(tickers)

    current_shares = pd.Series(0.0, index=tickers)
    for ticker, shares in current_positions.items():
        if ticker in current_shares.index:
            current_shares.loc[ticker] = float(shares)

    equity = float(current_cash) + float((current_shares * latest_prices).sum())
    target_w, risk_on, feats, scores = _latest_target_weights(ohlcv_map, index_close, config)

    target_dollars = equity * target_w
    target_shares = (target_dollars / latest_prices.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    recommendations: list[Recommendation] = []
    for ticker in tickers:
        current = float(current_shares.loc[ticker])
        target = float(target_shares.loc[ticker])
        delta = target - current

        if abs(delta) < min_trade_shares:
            action = "HOLD"
        elif delta > 0:
            action = "BUY"
        else:
            action = "SELL"

        c_over_ema200 = float(feats[ticker].iloc[-1]["c_over_ema200"])
        score = float(scores[ticker].iloc[-1]) if pd.notna(scores[ticker].iloc[-1]) else float("nan")
        if not np.isfinite(c_over_ema200) or c_over_ema200 <= 0:
            reason = "Below EMA200 filter"
        elif np.isfinite(score):
            reason = f"Eligible with score {score:.3f}"
        else:
            reason = "Eligible but score unavailable"

        recommendations.append(
            Recommendation(
                ticker=ticker,
                action=action,
                current_shares=current,
                target_shares=target,
                delta_shares=delta,
                reason=reason,
            )
        )

    return recommendations, target_w, risk_on
