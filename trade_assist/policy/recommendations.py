from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..ta.constants import COL_CLOSE, COL_VOLUME
from .constants import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    REASON_ADV_BELOW_TEMPLATE,
    REASON_BELOW_EMA200,
    REASON_LIQUIDITY_CAPPED_TEMPLATE,
    REASON_LIQUIDITY_DISALLOWS,
    REASON_NO_ADV,
    REASON_SCORE_TEMPLATE,
    REASON_SCORE_UNAVAILABLE,
    REASON_UNTRADABLE_PRICE,
    EPSILON,
)
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


def _apply_liquidity_to_target_shares(
    target_shares: pd.Series,
    current_shares: pd.Series,
    latest_prices: pd.Series,
    latest_adv_dollars: pd.Series,
    min_adv_dollars: float,
    max_trade_adv_fraction: float,
) -> tuple[pd.Series, dict[str, str]]:
    adjusted = target_shares.copy()
    notes: dict[str, str] = {}
    max_frac = max(float(max_trade_adv_fraction), 0.0)

    for ticker in adjusted.index:
        px = (
            float(latest_prices.loc[ticker])
            if pd.notna(latest_prices.loc[ticker])
            else np.nan
        )
        adv = (
            float(latest_adv_dollars.loc[ticker])
            if pd.notna(latest_adv_dollars.loc[ticker])
            else np.nan
        )
        if not np.isfinite(px) or px <= 0:
            adjusted.loc[ticker] = current_shares.loc[ticker]
            notes[ticker] = REASON_UNTRADABLE_PRICE
            continue

        delta = float(adjusted.loc[ticker] - current_shares.loc[ticker])
        if abs(delta) <= EPSILON:
            continue

        if not np.isfinite(adv) or adv <= 0:
            adjusted.loc[ticker] = current_shares.loc[ticker]
            notes[ticker] = REASON_NO_ADV
            continue

        if delta > 0 and min_adv_dollars > 0 and adv < min_adv_dollars:
            adjusted.loc[ticker] = current_shares.loc[ticker]
            notes[ticker] = REASON_ADV_BELOW_TEMPLATE.format(
                adv=adv, min_adv=min_adv_dollars
            )
            continue

        max_notional = adv * max_frac
        if max_notional <= 0:
            adjusted.loc[ticker] = current_shares.loc[ticker]
            notes[ticker] = REASON_LIQUIDITY_DISALLOWS
            continue

        requested_notional = abs(delta * px)
        if requested_notional > max_notional:
            capped_delta = np.sign(delta) * (max_notional / px)
            adjusted.loc[ticker] = current_shares.loc[ticker] + capped_delta
            notes[ticker] = REASON_LIQUIDITY_CAPPED_TEMPLATE.format(fraction=max_frac)

    return adjusted, notes


def _latest_target_weights(
    ohlcv_map: dict[str, pd.DataFrame],
    index_close: pd.Series | pd.DataFrame,
    config: PolicyConfig,
) -> tuple[pd.Series, int, dict[str, pd.DataFrame], pd.DataFrame]:
    tickers = list(ohlcv_map.keys())
    feats = {ticker: build_asset_features(ohlcv_map[ticker]) for ticker in tickers}
    scores = score_assets(feats, config.score_weights)

    close_df = pd.concat(
        {ticker: ohlcv_map[ticker][COL_CLOSE] for ticker in tickers}, axis=1
    )
    ret_df = close_df.pct_change()
    day = close_df.index[-1]

    regime = (
        compute_regime(index_close, config=config.regime)
        .reindex(close_df.index)
        .fillna(0)
        .astype(int)
    )
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
    index_close: pd.Series | pd.DataFrame,
    current_positions: dict[str, float],
    current_cash: float,
    config: PolicyConfig,
    min_trade_shares: float = 1.0,
) -> tuple[list[Recommendation], pd.Series, int]:
    tickers = list(ohlcv_map.keys())
    ticker_index = pd.Index(tickers, dtype=object)
    close_df = pd.concat(
        {ticker: ohlcv_map[ticker][COL_CLOSE] for ticker in tickers}, axis=1
    ).reindex(columns=ticker_index)
    volume_df = pd.concat(
        {
            ticker: (
                ohlcv_map[ticker][COL_VOLUME]
                if COL_VOLUME in ohlcv_map[ticker]
                else pd.Series(0.0, index=close_df.index)
            )
            for ticker in tickers
        },
        axis=1,
    ).reindex(columns=ticker_index)
    latest_day = close_df.index[-1]
    latest_prices = close_df.loc[latest_day]
    latest_adv_dollars = (volume_df.loc[latest_day] * latest_prices).fillna(0.0)

    current_shares = pd.Series(0.0, index=ticker_index)
    for ticker, shares in current_positions.items():
        if ticker in current_shares.index:
            current_shares.loc[ticker] = float(shares)

    equity = float(current_cash) + float((current_shares * latest_prices).sum())
    target_w, risk_on, feats, scores = _latest_target_weights(
        ohlcv_map, index_close, config
    )

    target_dollars = equity * target_w
    target_shares = (
        (target_dollars / latest_prices.replace(0.0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    target_shares, liquidity_notes = _apply_liquidity_to_target_shares(
        target_shares=target_shares,
        current_shares=current_shares,
        latest_prices=latest_prices,
        latest_adv_dollars=latest_adv_dollars,
        min_adv_dollars=config.liquidity.min_adv_dollars,
        max_trade_adv_fraction=config.liquidity.max_trade_adv_fraction,
    )

    recommendations: list[Recommendation] = []
    for ticker in tickers:
        current = float(current_shares.loc[ticker])
        target = float(target_shares.loc[ticker])
        delta = target - current

        if abs(delta) < min_trade_shares:
            action = ACTION_HOLD
        elif delta > 0:
            action = ACTION_BUY
        else:
            action = ACTION_SELL

        c_over_ema200 = float(feats[ticker].iloc[-1]["c_over_ema200"])
        score = (
            float(scores[ticker].iloc[-1])
            if pd.notna(scores[ticker].iloc[-1])
            else float("nan")
        )
        if not np.isfinite(c_over_ema200) or c_over_ema200 <= 0:
            reason = REASON_BELOW_EMA200
        elif np.isfinite(score):
            reason = REASON_SCORE_TEMPLATE.format(score=score)
        else:
            reason = REASON_SCORE_UNAVAILABLE
        if ticker in liquidity_notes:
            reason = f"{reason}; {liquidity_notes[ticker]}"

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
