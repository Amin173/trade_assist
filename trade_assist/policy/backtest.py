from __future__ import annotations

import numpy as np
import pandas as pd

from ..ta.constants import COL_CLOSE, COL_OPEN, COL_VOLUME
from .constants import (
    BPS_DENOMINATOR,
    DEFAULT_INITIAL_CASH,
    EPSILON,
    EVENT_REBALANCE,
    EVENT_REBALANCE_AND_RISK_EXIT,
    EVENT_RISK_EXIT,
    ONE,
    REASON_STOP_LOSS,
    REASON_TRAILING_STOP,
    RISK_OFF_LABEL,
    RISK_ON_FLAG,
    RISK_ON_LABEL,
    START_DATE_SENTINEL,
    ZERO,
)
from .config import PolicyConfig
from .features import build_asset_features, score_assets
from .models import BacktestResult
from .portfolio import estimate_slippage_bps, softmax_weights, vol_target_scale
from .regime import compute_regime
from .utils import ledoit_wolf_shrinkage_cov


def _apply_liquidity_caps(
    trade_shares: pd.Series,
    fill_px: pd.Series,
    adv_dollars: pd.Series,
    min_adv_dollars: float,
    max_trade_adv_fraction: float,
) -> tuple[pd.Series, int]:
    capped = trade_shares.copy()
    touched = 0
    max_frac = max(float(max_trade_adv_fraction), ZERO)

    for ticker in capped.index:
        shares = float(capped.loc[ticker])
        if abs(shares) <= EPSILON:
            continue

        px = float(fill_px.loc[ticker]) if pd.notna(fill_px.loc[ticker]) else np.nan
        adv = float(adv_dollars.loc[ticker]) if pd.notna(adv_dollars.loc[ticker]) else np.nan
        if not np.isfinite(px) or px <= ZERO or not np.isfinite(adv) or adv <= ZERO:
            capped.loc[ticker] = ZERO
            touched += 1
            continue

        is_buy = shares > ZERO
        if is_buy and min_adv_dollars > ZERO and adv < min_adv_dollars:
            capped.loc[ticker] = ZERO
            touched += 1
            continue

        max_notional = adv * max_frac
        if max_notional <= ZERO:
            capped.loc[ticker] = ZERO
            touched += 1
            continue

        notional = abs(shares * px)
        if notional > max_notional:
            capped.loc[ticker] = shares * (max_notional / notional)
            touched += 1

    return capped, touched


def _enforce_min_trade_shares(
    trade_shares: pd.Series,
    min_trade_shares: float,
    exempt_tickers: set[str] | None = None,
) -> pd.Series:
    threshold = max(float(min_trade_shares), ZERO)
    if threshold <= ZERO:
        return trade_shares

    filtered = trade_shares.copy()
    for ticker in filtered.index:
        if exempt_tickers and ticker in exempt_tickers:
            continue
        if abs(float(filtered.loc[ticker])) < threshold:
            filtered.loc[ticker] = ZERO
    return filtered


def run_policy(
    ohlcv_map: dict[str, pd.DataFrame],
    index_close: pd.Series | pd.DataFrame,
    config: PolicyConfig | None = None,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    initial_positions: dict[str, float] | None = None,
) -> BacktestResult:
    cfg = config or PolicyConfig()
    tickers = list(ohlcv_map.keys())
    if not tickers:
        raise ValueError("ohlcv_map must include at least one ticker")

    dates = ohlcv_map[tickers[0]].index

    feats = {ticker: build_asset_features(ohlcv_map[ticker]) for ticker in tickers}
    scores = score_assets(feats, cfg.score_weights)

    regime = compute_regime(index_close, config=cfg.regime).reindex(dates).fillna(0).astype(int)

    close_df = pd.concat({ticker: ohlcv_map[ticker][COL_CLOSE] for ticker in tickers}, axis=1)
    open_df = pd.concat(
        {
            ticker: ohlcv_map[ticker][COL_OPEN] if COL_OPEN in ohlcv_map[ticker] else ohlcv_map[ticker][COL_CLOSE]
            for ticker in tickers
        },
        axis=1,
    )
    volume_df = pd.concat(
        {
            ticker: ohlcv_map[ticker][COL_VOLUME]
            if COL_VOLUME in ohlcv_map[ticker]
            else pd.Series(ZERO, index=dates)
            for ticker in tickers
        },
        axis=1,
    )

    ret_df = close_df.pct_change()
    next_fill_px = open_df.shift(-1)
    next_adv_dollars = volume_df.shift(-1) * next_fill_px

    cash = float(initial_cash)
    holdings = pd.Series(ZERO, index=tickers)
    if initial_positions:
        for ticker, shares in initial_positions.items():
            if ticker in holdings.index:
                holdings.loc[ticker] = float(shares)
    last_trade_day = pd.Series(pd.Timestamp(START_DATE_SENTINEL), index=tickers)
    entry_price = pd.Series(np.nan, index=tickers, dtype=float)
    peak_since_entry = pd.Series(np.nan, index=tickers, dtype=float)
    cooldown_until = pd.Series(pd.Timestamp("1900-01-01"), index=tickers)

    first_prices = close_df.iloc[0].reindex(tickers)
    for ticker in tickers:
        if holdings.loc[ticker] > 0:
            px0 = first_prices.loc[ticker]
            if pd.notna(px0) and float(px0) > ZERO:
                entry_price.loc[ticker] = float(px0)
                peak_since_entry.loc[ticker] = float(px0)

    account_rows: list[dict[str, float | pd.Timestamp]] = []
    rebalance_rows: list[dict[str, float | int | str]] = []

    rebalance_days = pd.Series(1, index=dates).resample(cfg.rebalance_freq).last().index
    rebalance_set = set(rebalance_days)

    for day in dates[:-1]:
        prices = close_df.loc[day].reindex(tickers)
        position_values = (holdings * prices).fillna(ZERO)
        equity = cash + float(position_values.sum())
        row: dict[str, float | pd.Timestamp] = {"date": day, "equity": float(equity), "cash": float(cash)}
        for ticker in tickers:
            row[f"{ticker}_value"] = float(position_values.loc[ticker])
        account_rows.append(row)

        forced_exit_reasons: dict[str, str] = {}
        if cfg.risk_exit.stop_loss_pct > ZERO or cfg.risk_exit.trailing_stop_pct > ZERO:
            for ticker in tickers:
                if holdings.loc[ticker] <= ZERO:
                    continue
                px = prices.loc[ticker]
                if pd.isna(px) or float(px) <= ZERO:
                    continue
                px_val = float(px)

                if pd.isna(peak_since_entry.loc[ticker]) or float(peak_since_entry.loc[ticker]) <= ZERO:
                    peak_since_entry.loc[ticker] = px_val
                else:
                    peak_since_entry.loc[ticker] = max(float(peak_since_entry.loc[ticker]), px_val)

                stop_hit = False
                trail_hit = False
                if cfg.risk_exit.stop_loss_pct > ZERO:
                    basis = entry_price.loc[ticker]
                    if pd.notna(basis) and float(basis) > ZERO:
                        stop_hit = px_val <= float(basis) * (ONE - cfg.risk_exit.stop_loss_pct)

                if cfg.risk_exit.trailing_stop_pct > ZERO:
                    peak = peak_since_entry.loc[ticker]
                    if pd.notna(peak) and float(peak) > ZERO:
                        trail_hit = px_val <= float(peak) * (ONE - cfg.risk_exit.trailing_stop_pct)

                if stop_hit or trail_hit:
                    reason_parts: list[str] = []
                    if stop_hit:
                        reason_parts.append(REASON_STOP_LOSS)
                    if trail_hit:
                        reason_parts.append(REASON_TRAILING_STOP)
                    forced_exit_reasons[ticker] = "+".join(reason_parts)

        do_rebalance = day in rebalance_set
        if not do_rebalance and not forced_exit_reasons:
            continue

        cash_before = cash
        risk_on = int(regime.loc[day]) == RISK_ON_FLAG
        gross_cap = cfg.gross_cap_risk_on if risk_on else cfg.gross_cap_risk_off
        target_vol = cfg.target_vol_risk_on if risk_on else cfg.target_vol_risk_on * cfg.risk_off_target_vol_multiplier

        target_shares = holdings.copy()
        eligible: list[str] = []
        if do_rebalance:
            target_shares = pd.Series(ZERO, index=tickers)
            for ticker in tickers:
                in_cooldown = holdings.loc[ticker] <= ZERO and day < cooldown_until.loc[ticker]
                if in_cooldown:
                    continue
                val = feats[ticker].loc[day, "c_over_ema200"]
                if np.isfinite(val) and val > 0:
                    eligible.append(ticker)

            if eligible:
                s = scores.loc[day, eligible].dropna()
                if len(s) > 0:
                    w = softmax_weights(s)
                    w = w.clip(upper=cfg.max_weight)
                    if w.sum() > ZERO:
                        w = w / w.sum()

                    window = ret_df.loc[:day, eligible].tail(cfg.covariance_lookback)
                    cov = ledoit_wolf_shrinkage_cov(window)
                    if cov.isna().values.any() or cov.empty:
                        cov = window.cov().fillna(ZERO)

                    scale = vol_target_scale(w, cov, target_vol)
                    w = w * min(scale, ONE)

                    if w.sum() > gross_cap:
                        w = w * (gross_cap / w.sum())

                    fill_px_eligible = next_fill_px.loc[day, eligible]
                    tradable = fill_px_eligible[fill_px_eligible > ZERO].dropna()
                    if len(tradable) > ZERO:
                        w = w.reindex(tradable.index).fillna(ZERO)
                        target_dollars = equity * w
                        target_shares.loc[tradable.index] = target_dollars / tradable

        for ticker in forced_exit_reasons:
            target_shares.loc[ticker] = ZERO

        for ticker in tickers:
            if ticker in forced_exit_reasons:
                continue
            if (day - last_trade_day[ticker]).days < cfg.min_hold_days:
                target_shares.loc[ticker] = holdings.loc[ticker]

        fill_px_all = next_fill_px.loc[day].reindex(tickers)
        tradeable_mask = (fill_px_all > ZERO) & fill_px_all.notna()
        target_shares.loc[~tradeable_mask] = holdings.loc[~tradeable_mask]

        trade_shares = target_shares - holdings
        trade_shares = _enforce_min_trade_shares(
            trade_shares=trade_shares,
            min_trade_shares=cfg.min_trade_shares,
            exempt_tickers=set(forced_exit_reasons.keys()),
        )
        adv = next_adv_dollars.loc[day].reindex(tickers).fillna(ZERO)

        trade_shares, liquidity_touched = _apply_liquidity_caps(
            trade_shares=trade_shares,
            fill_px=fill_px_all,
            adv_dollars=adv,
            min_adv_dollars=cfg.liquidity.min_adv_dollars,
            max_trade_adv_fraction=cfg.liquidity.max_trade_adv_fraction,
        )
        trade_shares = _enforce_min_trade_shares(
            trade_shares=trade_shares,
            min_trade_shares=cfg.min_trade_shares,
            exempt_tickers=set(forced_exit_reasons.keys()),
        )
        target_shares = holdings + trade_shares

        trade_notional = (trade_shares * fill_px_all).fillna(ZERO)
        buy_notional = float(trade_notional[trade_notional > 0].sum())
        sell_notional = float(-trade_notional[trade_notional < 0].sum())

        buy_slippage = ZERO
        sell_slippage = ZERO
        for ticker in tickers:
            if abs(float(trade_shares.loc[ticker])) <= EPSILON:
                continue
            notional = abs(float(trade_notional.loc[ticker]))
            bps = estimate_slippage_bps(notional, float(adv.loc[ticker]))
            slip = notional * (bps / BPS_DENOMINATOR)
            if trade_notional.loc[ticker] > ZERO:
                buy_slippage += slip
            else:
                sell_slippage += slip

        # Keep cash non-negative in this no-leverage baseline by scaling buys down
        # after accounting for expected slippage and same-day sell proceeds.
        available_for_buys = cash + sell_notional - sell_slippage
        estimated_buy_cost = buy_notional + buy_slippage
        buy_scale = ONE
        if available_for_buys <= ZERO:
            buy_scale = ZERO
        elif estimated_buy_cost > available_for_buys and estimated_buy_cost > ZERO:
            buy_scale = max(ZERO, min(ONE, available_for_buys / estimated_buy_cost))

        if buy_scale < ONE:
            buy_mask = trade_shares > ZERO
            trade_shares.loc[buy_mask] = trade_shares.loc[buy_mask] * buy_scale
            trade_shares = _enforce_min_trade_shares(
                trade_shares=trade_shares,
                min_trade_shares=cfg.min_trade_shares,
                exempt_tickers=set(forced_exit_reasons.keys()),
            )
            target_shares = holdings + trade_shares
            trade_notional = (trade_shares * fill_px_all).fillna(ZERO)

            buy_notional = float(trade_notional[trade_notional > 0].sum())
            sell_notional = float(-trade_notional[trade_notional < 0].sum())
            buy_slippage = ZERO
            sell_slippage = ZERO
            for ticker in tickers:
                if abs(float(trade_shares.loc[ticker])) <= EPSILON:
                    continue
                notional = abs(float(trade_notional.loc[ticker]))
                bps = estimate_slippage_bps(notional, float(adv.loc[ticker]))
                slip = notional * (bps / BPS_DENOMINATOR)
                if trade_notional.loc[ticker] > ZERO:
                    buy_slippage += slip
                else:
                    sell_slippage += slip

        slippage_cost = buy_slippage + sell_slippage
        old_holdings = holdings.copy()
        cash -= float(trade_notional.sum())
        cash -= slippage_cost
        cash = max(cash, ZERO)
        holdings = target_shares
        traded = trade_shares[trade_shares.abs() > EPSILON].index
        last_trade_day.loc[traded] = day

        for ticker in tickers:
            prev_shares = float(old_holdings.loc[ticker])
            new_shares = float(holdings.loc[ticker])
            delta = float(trade_shares.loc[ticker])
            fill = fill_px_all.loc[ticker]
            day_px = prices.loc[ticker]
            prev_entry = entry_price.loc[ticker]

            if new_shares <= ZERO:
                entry_price.loc[ticker] = np.nan
                peak_since_entry.loc[ticker] = np.nan
                if ticker in forced_exit_reasons and prev_shares > ZERO and cfg.risk_exit.cooldown_days > ZERO:
                    cooldown_until.loc[ticker] = day + pd.Timedelta(days=cfg.risk_exit.cooldown_days)
                continue

            if prev_shares <= ZERO and new_shares > ZERO:
                if pd.notna(fill) and float(fill) > ZERO:
                    entry_price.loc[ticker] = float(fill)
                    peak_since_entry.loc[ticker] = float(fill)
                elif pd.notna(day_px) and float(day_px) > ZERO:
                    entry_price.loc[ticker] = float(day_px)
                    peak_since_entry.loc[ticker] = float(day_px)
                continue

            if delta > ZERO and pd.notna(fill) and float(fill) > ZERO:
                if pd.notna(prev_entry) and float(prev_entry) > ZERO:
                    entry_price.loc[ticker] = (
                        (prev_shares * float(prev_entry)) + (delta * float(fill))
                    ) / max(new_shares, EPSILON)
                else:
                    entry_price.loc[ticker] = float(fill)

            if pd.notna(day_px) and float(day_px) > ZERO:
                if pd.isna(peak_since_entry.loc[ticker]) or float(peak_since_entry.loc[ticker]) <= ZERO:
                    peak_since_entry.loc[ticker] = float(day_px)
                else:
                    peak_since_entry.loc[ticker] = max(float(peak_since_entry.loc[ticker]), float(day_px))

        event = (
            EVENT_REBALANCE_AND_RISK_EXIT
            if (do_rebalance and forced_exit_reasons)
            else (EVENT_REBALANCE if do_rebalance else EVENT_RISK_EXIT)
        )
        risk_on = int(regime.loc[day]) == RISK_ON_FLAG
        rebalance_rows.append(
            {
                "date": str(day.date()),
                "event": event,
                "regime": RISK_ON_LABEL if risk_on else RISK_OFF_LABEL,
                "equity_before": float(equity),
                "cash_before": float(cash_before),
                "cash_after": float(cash),
                "eligible_count": int(len(eligible)),
                "trade_count": int((trade_shares.abs() > EPSILON).sum()),
                "risk_exit_count": int(len(forced_exit_reasons)),
                "risk_exit_tickers": ",".join(sorted(forced_exit_reasons.keys())),
                "liquidity_capped_count": int(liquidity_touched),
                "buy_notional": float(buy_notional),
                "sell_notional": float(sell_notional),
                "slippage_cost": float(slippage_cost),
                "buy_scale": float(buy_scale),
            }
        )

    final_day = dates[-1]
    final_prices = close_df.loc[final_day].reindex(tickers)
    final_position_values = (holdings * final_prices).fillna(ZERO)
    final_equity = cash + float(final_position_values.sum())
    final_row: dict[str, float | pd.Timestamp] = {"date": final_day, "equity": float(final_equity), "cash": float(cash)}
    for ticker in tickers:
        final_row[f"{ticker}_value"] = float(final_position_values.loc[ticker])
    account_rows.append(final_row)

    account_history = pd.DataFrame(account_rows).set_index("date").sort_index()
    equity_curve = account_history["equity"].rename("equity")
    rebalance_log = pd.DataFrame(rebalance_rows)
    return BacktestResult(
        equity_curve=equity_curve,
        final_holdings=holdings,
        final_cash=float(cash),
        regime=regime,
        rebalance_log=rebalance_log,
        account_history=account_history,
    )
