from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PolicyConfig
from .features import build_asset_features, score_assets
from .models import BacktestResult
from .portfolio import estimate_slippage_bps, softmax_weights, vol_target_scale
from .regime import compute_regime
from .utils import ledoit_wolf_shrinkage_cov


def run_policy(
    ohlcv_map: dict[str, pd.DataFrame],
    index_close: pd.Series,
    config: PolicyConfig | None = None,
    initial_cash: float = 100_000.0,
    initial_positions: dict[str, float] | None = None,
) -> BacktestResult:
    cfg = config or PolicyConfig()
    tickers = list(ohlcv_map.keys())
    if not tickers:
        raise ValueError("ohlcv_map must include at least one ticker")

    dates = ohlcv_map[tickers[0]].index

    feats = {ticker: build_asset_features(ohlcv_map[ticker]) for ticker in tickers}
    scores = score_assets(feats, cfg.score_weights)

    regime = compute_regime(index_close).reindex(dates).fillna(0).astype(int)

    close_df = pd.concat({ticker: ohlcv_map[ticker]["Close"] for ticker in tickers}, axis=1)
    open_df = pd.concat(
        {
            ticker: ohlcv_map[ticker]["Open"] if "Open" in ohlcv_map[ticker] else ohlcv_map[ticker]["Close"]
            for ticker in tickers
        },
        axis=1,
    )
    volume_df = pd.concat(
        {
            ticker: ohlcv_map[ticker]["Volume"] if "Volume" in ohlcv_map[ticker] else pd.Series(0.0, index=dates)
            for ticker in tickers
        },
        axis=1,
    )

    ret_df = close_df.pct_change()
    next_fill_px = open_df.shift(-1)
    next_adv_dollars = volume_df.shift(-1) * next_fill_px

    cash = float(initial_cash)
    holdings = pd.Series(0.0, index=tickers)
    if initial_positions:
        for ticker, shares in initial_positions.items():
            if ticker in holdings.index:
                holdings.loc[ticker] = float(shares)
    last_trade_day = pd.Series(pd.Timestamp("1900-01-01"), index=tickers)
    account_rows: list[dict[str, float | pd.Timestamp]] = []
    rebalance_rows: list[dict[str, float | int | str]] = []

    rebalance_days = pd.Series(1, index=dates).resample(cfg.rebalance_freq).last().index
    rebalance_set = set(rebalance_days)

    for day in dates[:-1]:
        prices = close_df.loc[day].reindex(tickers)
        position_values = (holdings * prices).fillna(0.0)
        equity = cash + float(position_values.sum())
        row: dict[str, float | pd.Timestamp] = {"date": day, "equity": float(equity), "cash": float(cash)}
        for ticker in tickers:
            row[f"{ticker}_value"] = float(position_values.loc[ticker])
        account_rows.append(row)

        if day not in rebalance_set:
            continue

        risk_on = int(regime.loc[day]) == 1
        gross_cap = cfg.gross_cap_risk_on if risk_on else cfg.gross_cap_risk_off
        target_vol = cfg.target_vol_risk_on if risk_on else cfg.target_vol_risk_on * cfg.risk_off_target_vol_multiplier
        cash_before = cash

        eligible: list[str] = []
        for ticker in tickers:
            val = feats[ticker].loc[day, "c_over_ema200"]
            if np.isfinite(val) and val > 0:
                eligible.append(ticker)

        target_shares = pd.Series(0.0, index=tickers)
        if eligible:
            s = scores.loc[day, eligible].dropna()
            if len(s) > 0:
                w = softmax_weights(s)
                w = w.clip(upper=cfg.max_weight)
                if w.sum() > 0:
                    w = w / w.sum()

                window = ret_df.loc[:day, eligible].tail(cfg.covariance_lookback)
                cov = ledoit_wolf_shrinkage_cov(window)
                if cov.isna().values.any() or cov.empty:
                    cov = window.cov().fillna(0.0)

                scale = vol_target_scale(w, cov, target_vol)
                w = w * min(scale, 1.0)

                if w.sum() > gross_cap:
                    w = w * (gross_cap / w.sum())

                fill_px_eligible = next_fill_px.loc[day, eligible]
                tradable = fill_px_eligible[fill_px_eligible > 0].dropna()
                if len(tradable) > 0:
                    w = w.reindex(tradable.index).fillna(0.0)
                    target_dollars = equity * w
                    target_shares.loc[tradable.index] = target_dollars / tradable

        for ticker in tickers:
            if (day - last_trade_day[ticker]).days < cfg.min_hold_days:
                target_shares.loc[ticker] = holdings.loc[ticker]

        fill_px_all = next_fill_px.loc[day].reindex(tickers)
        tradeable_mask = (fill_px_all > 0) & fill_px_all.notna()
        target_shares.loc[~tradeable_mask] = holdings.loc[~tradeable_mask]

        trade_shares = target_shares - holdings
        trade_notional = (trade_shares * fill_px_all).fillna(0.0)

        adv = next_adv_dollars.loc[day].reindex(tickers).fillna(0.0)
        buy_notional = float(trade_notional[trade_notional > 0].sum())
        sell_notional = float(-trade_notional[trade_notional < 0].sum())

        buy_slippage = 0.0
        sell_slippage = 0.0
        for ticker in tickers:
            if trade_shares.loc[ticker] == 0:
                continue
            notional = abs(float(trade_notional.loc[ticker]))
            bps = estimate_slippage_bps(notional, float(adv.loc[ticker]))
            slip = notional * (bps / 10_000.0)
            if trade_notional.loc[ticker] > 0:
                buy_slippage += slip
            else:
                sell_slippage += slip

        # Keep cash non-negative in this no-leverage baseline by scaling buys down
        # after accounting for expected slippage and same-day sell proceeds.
        available_for_buys = cash + sell_notional - sell_slippage
        estimated_buy_cost = buy_notional + buy_slippage
        buy_scale = 1.0
        if available_for_buys <= 0:
            buy_scale = 0.0
        elif estimated_buy_cost > available_for_buys and estimated_buy_cost > 0:
            buy_scale = max(0.0, min(1.0, available_for_buys / estimated_buy_cost))

        if buy_scale < 1.0:
            buy_mask = trade_shares > 0
            trade_shares.loc[buy_mask] = trade_shares.loc[buy_mask] * buy_scale
            target_shares = holdings + trade_shares
            trade_notional = (trade_shares * fill_px_all).fillna(0.0)

            buy_notional = float(trade_notional[trade_notional > 0].sum())
            sell_notional = float(-trade_notional[trade_notional < 0].sum())
            buy_slippage = 0.0
            sell_slippage = 0.0
            for ticker in tickers:
                if trade_shares.loc[ticker] == 0:
                    continue
                notional = abs(float(trade_notional.loc[ticker]))
                bps = estimate_slippage_bps(notional, float(adv.loc[ticker]))
                slip = notional * (bps / 10_000.0)
                if trade_notional.loc[ticker] > 0:
                    buy_slippage += slip
                else:
                    sell_slippage += slip

        slippage_cost = buy_slippage + sell_slippage

        cash -= float(trade_notional.sum())
        cash -= slippage_cost
        cash = max(cash, 0.0)
        holdings = target_shares
        traded = trade_shares[trade_shares != 0.0].index
        last_trade_day.loc[traded] = day

        rebalance_rows.append(
            {
                "date": str(day.date()),
                "regime": "risk_on" if risk_on else "risk_off",
                "equity_before": float(equity),
                "cash_before": float(cash_before),
                "cash_after": float(cash),
                "eligible_count": int(len(eligible)),
                "trade_count": int((trade_shares != 0).sum()),
                "buy_notional": float(buy_notional),
                "sell_notional": float(sell_notional),
                "slippage_cost": float(slippage_cost),
                "buy_scale": float(buy_scale),
            }
        )

    final_day = dates[-1]
    final_prices = close_df.loc[final_day].reindex(tickers)
    final_position_values = (holdings * final_prices).fillna(0.0)
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
