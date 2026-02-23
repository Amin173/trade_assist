from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config_validation import validate_config, validate_policy
from .policy.constants import ANNUAL_TRADING_DAYS, ONE, RISK_ON_FLAG, ZERO
from .policy import PolicyConfig, run_policy
from .policy.recommendations import recommend_positions
from .ta.constants import (
    COL_CLOSE,
    DEFAULT_CACHE_DIR,
    DEFAULT_CACHE_TTL_HOURS,
    DEFAULT_INDEX_TICKER,
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
)
from .ta.data import fetch_ohlcv_map

KEY_DATA = "data"
KEY_TICKERS = "tickers"
KEY_PORTFOLIO = "portfolio"
KEY_CASH = "cash"
KEY_POSITIONS = "positions"
KEY_POLICY = "policy"
KEY_POLICY_PATH = "policy_path"
KEY_OUTPUT = "output"
KEY_RECOMMENDATION = "recommendation"

KEY_INDEX_TICKER = "index_ticker"
KEY_REGIME_TICKERS = "regime_tickers"
KEY_PERIOD = "period"
KEY_INTERVAL = "interval"
KEY_USE_CACHE = "use_cache"
KEY_CACHE_DIR = "cache_dir"
KEY_CACHE_TTL_HOURS = "cache_ttl_hours"
KEY_FORCE_REFRESH = "force_refresh"

DEFAULT_MIN_TRADE_SHARES = ONE
DEFAULT_REBALANCE_LOG_TAIL = 20
DEFAULT_DAYS_PER_YEAR = 365.25
INVESTED_EPSILON = 1e-9
PLOT_DPI = 140
PERCENT_SCALE = 100.0
TRADE_MARKER_SIZE = 90
TRADE_MARKER_ALPHA = 0.95
TRADE_MARKER_EDGE_WIDTH = 0.8
TRADE_MARKER_OFFSET_RATIO = 0.01

LABEL_RISK_ON = "RISK_ON"
LABEL_RISK_OFF = "RISK_OFF"
NO_TARGET_EXPOSURE_TEXT = "No target exposure (all cash)"


def _load_config(path: str) -> tuple[dict[str, Any], Path]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f), cfg_path


def _require_keys(payload: dict[str, Any], required: list[str], scope: str = "config") -> None:
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing required {scope} key(s): {', '.join(missing)}")


def _extract_data_cfg(
    payload: dict[str, Any],
) -> tuple[list[str], str, list[str], str, str, bool, str, float | None, bool]:
    _require_keys(payload, [KEY_DATA], scope="config")
    data_cfg = payload[KEY_DATA]
    _require_keys(data_cfg, [KEY_TICKERS], scope="data")

    tickers = data_cfg[KEY_TICKERS]
    if not isinstance(tickers, list) or not tickers:
        raise ValueError("data.tickers must be a non-empty list of symbols")

    index_ticker = data_cfg.get(KEY_INDEX_TICKER, DEFAULT_INDEX_TICKER)
    regime_tickers_raw = data_cfg.get(KEY_REGIME_TICKERS, [])
    if regime_tickers_raw is None:
        regime_tickers_raw = []
    if not isinstance(regime_tickers_raw, list):
        raise ValueError("data.regime_tickers must be a list of symbols when provided")
    regime_tickers = [str(symbol) for symbol in regime_tickers_raw if str(symbol)]
    period = data_cfg.get(KEY_PERIOD, DEFAULT_PERIOD)
    interval = data_cfg.get(KEY_INTERVAL, DEFAULT_INTERVAL)
    use_cache = bool(data_cfg.get(KEY_USE_CACHE, False))
    cache_dir = str(data_cfg.get(KEY_CACHE_DIR, DEFAULT_CACHE_DIR))
    cache_ttl_hours_raw = data_cfg.get(KEY_CACHE_TTL_HOURS, DEFAULT_CACHE_TTL_HOURS)
    cache_ttl_hours = None if cache_ttl_hours_raw is None else float(cache_ttl_hours_raw)
    force_refresh = bool(data_cfg.get(KEY_FORCE_REFRESH, False))
    return (
        tickers,
        index_ticker,
        regime_tickers,
        period,
        interval,
        use_cache,
        cache_dir,
        cache_ttl_hours,
        force_refresh,
    )


def _load_market_data(
    payload: dict[str, Any],
    config_path: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.Series | pd.DataFrame]:
    (
        tickers,
        index_ticker,
        regime_tickers,
        period,
        interval,
        use_cache,
        cache_dir,
        cache_ttl_hours,
        force_refresh,
    ) = _extract_data_cfg(payload)
    cache_root = Path(cache_dir)
    if config_path is not None and not cache_root.is_absolute():
        cache_root = (config_path.parent / cache_root).resolve()

    ohlcv_map = fetch_ohlcv_map(
        tickers=tickers,
        period=period,
        interval=interval,
        use_cache=use_cache,
        cache_dir=cache_root,
        cache_ttl_hours=cache_ttl_hours,
        force_refresh=force_refresh,
    )
    regime_symbols = [index_ticker]
    regime_symbols.extend([symbol for symbol in regime_tickers if symbol != index_ticker])
    regime_map = fetch_ohlcv_map(
        regime_symbols,
        period=period,
        interval=interval,
        use_cache=use_cache,
        cache_dir=cache_root,
        cache_ttl_hours=cache_ttl_hours,
        force_refresh=force_refresh,
    )
    regime_close = pd.concat({symbol: regime_map[symbol][COL_CLOSE] for symbol in regime_symbols}, axis=1)
    if regime_close.shape[1] == 1:
        return ohlcv_map, regime_close.iloc[:, 0]
    return ohlcv_map, regime_close


def _resolve_policy_config(payload: dict[str, Any], config_path: Path) -> PolicyConfig:
    policy_path = payload.get(KEY_POLICY_PATH)
    if policy_path is not None:
        path = Path(str(policy_path))
        if not path.is_absolute():
            path = config_path.parent / path
        if not path.exists():
            raise ValueError(f"policy_path not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            policy_payload = json.load(f)
        if not isinstance(policy_payload, dict):
            raise ValueError(f"policy_path must point to a JSON object: {path}")
        validate_policy(policy_payload)
        return PolicyConfig.from_dict(policy_payload)

    if KEY_POLICY in payload:
        policy_payload = payload.get(KEY_POLICY, {})
        if not isinstance(policy_payload, dict):
            raise ValueError("config.policy must be a JSON object")
        validate_policy(policy_payload)
        return PolicyConfig.from_dict(policy_payload)

    return PolicyConfig()


def _format_recommendations(rows: list) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": r.ticker,
                "action": r.action,
                "current_shares": round(r.current_shares, 4),
                "target_shares": round(r.target_shares, 4),
                "delta_shares": round(r.delta_shares, 4),
                "reason": r.reason,
            }
            for r in rows
        ]
    )


def _compute_performance_stats(equity_curve: pd.Series) -> dict[str, float]:
    eq = equity_curve.dropna()
    if len(eq) < 2:
        return {
            "total_return_pct": ZERO,
            "cagr_pct": ZERO,
            "annualized_vol_pct": ZERO,
            "sharpe": ZERO,
            "sortino": ZERO,
            "max_drawdown_pct": ZERO,
            "calmar": ZERO,
            "win_rate_pct": ZERO,
            "best_day_pct": ZERO,
            "worst_day_pct": ZERO,
        }

    returns = eq.pct_change().dropna()
    if len(returns) == 0:
        returns = pd.Series([ZERO], index=eq.index[:1])

    total_return = float(eq.iloc[-1] / eq.iloc[0] - ONE)
    duration_days = max((eq.index[-1] - eq.index[0]).days, 1)
    years = duration_days / DEFAULT_DAYS_PER_YEAR
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (ONE / years) - ONE) if eq.iloc[0] > ZERO else ZERO

    annualized_vol = float(returns.std() * np.sqrt(ANNUAL_TRADING_DAYS))
    annualized_mean = float(returns.mean() * ANNUAL_TRADING_DAYS)
    sharpe = float(annualized_mean / annualized_vol) if annualized_vol > ZERO else ZERO

    downside = returns[returns < 0]
    downside_vol = float(downside.std() * np.sqrt(ANNUAL_TRADING_DAYS)) if len(downside) > 1 else ZERO
    sortino = float(annualized_mean / downside_vol) if downside_vol > ZERO else ZERO

    running_max = eq.cummax()
    drawdown = eq / running_max - ONE
    max_drawdown = float(drawdown.min())
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown < ZERO else ZERO

    win_rate = float((returns > 0).mean())
    best_day = float(returns.max())
    worst_day = float(returns.min())

    return {
        "total_return_pct": total_return * PERCENT_SCALE,
        "cagr_pct": cagr * PERCENT_SCALE,
        "annualized_vol_pct": annualized_vol * PERCENT_SCALE,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_drawdown * PERCENT_SCALE,
        "calmar": calmar,
        "win_rate_pct": win_rate * PERCENT_SCALE,
        "best_day_pct": best_day * PERCENT_SCALE,
        "worst_day_pct": worst_day * PERCENT_SCALE,
    }


def _print_performance_stats(stats: dict[str, float]) -> None:
    print("\nPerformance stats:")
    print(f"- total_return: {stats['total_return_pct']:.2f}%")
    print(f"- cagr: {stats['cagr_pct']:.2f}%")
    print(f"- annualized_vol: {stats['annualized_vol_pct']:.2f}%")
    print(f"- sharpe: {stats['sharpe']:.3f}")
    print(f"- sortino: {stats['sortino']:.3f}")
    print(f"- max_drawdown: {stats['max_drawdown_pct']:.2f}%")
    print(f"- calmar: {stats['calmar']:.3f}")
    print(f"- win_rate: {stats['win_rate_pct']:.2f}%")
    print(f"- best_day: {stats['best_day_pct']:.2f}%")
    print(f"- worst_day: {stats['worst_day_pct']:.2f}%")


def _compute_full_hold_benchmarks(
    ohlcv_map: dict[str, pd.DataFrame],
    index: pd.Index,
    start_equity: float,
    start_date: pd.Timestamp,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    selected = tickers if tickers else list(ohlcv_map.keys())
    out = pd.DataFrame(index=index)

    for ticker in selected:
        if ticker not in ohlcv_map:
            continue
        close = ohlcv_map[ticker][COL_CLOSE].reindex(index).ffill()
        if close.empty or start_date not in close.index:
            continue
        base = close.loc[start_date]
        if pd.isna(base) or base <= 0:
            continue
        series = pd.Series(np.nan, index=index, dtype=float)
        active_mask = series.index >= start_date
        series.loc[active_mask] = start_equity * (close.loc[active_mask] / base)
        out[f"{ticker}_full_hold"] = series
    return out


def _handle_backtest_output(backtest, output_cfg: dict[str, Any], ohlcv_map: dict[str, pd.DataFrame]) -> None:
    verbose = bool(output_cfg.get("verbose", False))
    print_performance_stats = bool(output_cfg.get("print_performance_stats", True))
    print_rebalance_log = bool(output_cfg.get("print_rebalance_log", False))
    rebalance_tail = int(output_cfg.get("rebalance_log_tail", DEFAULT_REBALANCE_LOG_TAIL))
    save_equity_csv = output_cfg.get("save_equity_curve_csv")
    save_account_history_csv = output_cfg.get("save_account_history_csv")
    save_position_values_csv = output_cfg.get("save_position_values_csv")
    save_full_hold_benchmarks_csv = output_cfg.get("save_full_hold_benchmarks_csv")
    save_rebalance_csv = output_cfg.get("save_rebalance_log_csv")
    save_stats_json = output_cfg.get("save_performance_stats_json")
    plot_equity_curve = bool(output_cfg.get("plot_equity_curve", False))
    plot_trade_markers = bool(output_cfg.get("plot_trade_markers", False))
    plot_full_hold_benchmarks = bool(output_cfg.get("plot_full_hold_benchmarks", False))
    full_hold_benchmark_tickers = output_cfg.get("full_hold_benchmark_tickers")
    equity_plot_path = output_cfg.get("equity_curve_plot_path")
    position_value_cols = [col for col in backtest.account_history.columns if str(col).endswith("_value")]
    benchmark_tickers = (
        [str(x) for x in full_hold_benchmark_tickers]
        if isinstance(full_hold_benchmark_tickers, list)
        else None
    )
    full_hold_benchmarks = pd.DataFrame(index=backtest.account_history.index)
    if plot_full_hold_benchmarks or save_full_hold_benchmarks_csv:
        account = backtest.account_history
        benchmark_start_date = account.index[0]
        if position_value_cols:
            invested = account[position_value_cols].sum(axis=1)
            invested_mask = invested.abs() > INVESTED_EPSILON
            if invested_mask.any():
                benchmark_start_date = invested_mask[invested_mask].index[0]
        start_equity = float(account.loc[benchmark_start_date, "equity"]) if len(account) else ZERO
        full_hold_benchmarks = _compute_full_hold_benchmarks(
            ohlcv_map=ohlcv_map,
            index=account.index,
            start_equity=start_equity,
            start_date=benchmark_start_date,
            tickers=benchmark_tickers,
        )
        if verbose:
            print(f"Full-hold benchmark start date: {benchmark_start_date.date()}")

    stats = _compute_performance_stats(backtest.equity_curve)
    if print_performance_stats:
        _print_performance_stats(stats)

    if save_stats_json:
        path = Path(str(save_stats_json))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"Saved performance stats JSON: {path}")

    if verbose:
        print(f"Backtest points: {len(backtest.equity_curve)}")
        if len(backtest.equity_curve) > 1:
            ret = backtest.equity_curve.iloc[-1] / backtest.equity_curve.iloc[0] - ONE
            print(f"Total return: {ret * PERCENT_SCALE:.2f}%")
        if not backtest.rebalance_log.empty:
            avg_scale = float(backtest.rebalance_log["buy_scale"].mean())
            print(f"Average buy_scale: {avg_scale:.4f}")

    if print_rebalance_log and not backtest.rebalance_log.empty:
        print("\nRebalance log:")
        print(backtest.rebalance_log.tail(rebalance_tail).to_string(index=False))

    if save_equity_csv:
        path = Path(str(save_equity_csv))
        path.parent.mkdir(parents=True, exist_ok=True)
        backtest.equity_curve.rename("equity").to_csv(path, header=True)
        print(f"Saved equity curve CSV: {path}")

    if save_account_history_csv:
        path = Path(str(save_account_history_csv))
        path.parent.mkdir(parents=True, exist_ok=True)
        backtest.account_history.to_csv(path, index=True)
        print(f"Saved account history CSV: {path}")

    if save_position_values_csv and position_value_cols:
        path = Path(str(save_position_values_csv))
        path.parent.mkdir(parents=True, exist_ok=True)
        backtest.account_history[position_value_cols].to_csv(path, index=True)
        print(f"Saved position values CSV: {path}")

    if save_full_hold_benchmarks_csv and not full_hold_benchmarks.empty:
        path = Path(str(save_full_hold_benchmarks_csv))
        path.parent.mkdir(parents=True, exist_ok=True)
        full_hold_benchmarks.to_csv(path, index=True)
        print(f"Saved full-hold benchmark CSV: {path}")

    if save_rebalance_csv and not backtest.rebalance_log.empty:
        path = Path(str(save_rebalance_csv))
        path.parent.mkdir(parents=True, exist_ok=True)
        backtest.rebalance_log.to_csv(path, index=False)
        print(f"Saved rebalance log CSV: {path}")

    if plot_equity_curve:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        account = backtest.account_history
        ax.plot(account.index, account["equity"], label="Equity", linewidth=2.0)
        if "cash" in account.columns:
            ax.plot(account.index, account["cash"], label="Cash", linewidth=1.8)
        for col in position_value_cols:
            ax.plot(account.index, account[col], label=col.replace("_value", ""), alpha=0.9)
        if plot_full_hold_benchmarks and not full_hold_benchmarks.empty:
            for col in full_hold_benchmarks.columns:
                label = col.replace("_full_hold", "") + " (full hold)"
                ax.plot(
                    full_hold_benchmarks.index,
                    full_hold_benchmarks[col],
                    label=label,
                    linestyle="--",
                    alpha=0.9,
                )
        if plot_trade_markers and not backtest.rebalance_log.empty:
            rebalance = backtest.rebalance_log.copy()
            rebalance["date"] = pd.to_datetime(rebalance["date"], errors="coerce")
            rebalance = rebalance.dropna(subset=["date"])
            if not rebalance.empty:
                equity_series = account["equity"]
                y_span = float(equity_series.max() - equity_series.min())
                y_offset = max(y_span * TRADE_MARKER_OFFSET_RATIO, ONE)
                buy_days = pd.DatetimeIndex(rebalance.loc[rebalance["buy_notional"] > 0, "date"].unique())
                sell_days = pd.DatetimeIndex(rebalance.loc[rebalance["sell_notional"] > 0, "date"].unique())
                buy_points = equity_series.reindex(buy_days).dropna()
                sell_points = equity_series.reindex(sell_days).dropna()
                if not buy_points.empty:
                    ax.scatter(
                        buy_points.index,
                        buy_points.values + y_offset,
                        marker="^",
                        color="tab:green",
                        s=TRADE_MARKER_SIZE,
                        alpha=TRADE_MARKER_ALPHA,
                        edgecolors="black",
                        linewidths=TRADE_MARKER_EDGE_WIDTH,
                        zorder=6,
                        label="Buy",
                    )
                if not sell_points.empty:
                    ax.scatter(
                        sell_points.index,
                        sell_points.values - y_offset,
                        marker="v",
                        color="tab:red",
                        s=TRADE_MARKER_SIZE,
                        alpha=TRADE_MARKER_ALPHA,
                        edgecolors="black",
                        linewidths=TRADE_MARKER_EDGE_WIDTH,
                        zorder=6,
                        label="Sell",
                    )
        ax.set_title("Backtest Equity, Cash, and Position Values")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", ncol=2)
        fig.tight_layout()

        if equity_plot_path:
            path = Path(str(equity_plot_path))
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=PLOT_DPI)
            print(f"Saved equity curve plot: {path}")
            plt.close(fig)
        else:
            plt.show()


def backtest_from_config(config_path: str) -> int:
    payload, cfg_path = _load_config(config_path)
    validate_config(payload, command="backtest")
    _require_keys(payload, [KEY_PORTFOLIO], scope="config")

    policy_cfg = _resolve_policy_config(payload, cfg_path)
    portfolio_cfg = payload[KEY_PORTFOLIO]
    _require_keys(portfolio_cfg, [KEY_CASH], scope="portfolio")
    initial_cash = float(portfolio_cfg[KEY_CASH])
    initial_positions = portfolio_cfg.get(KEY_POSITIONS, {})
    output_cfg = payload.get(KEY_OUTPUT, {})

    ohlcv_map, index_close = _load_market_data(payload, cfg_path)
    backtest = run_policy(
        ohlcv_map=ohlcv_map,
        index_close=index_close,
        config=policy_cfg,
        initial_cash=initial_cash,
        initial_positions=initial_positions,
    )

    print(f"Backtest final equity: {backtest.equity_curve.iloc[-1]:.2f}")
    print(f"Backtest final cash: {backtest.final_cash:.2f}")
    print("Final holdings (shares):")
    print(backtest.final_holdings.round(4).to_string())
    _handle_backtest_output(backtest, output_cfg, ohlcv_map)
    return 0


def recommend_from_config(config_path: str) -> int:
    payload, cfg_path = _load_config(config_path)
    validate_config(payload, command="recommend")
    _require_keys(payload, [KEY_PORTFOLIO], scope="config")

    policy_cfg = _resolve_policy_config(payload, cfg_path)
    portfolio_cfg = payload[KEY_PORTFOLIO]
    rec_cfg = payload.get(KEY_RECOMMENDATION, {})

    _require_keys(portfolio_cfg, [KEY_CASH], scope="portfolio")
    current_cash = float(portfolio_cfg[KEY_CASH])
    current_positions = portfolio_cfg.get(KEY_POSITIONS, {})
    min_trade_shares = float(rec_cfg.get("min_trade_shares", DEFAULT_MIN_TRADE_SHARES))

    ohlcv_map, index_close = _load_market_data(payload, cfg_path)
    recommendations, target_weights, risk_on = recommend_positions(
        ohlcv_map=ohlcv_map,
        index_close=index_close,
        current_positions=current_positions,
        current_cash=current_cash,
        config=policy_cfg,
        min_trade_shares=min_trade_shares,
    )

    regime_label = LABEL_RISK_ON if risk_on == RISK_ON_FLAG else LABEL_RISK_OFF
    print(f"Regime: {regime_label}")
    print("Target weights:")
    nonzero = target_weights[target_weights > 0].sort_values(ascending=False).round(4)
    print(nonzero.to_string() if len(nonzero) else NO_TARGET_EXPOSURE_TEXT)

    print("\nRecommendations:")
    rec_df = _format_recommendations(recommendations)
    print(rec_df.to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-assist", description="Trade Assist CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    backtest_parser = sub.add_parser("backtest", help="Run policy backtest")
    backtest_parser.add_argument("--config", required=True, help="Path to backtest JSON config file")

    recommend_parser = sub.add_parser("recommend", help="Generate buy/sell/hold recommendations")
    recommend_parser.add_argument("--config", required=True, help="Path to recommendation JSON config file")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "backtest":
        return backtest_from_config(args.config)
    if args.command == "recommend":
        return recommend_from_config(args.config)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
