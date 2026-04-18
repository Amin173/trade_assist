from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from .config_validation import (
    ConfigValidationError,
    PolicyValidationError,
    validate_config,
    validate_policy,
)
from .policy import (
    PolicyAdapter,
    compute_performance_stats,
    get_policy_adapter,
)
from .policy.constants import RISK_ON_FLAG
from .ta.constants import (
    COL_CLOSE,
    DEFAULT_CACHE_DIR,
    DEFAULT_CACHE_TTL_HOURS,
    DEFAULT_INDEX_TICKER,
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
)
from .ta.data import fetch_ohlcv_map
from .tuning import TuningInterrupted, tune_policy

KEY_DATA = "data"
KEY_TICKERS = "tickers"
KEY_PORTFOLIO = "portfolio"
KEY_CASH = "cash"
KEY_POSITIONS = "positions"
KEY_POLICY = "policy"
KEY_POLICY_TYPE = "policy_type"
KEY_POLICY_PATH = "policy_path"
KEY_OUTPUT = "output"
KEY_RECOMMENDATION = "recommendation"
KEY_SPLIT = "split"
KEY_TUNING = "tuning"
KEY_OBJECTIVE = "objective"

KEY_INDEX_TICKER = "index_ticker"
KEY_REGIME_TICKERS = "regime_tickers"
KEY_PERIOD = "period"
KEY_INTERVAL = "interval"
KEY_USE_CACHE = "use_cache"
KEY_CACHE_DIR = "cache_dir"
KEY_CACHE_TTL_HOURS = "cache_ttl_hours"
KEY_FORCE_REFRESH = "force_refresh"

DEFAULT_POLICY_TYPE = "v1"
DEFAULT_MIN_TRADE_SHARES = 1.0
DEFAULT_REBALANCE_LOG_TAIL = 20
INVESTED_EPSILON = 1e-9
PLOT_DPI = 140
PERCENT_SCALE = 100.0
TRADE_MARKER_SIZE = 90
TRADE_MARKER_ALPHA = 0.95
TRADE_MARKER_EDGE_WIDTH = 0.8
TRADE_MARKER_OFFSET_RATIO = 0.01
DEFAULT_TUNE_OUTPUT_DIR = "outputs/tuning"
DEFAULT_TUNE_PRINT_TOP_K = 10

LABEL_RISK_ON = "RISK_ON"
LABEL_RISK_OFF = "RISK_OFF"
NO_TARGET_EXPOSURE_TEXT = "No target exposure (all cash)"


def _load_config(path: str) -> tuple[dict[str, Any], Path]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f), cfg_path


def _require_keys(
    payload: dict[str, Any], required: list[str], scope: str = "config"
) -> None:
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
    cache_ttl_hours = (
        None if cache_ttl_hours_raw is None else float(cache_ttl_hours_raw)
    )
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
    regime_symbols.extend(
        [symbol for symbol in regime_tickers if symbol != index_ticker]
    )
    regime_map = fetch_ohlcv_map(
        regime_symbols,
        period=period,
        interval=interval,
        use_cache=use_cache,
        cache_dir=cache_root,
        cache_ttl_hours=cache_ttl_hours,
        force_refresh=force_refresh,
    )
    regime_close = pd.concat(
        {symbol: regime_map[symbol][COL_CLOSE] for symbol in regime_symbols}, axis=1
    )
    if regime_close.shape[1] == 1:
        return ohlcv_map, regime_close.iloc[:, 0]
    return ohlcv_map, regime_close


def _resolve_policy_adapter(payload: dict[str, Any]) -> PolicyAdapter:
    policy_type = str(payload.get(KEY_POLICY_TYPE, DEFAULT_POLICY_TYPE))
    return get_policy_adapter(policy_type)


def _resolve_policy_payload(
    payload: dict[str, Any],
    config_path: Path,
    adapter: PolicyAdapter,
) -> dict[str, Any]:
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
        validate_policy(policy_payload, policy_type=adapter.policy_type)
        return policy_payload

    if KEY_POLICY in payload:
        policy_payload = payload.get(KEY_POLICY, {})
        if not isinstance(policy_payload, dict):
            raise ValueError("config.policy must be a JSON object")
        validate_policy(policy_payload, policy_type=adapter.policy_type)
        return policy_payload

    return adapter.default_policy_payload()


def _resolve_policy_config(
    payload: dict[str, Any],
    config_path: Path,
    adapter: PolicyAdapter,
) -> Any:
    policy_payload = _resolve_policy_payload(payload, config_path, adapter)
    return adapter.build_policy_config(policy_payload)


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
    return compute_performance_stats(equity_curve)


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


def _handle_backtest_output(
    backtest, output_cfg: dict[str, Any], ohlcv_map: dict[str, pd.DataFrame]
) -> None:
    verbose = bool(output_cfg.get("verbose", False))
    print_performance_stats = bool(output_cfg.get("print_performance_stats", True))
    print_rebalance_log = bool(output_cfg.get("print_rebalance_log", False))
    rebalance_tail = int(
        output_cfg.get("rebalance_log_tail", DEFAULT_REBALANCE_LOG_TAIL)
    )
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
    position_value_cols = [
        col for col in backtest.account_history.columns if str(col).endswith("_value")
    ]
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
        start_equity = (
            float(account.loc[benchmark_start_date, "equity"]) if len(account) else 0.0
        )
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
            ret = backtest.equity_curve.iloc[-1] / backtest.equity_curve.iloc[0] - 1.0
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
            ax.plot(
                account.index, account[col], label=col.replace("_value", ""), alpha=0.9
            )
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
            if "trade_count" in rebalance.columns:
                rebalance = rebalance[rebalance["trade_count"] > 0]
            if not rebalance.empty:
                equity_series = account["equity"]
                y_span = float(equity_series.max() - equity_series.min())
                y_offset = max(y_span * TRADE_MARKER_OFFSET_RATIO, 1.0)
                buy_days = pd.DatetimeIndex(
                    rebalance.loc[rebalance["buy_notional"] > 0.0, "date"].unique()
                )
                sell_days = pd.DatetimeIndex(
                    rebalance.loc[rebalance["sell_notional"] > 0.0, "date"].unique()
                )
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

    adapter = _resolve_policy_adapter(payload)
    policy_cfg = _resolve_policy_config(payload, cfg_path, adapter)
    portfolio_cfg = payload[KEY_PORTFOLIO]
    _require_keys(portfolio_cfg, [KEY_CASH], scope="portfolio")
    initial_cash = float(portfolio_cfg[KEY_CASH])
    initial_positions = portfolio_cfg.get(KEY_POSITIONS, {})
    output_cfg = payload.get(KEY_OUTPUT, {})

    ohlcv_map, index_close = _load_market_data(payload, cfg_path)
    backtest = adapter.run_backtest(
        ohlcv_map=ohlcv_map,
        index_close=index_close,
        policy_config=policy_cfg,
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

    adapter = _resolve_policy_adapter(payload)
    policy_cfg = _resolve_policy_config(payload, cfg_path, adapter)
    portfolio_cfg = payload[KEY_PORTFOLIO]
    rec_cfg = payload.get(KEY_RECOMMENDATION, {})

    _require_keys(portfolio_cfg, [KEY_CASH], scope="portfolio")
    current_cash = float(portfolio_cfg[KEY_CASH])
    current_positions = portfolio_cfg.get(KEY_POSITIONS, {})
    min_trade_shares = float(rec_cfg.get("min_trade_shares", DEFAULT_MIN_TRADE_SHARES))

    ohlcv_map, index_close = _load_market_data(payload, cfg_path)
    recommendations, target_weights, risk_on = adapter.recommend(
        ohlcv_map=ohlcv_map,
        index_close=index_close,
        current_positions=current_positions,
        current_cash=current_cash,
        policy_config=policy_cfg,
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


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":")))
        f.write("\n")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_tuning_run_id() -> str:
    return datetime.now(timezone.utc).strftime("tune_%Y%m%dT%H%M%S_%fZ")


def _tuning_summary_metric_explanations() -> dict[str, str]:
    return {
        "status": "Whether the tuning run completed normally or was aborted/interrupted.",
        "trial_count": "Number of parameter combinations evaluated during tuning.",
        "target_trial_count": "Target number of trials requested in the config.",
        "window_count": "Number of walk-forward validation windows used per trial.",
        "best_trial_id": "Identifier of the highest-ranked trial after scoring and feasibility checks.",
        "best_score": "Composite tuning objective score. Higher is better; infeasible windows incur a large penalty.",
        "best_mean_cagr_pct": "Average annualized return across the validation windows for the best trial.",
        "best_mean_max_drawdown_pct": "Average worst peak-to-trough drawdown across validation windows for the best trial.",
        "best_mean_sharpe": "Average Sharpe ratio across validation windows for the best trial.",
        "best_mean_trade_count": "Average number of executed trades per validation window for the best trial.",
        "run_log_path": "Cache log with per-trial events for this tuning run.",
        "final_test_stats": "Optional out-of-sample metrics on the configured final test range.",
        "final_test_stats.cagr_pct": "Annualized return over the final test period.",
        "final_test_stats.max_drawdown_pct": "Worst peak-to-trough drawdown during the final test period.",
        "final_test_stats.annualized_vol_pct": "Annualized volatility during the final test period.",
        "final_test_stats.sharpe": "Sharpe ratio during the final test period.",
        "final_test_stats.sortino": "Sortino ratio during the final test period.",
        "final_test_stats.total_return_pct": "Total percentage return over the final test period.",
    }


def _build_tuning_summary_payload(
    status: str,
    policy_type: str,
    trial_count: int,
    target_trial_count: int,
    window_count: int,
    best_trial: Any | None,
    final_test_stats: dict[str, float] | None,
    run_log_path: str | None = None,
) -> dict[str, Any]:
    def _trial_value(trial: Any, key: str) -> Any:
        if isinstance(trial, dict):
            return trial.get(key)
        return getattr(trial, key)

    summary = {
        "status": status,
        "policy_type": policy_type,
        "trial_count": int(trial_count),
        "target_trial_count": int(target_trial_count),
        "window_count": int(window_count),
        "best_trial_id": None,
        "best_score": None,
        "best_mean_cagr_pct": None,
        "best_mean_max_drawdown_pct": None,
        "best_mean_sharpe": None,
        "best_mean_trade_count": None,
        "final_test_stats": final_test_stats,
        "run_log_path": run_log_path,
        "metric_explanations": _tuning_summary_metric_explanations(),
    }
    if best_trial is not None:
        summary.update(
            {
                "best_trial_id": int(_trial_value(best_trial, "trial_id")),
                "best_score": float(_trial_value(best_trial, "score")),
                "best_mean_cagr_pct": float(_trial_value(best_trial, "mean_cagr_pct")),
                "best_mean_max_drawdown_pct": float(
                    _trial_value(best_trial, "mean_max_drawdown_pct")
                ),
                "best_mean_sharpe": float(_trial_value(best_trial, "mean_sharpe")),
                "best_mean_trade_count": float(
                    _trial_value(best_trial, "mean_trade_count")
                ),
            }
        )
    return summary


def _resolve_tune_output_dir(output_cfg: dict[str, Any], cfg_path: Path) -> Path:
    output_dir = Path(str(output_cfg.get("dir", DEFAULT_TUNE_OUTPUT_DIR)))
    if not output_dir.is_absolute():
        output_dir = (cfg_path.parent / output_dir).resolve()
    return output_dir


def _serialize_trial_for_cache(
    completed_count: int,
    total_trials: int,
    elapsed_seconds: float,
    trial: Any,
) -> dict[str, Any]:
    return {
        "event": "trial_completed",
        "logged_at": _utc_now_iso(),
        "completed_count": int(completed_count),
        "total_trials": int(total_trials),
        "elapsed_seconds": float(elapsed_seconds),
        "trial": {
            **trial.to_row(),
            "failed_windows": list(trial.failed_windows),
            "failed_criteria": list(trial.failed_criteria),
            "policy_payload": trial.policy_payload,
        },
    }


def _read_tuning_run_log(path: Path) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))
    start_event = next(
        (event for event in events if event.get("event") == "run_context"), None
    )
    finish_event = next(
        (event for event in reversed(events) if event.get("event") == "finish"),
        None,
    )
    trial_events = [
        event for event in events if event.get("event") == "trial_completed"
    ]
    return {
        "events": events,
        "start": start_event,
        "finish": finish_event,
        "trial_events": trial_events,
    }


def _trials_dataframe_from_cache_log(log_data: dict[str, Any]) -> pd.DataFrame:
    rows = [event["trial"] for event in log_data["trial_events"]]
    if not rows:
        return pd.DataFrame(
            columns=[
                "trial_id",
                "score",
                "feasible",
                "window_count",
                "failed_window_count",
                "failed_windows",
                "failed_criteria",
                "mean_cagr_pct",
                "mean_max_drawdown_pct",
                "mean_annualized_vol_pct",
                "mean_sharpe",
                "mean_sortino",
                "mean_trade_count",
                "overrides",
                "policy_payload",
            ]
        )
    return pd.DataFrame(rows).sort_values(["score", "trial_id"], ascending=[False, True])


def _best_trial_record_from_cache(log_data: dict[str, Any]) -> dict[str, Any] | None:
    trials_df = _trials_dataframe_from_cache_log(log_data)
    if trials_df.empty:
        return None
    best_trial_id = int(trials_df.iloc[0]["trial_id"])
    for event in log_data["trial_events"]:
        trial = event["trial"]
        if int(trial["trial_id"]) == best_trial_id:
            return trial
    return None


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_clock_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_failed_criteria_preview(
    failed_criteria: tuple[str, ...] | list[str], max_items: int = 2
) -> str:
    items = list(failed_criteria)
    if not items:
        return "-"
    if len(items) <= max_items:
        return ", ".join(items)
    shown = ", ".join(items[:max_items])
    remaining = len(items) - max_items
    return f"{shown}, +{remaining} more"


def _format_progress_failed_criteria_preview(
    failed_criteria: tuple[str, ...] | list[str], max_items: int = 1
) -> str:
    preview = _format_failed_criteria_preview(failed_criteria, max_items=max_items)
    return preview.replace("(observed ", "(").replace("))", ")")


def _print_tuning_cache_summary(
    log_data: dict[str, Any], top_k: int, run_log_path: Path
) -> None:
    trials_df = _trials_dataframe_from_cache_log(log_data)
    start_event = log_data["start"] or {}
    finish_event = log_data["finish"] or {}
    target_trials = int(
        finish_event.get(
            "total_trials",
            start_event.get("target_trial_count", len(trials_df)),
        )
    )
    status = str(finish_event.get("status", "completed"))
    window_count = int(start_event.get("window_count", 0))

    print("\nRun summary:")
    print(f"- status: {status}")
    print(f"- completed_trials: {len(trials_df)}/{target_trials}")
    if window_count > 0:
        print(f"- validation_windows: {window_count}")
    print(f"- run_log: {run_log_path}")

    if trials_df.empty:
        print("No completed trials were logged before the run stopped.")
        return

    best = trials_df.iloc[0]
    print("Best logged trial:")
    print(
        f"- score: {float(best['score']):.4f} "
        "(composite objective; higher is better)"
    )
    print(
        f"- mean_cagr_pct: {float(best['mean_cagr_pct']):.2f}% "
        "(average annualized return across validation windows)"
    )
    print(
        f"- mean_max_drawdown_pct: {float(best['mean_max_drawdown_pct']):.2f}% "
        "(average worst peak-to-trough drawdown across validation windows)"
    )
    print(
        f"- mean_sharpe: {float(best['mean_sharpe']):.3f} "
        "(average Sharpe ratio across validation windows)"
    )
    print(
        f"- mean_trade_count: {float(best['mean_trade_count']):.2f} "
        "(average executed trades per validation window)"
    )
    print(
        f"- failed_window_count: {int(best['failed_window_count'])}/"
        f"{int(best['window_count'])}"
    )
    print(f"- failed_criteria: {best['failed_criteria'] or '-'}")

    display_cols = [
        "trial_id",
        "score",
        "mean_cagr_pct",
        "mean_max_drawdown_pct",
        "mean_sharpe",
        "mean_trade_count",
        "failed_window_count",
        "failed_criteria",
        "feasible",
    ]
    print("\nTop logged trials:")
    print(trials_df.head(top_k)[display_cols].to_string(index=False))


def _write_tuning_outputs_from_cache(
    log_data: dict[str, Any],
    output_dir: Path,
    output_cfg: dict[str, Any],
    policy_type: str,
    run_log_path: Path,
) -> None:
    trials_df = _trials_dataframe_from_cache_log(log_data)
    start_event = log_data["start"] or {}
    finish_event = log_data["finish"] or {}
    best_trial = _best_trial_record_from_cache(log_data)
    summary = _build_tuning_summary_payload(
        status=str(finish_event.get("status", "completed")),
        policy_type=policy_type,
        trial_count=len(trials_df),
        target_trial_count=int(
            finish_event.get(
                "total_trials", start_event.get("target_trial_count", len(trials_df))
            )
        ),
        window_count=int(start_event.get("window_count", 0)),
        best_trial=best_trial,
        final_test_stats=finish_event.get("final_test_stats"),
        run_log_path=str(run_log_path),
    )

    if bool(output_cfg.get("save_trials_csv", True)):
        output_dir.mkdir(parents=True, exist_ok=True)
        trials_path = output_dir / "tuning_trials.csv"
        trials_df.to_csv(trials_path, index=False)
        print(f"Saved tuning trials CSV: {trials_path}")

    if bool(output_cfg.get("save_best_policy_json", True)) and best_trial is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        best_policy_path = output_dir / "best_policy.json"
        _save_json(best_policy_path, best_trial["policy_payload"])
        print(f"Saved best policy JSON: {best_policy_path}")

    if bool(output_cfg.get("save_summary_json", True)):
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "tuning_summary.json"
        _save_json(summary_path, summary)
        print(f"Saved tuning summary JSON: {summary_path}")


def tune_from_config(config_path: str) -> int:
    payload, cfg_path = _load_config(config_path)
    validate_config(payload, command="tune")
    _require_keys(payload, [KEY_PORTFOLIO, KEY_TUNING], scope="config")

    adapter = _resolve_policy_adapter(payload)
    base_policy_payload = _resolve_policy_payload(payload, cfg_path, adapter)

    portfolio_cfg = payload[KEY_PORTFOLIO]
    _require_keys(portfolio_cfg, [KEY_CASH], scope="portfolio")
    initial_cash = float(portfolio_cfg[KEY_CASH])
    initial_positions = portfolio_cfg.get(KEY_POSITIONS, {})

    split_cfg = payload.get(KEY_SPLIT, {}) or {}
    tuning_cfg = payload[KEY_TUNING]
    objective_cfg = payload.get(KEY_OBJECTIVE, {}) or {}
    output_cfg = payload.get(KEY_OUTPUT, {}) or {}
    output_dir = _resolve_tune_output_dir(output_cfg, cfg_path)
    top_k = int(output_cfg.get("print_top_k", DEFAULT_TUNE_PRINT_TOP_K))
    run_log_path = (
        output_dir / ".cache" / "tuning_runs" / f"{_make_tuning_run_id()}.jsonl"
    )

    ohlcv_map, index_close = _load_market_data(payload, cfg_path)
    from .tuning.engine import _build_walk_forward_windows, _iter_overrides

    window_count = len(
        _build_walk_forward_windows(
            next(iter(ohlcv_map.values())).index,
            split_cfg,
        )
    )
    target_trial_count = len(_iter_overrides(tuning_cfg.get("search_space", {}), tuning_cfg))
    _append_jsonl(
        run_log_path,
        {
            "event": "run_context",
            "logged_at": _utc_now_iso(),
            "config_path": str(cfg_path.resolve()),
            "policy_type": adapter.policy_type,
            "target_trial_count": int(target_trial_count),
            "window_count": int(window_count),
            "workers": tuning_cfg.get("workers", 1),
        },
    )

    print(
        "Running tuning. Detailed trial logs are being written to cache.",
        flush=True,
    )
    tune_start = time.perf_counter()

    def _report_trial_progress(
        completed_count: int, total_trials: int, trial: Any
    ) -> None:
        total_width = len(str(total_trials))
        elapsed = time.perf_counter() - tune_start
        avg_seconds = elapsed / completed_count if completed_count > 0 else 0.0
        remaining_trials = max(total_trials - completed_count, 0)
        eta_seconds = avg_seconds * remaining_trials
        eta_text = (
            _format_clock_duration(eta_seconds)
            if math.isfinite(eta_seconds)
            else "unknown"
        )
        _append_jsonl(
            run_log_path,
            _serialize_trial_for_cache(
                completed_count=completed_count,
                total_trials=total_trials,
                elapsed_seconds=elapsed,
                trial=trial,
            ),
        )
        print(
            (
                f"{completed_count:>{total_width}}/{total_trials} "
                f"t{trial.trial_id:>{total_width}} "
                f"s={trial.score:>10.2f} "
                f"cagr={trial.mean_cagr_pct:>6.2f}% "
                f"mdd={trial.mean_max_drawdown_pct:>6.2f}% "
                f"tr={trial.mean_trade_count:>5.1f} "
                f"fail={trial.failed_window_count}/{trial.window_count} "
                f"crit={_format_progress_failed_criteria_preview(trial.failed_criteria)} "
                f"time={_format_clock_duration(elapsed)}/{eta_text}"
            ),
            flush=True,
        )

    status = "completed"
    final_test_stats: dict[str, float] | None = None
    return_code = 0
    try:
        result = tune_policy(
            adapter=adapter,
            ohlcv_map=ohlcv_map,
            index_close=index_close,
            initial_cash=initial_cash,
            initial_positions=initial_positions,
            base_policy_payload=base_policy_payload,
            split_cfg=split_cfg,
            tuning_cfg=tuning_cfg,
            objective_cfg=objective_cfg,
            progress_callback=_report_trial_progress,
        )
        final_test_stats = result.final_test_stats
    except TuningInterrupted:
        status = "aborted"
        return_code = 130
        print("\nTuning interrupted by user.", flush=True)
    except KeyboardInterrupt:
        status = "aborted"
        return_code = 130
        print("\nTuning interrupted by user.", flush=True)
    finally:
        elapsed = time.perf_counter() - tune_start
        log_data = _read_tuning_run_log(run_log_path)
        completed_trials = len(log_data["trial_events"])
        total_trials = (
            int(log_data["trial_events"][-1]["total_trials"])
            if log_data["trial_events"]
            else int(target_trial_count)
        )
        _append_jsonl(
            run_log_path,
            {
                "event": "finish",
                "logged_at": _utc_now_iso(),
                "status": status,
                "elapsed_seconds": float(elapsed),
                "completed_trials": int(completed_trials),
                "total_trials": int(total_trials),
                "final_test_stats": final_test_stats,
            },
        )
        log_data = _read_tuning_run_log(run_log_path)
        _print_tuning_cache_summary(
            log_data=log_data,
            top_k=top_k,
            run_log_path=run_log_path,
        )
        _write_tuning_outputs_from_cache(
            log_data=log_data,
            output_dir=output_dir,
            output_cfg=output_cfg,
            policy_type=adapter.policy_type,
            run_log_path=run_log_path,
        )
    return return_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trade-assist", description="Trade Assist CLI"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    backtest_parser = sub.add_parser("backtest", help="Run policy backtest")
    backtest_parser.add_argument(
        "--config", required=True, help="Path to backtest JSON config file"
    )

    recommend_parser = sub.add_parser(
        "recommend", help="Generate buy/sell/hold recommendations"
    )
    recommend_parser.add_argument(
        "--config", required=True, help="Path to recommendation JSON config file"
    )

    tune_parser = sub.add_parser(
        "tune", help="Tune policy parameters over historical data"
    )
    tune_parser.add_argument(
        "--config", required=True, help="Path to tuning JSON config file"
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "backtest":
            return backtest_from_config(args.config)
        if args.command == "recommend":
            return recommend_from_config(args.config)
        if args.command == "tune":
            return tune_from_config(args.config)
    except ConfigValidationError as exc:
        print(
            f"Error: {exc} [file: {Path(args.config).resolve()}]",
            file=sys.stderr,
        )
        return 2
    except PolicyValidationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
