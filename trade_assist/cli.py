from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config_validation import validate_config, validate_policy
from .policy import PolicyConfig, run_policy
from .policy.recommendations import recommend_positions
from .ta.data import fetch_ohlcv_map


def _load_config(path: str) -> tuple[dict[str, Any], Path]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f), cfg_path


def _require_keys(payload: dict[str, Any], required: list[str], scope: str = "config") -> None:
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing required {scope} key(s): {', '.join(missing)}")


def _extract_data_cfg(payload: dict[str, Any]) -> tuple[list[str], str, str, str]:
    _require_keys(payload, ["data"], scope="config")
    data_cfg = payload["data"]
    _require_keys(data_cfg, ["tickers"], scope="data")

    tickers = data_cfg["tickers"]
    if not isinstance(tickers, list) or not tickers:
        raise ValueError("data.tickers must be a non-empty list of symbols")

    index_ticker = data_cfg.get("index_ticker", "SPY")
    period = data_cfg.get("period", "5y")
    interval = data_cfg.get("interval", "1d")
    return tickers, index_ticker, period, interval


def _load_market_data(payload: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    tickers, index_ticker, period, interval = _extract_data_cfg(payload)
    ohlcv_map = fetch_ohlcv_map(tickers=tickers, period=period, interval=interval)
    index_df = fetch_ohlcv_map([index_ticker], period=period, interval=interval)[index_ticker]
    return ohlcv_map, index_df["Close"]


def _resolve_policy_config(payload: dict[str, Any], config_path: Path) -> PolicyConfig:
    policy_path = payload.get("policy_path")
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

    if "policy" in payload:
        policy_payload = payload.get("policy", {})
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
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "annualized_vol_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown_pct": 0.0,
            "calmar": 0.0,
            "win_rate_pct": 0.0,
            "best_day_pct": 0.0,
            "worst_day_pct": 0.0,
        }

    returns = eq.pct_change().dropna()
    if len(returns) == 0:
        returns = pd.Series([0.0], index=eq.index[:1])

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    duration_days = max((eq.index[-1] - eq.index[0]).days, 1)
    years = duration_days / 365.25
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if eq.iloc[0] > 0 else 0.0

    annualized_vol = float(returns.std() * np.sqrt(252))
    annualized_mean = float(returns.mean() * 252)
    sharpe = float(annualized_mean / annualized_vol) if annualized_vol > 0 else 0.0

    downside = returns[returns < 0]
    downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else 0.0
    sortino = float(annualized_mean / downside_vol) if downside_vol > 0 else 0.0

    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    max_drawdown = float(drawdown.min())
    calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 else 0.0

    win_rate = float((returns > 0).mean())
    best_day = float(returns.max())
    worst_day = float(returns.min())

    return {
        "total_return_pct": total_return * 100.0,
        "cagr_pct": cagr * 100.0,
        "annualized_vol_pct": annualized_vol * 100.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_drawdown * 100.0,
        "calmar": calmar,
        "win_rate_pct": win_rate * 100.0,
        "best_day_pct": best_day * 100.0,
        "worst_day_pct": worst_day * 100.0,
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


def _handle_backtest_output(backtest, output_cfg: dict[str, Any]) -> None:
    verbose = bool(output_cfg.get("verbose", False))
    print_performance_stats = bool(output_cfg.get("print_performance_stats", True))
    print_rebalance_log = bool(output_cfg.get("print_rebalance_log", False))
    rebalance_tail = int(output_cfg.get("rebalance_log_tail", 20))
    save_equity_csv = output_cfg.get("save_equity_curve_csv")
    save_rebalance_csv = output_cfg.get("save_rebalance_log_csv")
    save_stats_json = output_cfg.get("save_performance_stats_json")
    plot_equity_curve = bool(output_cfg.get("plot_equity_curve", False))
    equity_plot_path = output_cfg.get("equity_curve_plot_path")

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
            print(f"Total return: {ret * 100:.2f}%")
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

    if save_rebalance_csv and not backtest.rebalance_log.empty:
        path = Path(str(save_rebalance_csv))
        path.parent.mkdir(parents=True, exist_ok=True)
        backtest.rebalance_log.to_csv(path, index=False)
        print(f"Saved rebalance log CSV: {path}")

    if plot_equity_curve:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(backtest.equity_curve.index, backtest.equity_curve.values, label="Equity")
        ax.set_title("Backtest Equity Curve")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()

        if equity_plot_path:
            path = Path(str(equity_plot_path))
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=140)
            print(f"Saved equity curve plot: {path}")
            plt.close(fig)
        else:
            plt.show()


def backtest_from_config(config_path: str) -> int:
    payload, cfg_path = _load_config(config_path)
    validate_config(payload, command="backtest")
    _require_keys(payload, ["portfolio"], scope="config")

    policy_cfg = _resolve_policy_config(payload, cfg_path)
    portfolio_cfg = payload["portfolio"]
    _require_keys(portfolio_cfg, ["cash"], scope="portfolio")
    initial_cash = float(portfolio_cfg["cash"])
    initial_positions = portfolio_cfg.get("positions", {})
    output_cfg = payload.get("output", {})

    ohlcv_map, index_close = _load_market_data(payload)
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
    _handle_backtest_output(backtest, output_cfg)
    return 0


def recommend_from_config(config_path: str) -> int:
    payload, cfg_path = _load_config(config_path)
    validate_config(payload, command="recommend")
    _require_keys(payload, ["portfolio"], scope="config")

    policy_cfg = _resolve_policy_config(payload, cfg_path)
    portfolio_cfg = payload["portfolio"]
    rec_cfg = payload.get("recommendation", {})

    _require_keys(portfolio_cfg, ["cash"], scope="portfolio")
    current_cash = float(portfolio_cfg["cash"])
    current_positions = portfolio_cfg.get("positions", {})
    min_trade_shares = float(rec_cfg.get("min_trade_shares", 1.0))

    ohlcv_map, index_close = _load_market_data(payload)
    recommendations, target_weights, risk_on = recommend_positions(
        ohlcv_map=ohlcv_map,
        index_close=index_close,
        current_positions=current_positions,
        current_cash=current_cash,
        config=policy_cfg,
        min_trade_shares=min_trade_shares,
    )

    regime_label = "RISK_ON" if risk_on == 1 else "RISK_OFF"
    print(f"Regime: {regime_label}")
    print("Target weights:")
    nonzero = target_weights[target_weights > 0].sort_values(ascending=False).round(4)
    print(nonzero.to_string() if len(nonzero) else "No target exposure (all cash)")

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
