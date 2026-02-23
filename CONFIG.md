# Configuration Reference

Schemas:

- Backtest config: `trade_assist/schemas/backtest.schema.json`
- Recommend config: `trade_assist/schemas/recommend.schema.json`
- Policy file: `trade_assist/schemas/policy.schema.json`

## `trade-assist backtest`

Command:

```bash
trade-assist backtest --config <file>
```

Runs a historical simulation with your selected settings.

### Required top-level fields

- `data`
- `portfolio`

### Optional top-level fields

- `$schema`: schema hint for editors and tooling.
- `policy`: inline policy object.
- `policy_path`: path to an external policy JSON file.
- `output`: optional diagnostics and exports.

Policy selection order:

1. `policy_path`
2. `policy`
3. built-in defaults

If `policy_path` is relative, it is resolved relative to the config file.

### `data`

- `tickers` (`array[string]`, required): symbols the strategy can trade.
- `index_ticker` (`string`, default `SPY`): benchmark used for market regime detection.
- `period` (`string`, default `5y`): historical range to download.
- `interval` (`string`, default `1d`): bar interval.
- `use_cache` (`boolean`, default `false`): cache downloaded market data locally.
- `cache_dir` (`string`, default `.cache/market_data`): local folder for cache files. Relative paths are resolved from the config file location.
- `cache_ttl_hours` (`number | null`, default `24`): cache freshness window in hours. Use `null` for no expiry.
- `force_refresh` (`boolean`, default `false`): bypass cache and fetch fresh data now.
- If a live download fails and a cache file exists, the cached data is used as fallback.

Example:

```json
"data": {
  "tickers": ["MSFT", "NVDA"],
  "index_ticker": "SPY",
  "period": "5y",
  "interval": "1d",
  "use_cache": true,
  "cache_dir": ".cache/market_data",
  "cache_ttl_hours": 24,
  "force_refresh": false
}
```

### `portfolio`

- `cash` (`number`, required): starting cash for backtests.
- `positions` (`object<string, number>`, optional): shares by symbol.

Note: any symbol in `positions` that is not in `data.tickers` is ignored.

Example:

```json
"portfolio": {
  "cash": 100000,
  "positions": {
    "MSFT": 0,
    "NVDA": 0
  }
}
```

### `policy`

You can define policy settings inline (`policy`) or in an external file (`policy_path`).

This section controls three things:

- how often positions can change,
- how much capital the model is allowed to deploy,
- how symbols are ranked against each other.

Core controls:

- `rebalance_freq` (`string`, default `W-FRI`): schedule for recalculating target positions. Higher frequency reacts faster but increases turnover.
- `target_vol_risk_on` (`number`, default `0.22`): target portfolio volatility when the market regime is favorable. Higher values generally increase portfolio swings.
- `gross_cap_risk_on` (`number`, default `1.0`): maximum total invested exposure in favorable conditions. `1.0` means up to 100% invested.
- `gross_cap_risk_off` (`number`, default `0.3`): maximum total invested exposure in defensive conditions. Lower values keep more capital in cash.
- `max_weight` (`number`, default `0.45`): upper bound for a single symbol's portfolio weight. Lower values reduce single-name concentration.
- `min_hold_days` (`integer`, default `10`): minimum number of days before the model can resize a position. Higher values reduce churn.
- `risk_off_target_vol_multiplier` (`number`, default `0.5`): scales risk target in defensive conditions. Example: `0.22 * 0.5 = 0.11`.
- `covariance_lookback` (`integer`, default `252`): number of historical trading days used to estimate correlation/risk structure. Larger windows are steadier but slower to adapt.

`liquidity`:

- `min_adv_dollars` (`number`, default `0`): minimum daily dollar volume required for new buys. If a symbol trades less than this level, the model does not add to it.
- `max_trade_adv_fraction` (`number`, default `0.10`): maximum size for any single trade as a fraction of one day of dollar volume. Example: `0.10` means each buy/sell is capped at 10% of that symbol's daily dollar volume.

`risk_exit`:

- `stop_loss_pct` (`number`, default `0`): hard loss limit from entry price. Example: `0.12` exits when price falls 12% below entry.
- `trailing_stop_pct` (`number`, default `0`): moving loss limit from the highest price reached while in position. Example: `0.15` exits after a 15% pullback from that local high.
- `cooldown_days` (`integer`, default `0`): days to wait after a forced risk exit before opening a new position in that symbol.

`score_weights`:

- `c_over_ema200` (`number`, default `0.35`): emphasizes symbols trading above long-term trend.
- `ema50_over_ema200` (`number`, default `0.25`): emphasizes symbols with stronger medium-vs-long trend alignment.
- `mom_252_21` (`number`, default `0.25`): emphasizes medium-term momentum persistence.
- `vol_adjusted_momentum` (`number`, default `0.15`): emphasizes momentum after adjusting for volatility, to reduce preference for noisy moves.
- `breakout55` (`number`, default `0.10`): adds preference for symbols near or above recent breakout levels.

For `score_weights`, higher values increase that signal's influence in ranking. Values are relative; they do not need to sum to 1.

### Regime logic

The benchmark (`index_ticker`) is `risk_on = 1` only when all are true:

- `Close > EMA200`
- `EMA50.diff(10) > 0`
- `realized_volatility_20 < 0.35`

Otherwise, `risk_on = 0`.

Regime impact:

- Risk-on uses `target_vol_risk_on` and `gross_cap_risk_on`.
- Risk-off uses `target_vol_risk_on * risk_off_target_vol_multiplier` and `gross_cap_risk_off`.

### `output` (optional, backtest only)

- `verbose` (`boolean`, default `false`): print additional run details.
- `print_performance_stats` (`boolean`, default `true`): print risk/return metrics.
- `save_performance_stats_json` (`string`, optional): save metrics JSON.
- `print_rebalance_log` (`boolean`, default `false`): print rebalance table.
- `rebalance_log_tail` (`integer`, default `20`): rows to print from rebalance log.
- `save_equity_curve_csv` (`string`, optional): save equity curve CSV.
- `save_account_history_csv` (`string`, optional): save daily account history (`equity`, `cash`, and per-position values) as CSV.
- `save_position_values_csv` (`string`, optional): save per-position value series as CSV.
- `save_full_hold_benchmarks_csv` (`string`, optional): save full-hold benchmark series as CSV.
- `save_rebalance_log_csv` (`string`, optional): save rebalance log CSV.
- `plot_equity_curve` (`boolean`, default `false`): generate a chart with equity, cash, and per-position value lines.
- `plot_full_hold_benchmarks` (`boolean`, default `false`): overlay normalized full-hold benchmark lines (all starting equity in one ticker, held throughout).
- `full_hold_benchmark_tickers` (`array[string]`, optional): optional subset of tickers for full-hold overlays; defaults to all tickers.
- `equity_curve_plot_path` (`string`, optional): file path for saved chart.

Performance metrics include:

- total return
- CAGR
- annualized volatility
- Sharpe ratio
- Sortino ratio
- max drawdown
- Calmar ratio
- win rate
- best day / worst day

Cash rule:

- Backtest uses a no-leverage baseline. If buys exceed cash, buy orders are scaled down.
- Full-hold benchmark lines start from the strategy's first invested date (not necessarily the first date in the dataset).
- Rebalance log includes event details (`rebalance` vs `risk_exit`), forced-exit symbols, and liquidity-cap counters.

## `trade-assist recommend`

Command:

```bash
trade-assist recommend --config <file>
```

Compares current holdings to model target holdings and returns `BUY`, `SELL`, or `HOLD`.

### Required top-level fields

- `data`
- `portfolio`

### Optional top-level fields

- `$schema`
- `policy`
- `policy_path`
- `recommendation`

### Shared sections

- `data`: see backtest `data`.
- `portfolio`: see backtest `portfolio` (same schema, interpreted as current account state).
- `policy`: see backtest `policy` and `Regime logic`.

Recommendation-specific behavior:

- Liquidity controls from `policy.liquidity` are applied before final actions are printed, so recommendation sizes are capped to market depth.
- `risk_exit` rules are enforced in `backtest` only, because recommendation mode does not track per-position entry/high-water state.

### `recommendation`

- `min_trade_shares` (`number`, default `1.0`): minimum share difference required to trigger `BUY` or `SELL`.

Action rules:

- `BUY` if `target_shares - current_shares > min_trade_shares`
- `SELL` if `current_shares - target_shares > min_trade_shares`
- `HOLD` otherwise
