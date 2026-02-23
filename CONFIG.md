# Configuration Reference

This document defines all supported config fields for `trade-assist`.

Schemas:

- Backtest: `trade_assist/schemas/backtest.schema.json`
- Recommend: `trade_assist/schemas/recommend.schema.json`

## Backtest Configuration

Command:

```bash
trade-assist backtest --config <file>
```

What this command does:

- Simulate how your selected strategy would have performed in the past.

### Required top-level fields

- `data`
- `portfolio`

### Optional top-level fields

- `$schema`: schema path for editor validation.
- `policy`: inline strategy settings.
- `policy_path`: path to a policy JSON file. Relative paths are resolved from the config file location.
- `output`: optional diagnostics/logging/plot settings.

Note:

- If you use `policy_path`, the file must exist and contain a valid policy object.

### `data`

- `tickers` (`array[string]`, required): symbols to trade.
- `index_ticker` (`string`, default `SPY`): benchmark symbol used to detect market regime.
- `period` (`string`, default `5y`): historical range to fetch.
- `interval` (`string`, default `1d`): bar size.

Example:

```json
"data": {
  "tickers": ["MSFT", "NVDA"],
  "index_ticker": "SPY",
  "period": "5y",
  "interval": "1d"
}
```

### `portfolio`

- `cash` (`number`, required): starting cash for the simulation.
- `positions` (`object<string, number>`, optional): starting shares by symbol.

Important:

- Positions whose symbols are not in `data.tickers` are ignored.

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

### `policy` options

You can define strategy settings in either form:

- external file via `policy_path`
- inline `policy`

Resolution order:

1. `policy_path`
2. `policy`
3. internal defaults

Example:

- `"policy_path": "policies/baseline.json"`

Policy fields:

- `rebalance_freq` (`string`, default `W-FRI`): rebalance schedule (`W-FRI` = weekly on Friday).
- `target_vol_risk_on` (`number`, default `0.22`)
- `gross_cap_risk_on` (`number`, default `1.0`)
- `gross_cap_risk_off` (`number`, default `0.3`)
- `max_weight` (`number`, default `0.45`)
- `min_hold_days` (`integer`, default `10`)
- `risk_off_target_vol_multiplier` (`number`, default `0.5`)
- `covariance_lookback` (`integer`, default `252`)

`score_weights`:

- `c_over_ema200` (`number`, default `0.35`)
- `ema50_over_ema200` (`number`, default `0.25`)
- `mom_252_21` (`number`, default `0.25`)
- `vol_adjusted_momentum` (`number`, default `0.15`)
- `breakout55` (`number`, default `0.10`)

### Regime logic used by backtest

The model marks the benchmark as `risk_on = 1` only when all are true:

- `Close > EMA200`
- `EMA50.diff(10) > 0`
- `realized_volatility_20 < 0.35`

Otherwise, `risk_on = 0`.

Regime impact:

- Risk-on uses `target_vol_risk_on` and `gross_cap_risk_on`.
- Risk-off uses `target_vol_risk_on * risk_off_target_vol_multiplier` and `gross_cap_risk_off`.

### `output` (optional, backtest only)

- `verbose` (`boolean`, default `false`): print extra run summary.
- `print_performance_stats` (`boolean`, default `true`): print risk/return stats.
- `save_performance_stats_json` (`string`, optional): file path for performance stats JSON.
- `print_rebalance_log` (`boolean`, default `false`): print rebalance log table.
- `rebalance_log_tail` (`integer`, default `20`): number of rebalance rows to print.
- `save_equity_curve_csv` (`string`, optional): file path for equity curve CSV.
- `save_rebalance_log_csv` (`string`, optional): file path for rebalance log CSV.
- `plot_equity_curve` (`boolean`, default `false`): generate equity curve plot.
- `equity_curve_plot_path` (`string`, optional): if set, save plot to this path; otherwise show interactive plot.

Performance stats include:

- total return
- CAGR
- annualized volatility
- Sharpe ratio
- Sortino ratio
- max drawdown
- Calmar ratio
- win rate
- best day / worst day

Cash handling note:

- Backtest enforces no-leverage cash handling; buy orders are scaled down when needed so ending cash does not go below zero.

## Recommend Configuration

Command:

```bash
trade-assist recommend --config <file>
```

What this command does:

- Compare your current holdings to the model target and return `BUY` / `SELL` / `HOLD` suggestions.

### Required top-level fields

- `data`
- `portfolio`

### Optional top-level fields

- `$schema`
- `policy`
- `policy_path`
- `recommendation`

### `data`

(See Backtest `data`.) Same structure and behavior.

### `portfolio`

(See Backtest `portfolio`.) Same structure; here it represents your current account state.

### `policy` / regime behavior

(See Backtest `policy options` and `Regime logic used by backtest`.) Same behavior.

### `recommendation`

- `min_trade_shares` (`number`, default `1.0`): minimum share difference required to trigger `BUY` or `SELL`.

### Action mapping

Per symbol:

- `BUY` if `target_shares - current_shares > min_trade_shares`
- `SELL` if `current_shares - target_shares > min_trade_shares`
- `HOLD` otherwise
