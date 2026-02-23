# Quickstart

This guide shows how to test your own stocks and your own positions.

## 1. Pick the stocks you care about

Open one of these files:

- `config.backtest.example.json`
- `config.recommend.example.json`

Edit `data.tickers` with your symbols, for example:

```json
"tickers": ["MSFT", "NVDA", "AAPL"]
```

## 2. Pick a policy file

Set `policy_path` in config, for example:

- `"policy_path": "policies/baseline.json"`
- `"policy_path": "policies/defensive.json"`
- `"policy_path": "policies/aggressive.json"`

These policy files now include:

- liquidity limits (trade size vs market volume)
- early-exit rules (stop-loss / trailing stop)

## 3. Set your portfolio state

Edit `portfolio`:

- `cash`: cash available
- `positions`: shares you currently hold

Example:

```json
"portfolio": {
  "cash": 50000,
  "positions": {
    "MSFT": 20,
    "NVDA": 15,
    "AAPL": 10
  }
}
```

If you want to test from a clean slate, set all positions to `0`.

## 4. Run a backtest

Use the backtest template and run:

```bash
cp config.backtest.example.json config.backtest.json
trade-assist backtest --config config.backtest.json
```

This tells you how the policy would have performed historically from your configured starting state.
It also prints risk/return metrics (CAGR, volatility, Sharpe, max drawdown, and more).

Optional diagnostics:

- In `config.backtest.json`, set `output.verbose` to `true` for extra run stats.
- Set `output.save_equity_curve_csv` to export the equity curve.
- Set `output.save_account_history_csv` to export daily `equity`, `cash`, and per-position values.
- Set `output.save_position_values_csv` to export only per-position value series.
- Set `output.plot_full_hold_benchmarks` to `true` to overlay full-hold normalized ticker trends on the same chart.
- Optionally set `output.full_hold_benchmark_tickers` to choose which tickers to include.
- Set `output.save_rebalance_log_csv` to export rebalance rows.
- Set `output.plot_equity_curve` to `true` and `output.equity_curve_plot_path` to save a combined chart (equity + cash + positions).

Data caching:

- Set `data.use_cache` to `true` to cache market data locally.
- Use `data.cache_ttl_hours` to control refresh window.
- Set `data.force_refresh` to `true` when you want a fresh pull immediately.

## 5. Run recommendations

Use the recommend template and run:

```bash
cp config.recommend.example.json config.recommend.json
trade-assist recommend --config config.recommend.json
```

This compares your current portfolio with the model target and prints `BUY`, `SELL`, or `HOLD` per ticker.

## 6. Optional: test with MSFT/NVDA sample

```bash
trade-assist recommend --config config.msft_nvda.example.json
```

## 7. Tune policy only after baseline works

Once commands run end-to-end, adjust `policy` values (rebalance frequency, max weight, risk caps) one change at a time.

For full parameter definitions, see `CONFIG.md`.
