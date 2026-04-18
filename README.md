# trade_assist

`trade_assist` is a config-driven Python toolkit for trading strategy research. It helps you backtest policies, generate portfolio actions from current holdings, and tune strategy parameters with walk-forward validation.

## Highlights

- JSON-configured workflows for backtesting, recommendations, and tuning
- Built-in market regime logic, risk caps, stop-losses, trailing stops, and liquidity guards
- Walk-forward tuning with cached trial logs, progress reporting, and final summaries
- CSV, JSON, and chart outputs for analysis
- Extensible policy adapter interface for adding new policy types

## Core Commands

### `trade-assist backtest`

Run a historical simulation of a policy against a portfolio and market dataset.

```bash
trade-assist backtest --config config.backtest.json
```

### `trade-assist recommend`

Compare current holdings to the model target and print `BUY`, `SELL`, or `HOLD` actions.

```bash
trade-assist recommend --config config.recommend.json
```

### `trade-assist tune`

Search a policy parameter space and rank trials with walk-forward validation.

```bash
trade-assist tune --config config.tune.json
```

## Quick Start

```bash
git clone <your-fork-or-repo-url>
cd trade_assist
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make dev-setup
```

For a minimal runtime-only install:

```bash
pip install -e .
```

## Typical Workflow

1. Copy one of the example configs:

```bash
cp config.backtest.example.json config.backtest.json
cp config.recommend.example.json config.recommend.json
cp config.tune.example.json config.tune.json
```

2. Edit your symbols, policy file, and portfolio state.
3. Run a backtest to confirm the setup works end-to-end.
4. Run recommendations or tuning once the baseline looks sensible.

Example:

```bash
trade-assist backtest --config config.backtest.json
trade-assist recommend --config config.recommend.json
trade-assist tune --config config.tune.json
```

## Example Configs

- `config.backtest.example.json`
- `config.recommend.example.json`
- `config.tune.example.json`
- `policies/`

These examples cover:

- market data selection and caching
- starting cash and positions
- inline or file-based policy configs
- output exports and plots
- walk-forward tuning settings and search spaces

## Outputs

Depending on the command and config, `trade_assist` can write:

- performance stats JSON
- equity/account/rebalance CSVs
- benchmark comparison CSVs
- ranked tuning trial tables
- best-policy JSON files
- equity curve plots

Generated outputs are typically written under `outputs/`.

## Documentation

- [QUICKSTART.md](QUICKSTART.md)
  Quick setup guide for testing your own tickers and positions
- [CONFIG.md](CONFIG.md)
  Full configuration reference for backtest, recommend, tune, and policy files

## Development

Common `make` commands:

```bash
make dev-setup
make format
make check-format
make lint
make typecheck
make test
make check
make build
make release
```

Notes:

- `make format` runs autoflake and Black
- `make check` runs formatting checks, flake8, mypy, and tests
- `make release` runs the full quality gate before building source and wheel artifacts

## Extending

Policies are resolved through the adapter system. The default policy type is `v1`, and new adapters can be added under `trade_assist/policy/adapters/` or exposed through the `trade_assist.policy_adapters` entry-point group.

## Disclaimer

This project is for research and experimentation. It does not provide investment advice.
