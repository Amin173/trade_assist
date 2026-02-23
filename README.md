# trade_assist

`trade_assist` provides two CLI tools:

- `backtest`: run historical simulation of a policy.
- `recommend`: generate `BUY` / `SELL` / `HOLD` actions from current portfolio state.

## Installation

```bash
cd /path/to/project_root
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

This installs the package and all runtime dependencies from `setup.py`.

## Subcommands

### `trade-assist backtest`

Purpose: evaluate how a policy would have behaved historically.

```bash
trade-assist backtest --config config.backtest.json
```

Template: `config.backtest.example.json`

Backtest supports optional diagnostics (verbose logs, CSV exports for equity/cash/positions, plots, and optional full-hold benchmark overlays) via the `output` section in config.
Backtest output also includes core risk/return stats (return, CAGR, volatility, Sharpe, max drawdown).

### `trade-assist recommend`

Purpose: compare current holdings against model target holdings and return actions.

```bash
trade-assist recommend --config config.recommend.json
```

Templates:

- `config.recommend.example.json`
- `config.msft_nvda.example.json`

## Documentation

- Quickstart (how to configure and test your own symbols/positions): `QUICKSTART.md`
- Full config parameter reference: `CONFIG.md`

Policy files:

- Store policy JSON files under `policies/` (for example `policies/baseline.json`, `policies/defensive.json`, `policies/aggressive.json`).
- Select one in config with `policy_path`.

## Notes

- Configs are validated against JSON Schema before execution.
- Outputs are model-generated signals and not investment advice.
