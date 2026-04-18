# trade_assist

`trade_assist` provides three CLI tools:

- `backtest`: run historical simulation of a policy.
- `recommend`: generate `BUY` / `SELL` / `HOLD` actions from current portfolio state.
- `tune`: run walk-forward tuning over a policy search space.

## Installation

```bash
cd /path/to/project_root
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

This installs the package and all runtime dependencies from `setup.py`.

For development (includes `pytest`, `black`, `mypy`, and `flake8`):

```bash
pip install -e ".[dev]"
```

## Subcommands

### `trade-assist backtest`

Purpose: evaluate how a policy would have behaved historically.

```bash
trade-assist backtest --config config.backtest.json
```

Template: `config.backtest.example.json`

Backtest supports optional diagnostics (verbose logs, CSV exports for equity/cash/positions, plots, and optional full-hold benchmark overlays) via the `output` section in config.
Backtest output also includes core risk/return stats (return, CAGR, volatility, Sharpe, max drawdown).
Policy configs also support liquidity limits and early-exit rules (stop-loss / trailing-stop / cooldown).

### `trade-assist recommend`

Purpose: compare current holdings against model target holdings and return actions.

```bash
trade-assist recommend --config config.recommend.json
```

Templates:

- `config.recommend.example.json`

### `trade-assist tune`

Purpose: search policy parameter combinations and report the best configuration.

```bash
trade-assist tune --config config.tune.json
```

Template: `config.tune.example.json`

Set `tuning.workers` to an integer for a fixed process count, or `null` to use one worker per available logical CPU (up to the number of trials).

## Documentation

- Quickstart (how to configure and test your own symbols/positions): `QUICKSTART.md`
- Full config parameter reference: `CONFIG.md`

## Quality Checks

```bash
make check
```

Individual commands:

```bash
make format
make check-format
make lint
make typecheck
make test
```

## Notes

- Configs are validated against JSON Schema before execution.
- `policy_type` defaults to `v1` and resolves through the policy adapter registry.
- New policy types can be added as adapter modules (`trade_assist/policy/adapters/*.py`)
  or as entry-point plugins (`trade_assist.policy_adapters`).
- Outputs are model-generated signals and not investment advice.
