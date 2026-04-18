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
make dev-setup
```

The preferred developer workflow is to use `make` for setup, formatting, testing, and release tasks.
`make dev-setup` installs the package plus the development tooling used by the repo.

If you only want the runtime package without the dev tools, use:

```bash
pip install -e .
```

If you prefer the explicit lower-level install command, `make dev-setup` is equivalent to:

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

Preferred developer commands:

```bash
make dev-setup
make format
make check-format
make lint
make typecheck
make test
make build
make release
```

`make dev-setup` installs the development dependencies needed for formatting, testing, and packaging.
`make build` creates source and wheel artifacts in `dist/` without changing code or running tests.
`make release` auto-formats the Python codebase, runs the test suite, and then builds the release artifacts.

## Notes

- Configs are validated against JSON Schema before execution.
- `policy_type` defaults to `v1` and resolves through the policy adapter registry.
- New policy types can be added as adapter modules (`trade_assist/policy/adapters/*.py`)
  or as entry-point plugins (`trade_assist.policy_adapters`).
- Outputs are model-generated signals and not investment advice.
