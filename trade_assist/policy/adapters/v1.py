from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..backtest import run_policy
from ..config import PolicyConfig
from ..models import BacktestResult
from ..recommendations import Recommendation, recommend_positions


class V1PolicyAdapter:
    policy_type = "v1"

    def policy_schema(self) -> dict[str, Any]:
        schema_path = (
            Path(__file__).resolve().parents[2] / "schemas" / "policy.schema.json"
        )
        with schema_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def default_policy_payload(self) -> dict[str, Any]:
        return asdict(PolicyConfig())

    def build_policy_config(
        self, payload: dict[str, Any] | None = None
    ) -> PolicyConfig:
        return PolicyConfig.from_dict(payload or {})

    def run_backtest(
        self,
        ohlcv_map: dict[str, pd.DataFrame],
        index_close: pd.Series | pd.DataFrame,
        policy_config: PolicyConfig,
        initial_cash: float,
        initial_positions: dict[str, float] | None = None,
    ) -> BacktestResult:
        return run_policy(
            ohlcv_map=ohlcv_map,
            index_close=index_close,
            config=policy_config,
            initial_cash=initial_cash,
            initial_positions=initial_positions,
        )

    def recommend(
        self,
        ohlcv_map: dict[str, pd.DataFrame],
        index_close: pd.Series | pd.DataFrame,
        policy_config: PolicyConfig,
        current_positions: dict[str, float],
        current_cash: float,
        min_trade_shares: float,
    ) -> tuple[list[Recommendation], pd.Series, int]:
        return recommend_positions(
            ohlcv_map=ohlcv_map,
            index_close=index_close,
            current_positions=current_positions,
            current_cash=current_cash,
            config=policy_config,
            min_trade_shares=min_trade_shares,
        )


ADAPTER = V1PolicyAdapter()
