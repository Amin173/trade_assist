from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

from .models import BacktestResult
from .recommendations import Recommendation


class PolicyAdapter(Protocol):
    policy_type: str

    def policy_schema(self) -> dict[str, Any]: ...

    def default_policy_payload(self) -> dict[str, Any]: ...

    def build_policy_config(self, payload: dict[str, Any] | None = None) -> Any: ...

    def run_backtest(
        self,
        ohlcv_map: dict[str, pd.DataFrame],
        index_close: pd.Series | pd.DataFrame,
        policy_config: Any,
        initial_cash: float,
        initial_positions: dict[str, float] | None = None,
    ) -> BacktestResult: ...

    def recommend(
        self,
        ohlcv_map: dict[str, pd.DataFrame],
        index_close: pd.Series | pd.DataFrame,
        policy_config: Any,
        current_positions: dict[str, float],
        current_cash: float,
        min_trade_shares: float,
    ) -> tuple[list[Recommendation], pd.Series, int]: ...
