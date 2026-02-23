from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trade_assist.ta.constants import COL_CLOSE, COL_HIGH, COL_LOW, COL_OPEN, COL_VOLUME


def make_ohlcv(
    periods: int = 320,
    start_price: float = 100.0,
    step: float = 0.3,
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=periods)
    close = start_price + np.arange(periods, dtype=float) * step
    close_s = pd.Series(close, index=idx)
    open_s = close_s.copy()
    high_s = close_s * 1.01
    low_s = close_s * 0.99
    volume_s = pd.Series(float(volume), index=idx)
    return pd.DataFrame(
        {
            COL_OPEN: open_s,
            COL_HIGH: high_s,
            COL_LOW: low_s,
            COL_CLOSE: close_s,
            COL_VOLUME: volume_s,
        }
    )


@pytest.fixture(name="ohlcv_factory")
def fixture_ohlcv_factory():
    return make_ohlcv
