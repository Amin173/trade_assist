from __future__ import annotations

import json

import pandas as pd

from trade_assist import cli
from trade_assist.ta.constants import COL_CLOSE


def test_extract_data_cfg_defaults():
    payload = {"data": {"tickers": ["AAA"]}}
    (
        tickers,
        index_ticker,
        regime_tickers,
        period,
        interval,
        use_cache,
        cache_dir,
        cache_ttl_hours,
        force_refresh,
    ) = cli._extract_data_cfg(payload)
    assert tickers == ["AAA"]
    assert index_ticker == "SPY"
    assert regime_tickers == []
    assert period == "5y"
    assert interval == "1d"
    assert use_cache is False
    assert cache_dir == ".cache/market_data"
    assert cache_ttl_hours == 24.0
    assert force_refresh is False


def test_load_market_data_resolves_relative_cache_dir(monkeypatch, tmp_path):
    cfg_path = tmp_path / "configs" / "cfg.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("{}", encoding="utf-8")

    calls = []

    def fake_fetch_ohlcv_map(*args, **kwargs):
        tickers = kwargs.get("tickers", args[0] if args else [])
        period = kwargs.get("period")
        interval = kwargs.get("interval")
        use_cache = kwargs.get("use_cache")
        cache_dir = kwargs.get("cache_dir")
        cache_ttl_hours = kwargs.get("cache_ttl_hours")
        force_refresh = kwargs.get("force_refresh")
        calls.append(
            {
                "tickers": list(tickers),
                "period": period,
                "interval": interval,
                "use_cache": use_cache,
                "cache_dir": cache_dir,
                "cache_ttl_hours": cache_ttl_hours,
                "force_refresh": force_refresh,
            }
        )
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
        return {t: pd.DataFrame({COL_CLOSE: [123.0]}, index=idx) for t in tickers}

    monkeypatch.setattr(cli, "fetch_ohlcv_map", fake_fetch_ohlcv_map)

    payload = {
        "data": {
            "tickers": ["AAA"],
            "index_ticker": "SPY",
            "period": "1y",
            "interval": "1d",
            "use_cache": True,
            "cache_dir": "cache",
            "cache_ttl_hours": 48,
            "force_refresh": False,
        }
    }

    ohlcv_map, index_close = cli._load_market_data(payload, cfg_path)
    assert list(ohlcv_map.keys()) == ["AAA"]
    assert float(index_close.iloc[0]) == 123.0

    expected_cache_dir = (cfg_path.parent / "cache").resolve()
    assert len(calls) == 2
    assert calls[0]["cache_dir"] == expected_cache_dir
    assert calls[1]["cache_dir"] == expected_cache_dir


def test_load_market_data_with_regime_tickers_returns_close_matrix(
    monkeypatch, tmp_path
):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text("{}", encoding="utf-8")

    def fake_fetch_ohlcv_map(*args, **kwargs):
        tickers = kwargs.get("tickers", args[0] if args else [])
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
        out = {}
        for i, ticker in enumerate(tickers):
            out[ticker] = pd.DataFrame({COL_CLOSE: [100.0 + i]}, index=idx)
        return out

    monkeypatch.setattr(cli, "fetch_ohlcv_map", fake_fetch_ohlcv_map)

    payload = {
        "data": {
            "tickers": ["AAA"],
            "index_ticker": "SPY",
            "regime_tickers": ["QQQ", "IWM"],
        }
    }

    _, regime_close = cli._load_market_data(payload, cfg_path)
    assert isinstance(regime_close, pd.DataFrame)
    assert list(regime_close.columns) == ["SPY", "QQQ", "IWM"]


def test_resolve_policy_config_relative_path(tmp_path):
    cfg_path = tmp_path / "cfg" / "backtest.json"
    policy_path = cfg_path.parent / "policies" / "test_policy.json"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("{}", encoding="utf-8")
    policy_path.write_text(json.dumps({"max_weight": 0.2}), encoding="utf-8")

    payload = {"policy_path": "policies/test_policy.json"}
    cfg = cli._resolve_policy_config(payload, cfg_path)
    assert cfg.max_weight == 0.2


def test_compute_performance_stats_short_series_returns_zeros():
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
    eq = pd.Series([100_000.0], index=idx, name="equity")
    stats = cli._compute_performance_stats(eq)
    assert all(v == 0.0 for v in stats.values())
