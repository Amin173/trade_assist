from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from trade_assist import cli
from trade_assist.config_validation import ConfigValidationError
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
    adapter = cli._resolve_policy_adapter(payload)
    cfg = cli._resolve_policy_config(payload, cfg_path, adapter)
    assert cfg.max_weight == 0.2


def test_compute_performance_stats_short_series_returns_zeros():
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02")])
    eq = pd.Series([100_000.0], index=idx, name="equity")
    stats = cli._compute_performance_stats(eq)
    assert all(v == 0.0 for v in stats.values())


def test_build_parser_accepts_tune_subcommand():
    parser = cli.build_parser()
    args = parser.parse_args(["tune", "--config", "config.tune.json"])
    assert args.command == "tune"
    assert args.config == "config.tune.json"


def test_main_renders_config_validation_errors_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "recommend_from_config",
        lambda path: (_ for _ in ()).throw(
            ConfigValidationError(
                "Invalid recommend config: missing required field 'data' at top level."
            )
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["trade-assist", "recommend", "--config", "config.recommend.example.json"],
    )

    rc = cli.main()

    captured = capsys.readouterr()
    assert rc == 2
    assert "Error: Invalid recommend config: missing required field 'data'" in (
        captured.err
    )
    assert "file:" in captured.err
    assert captured.out == ""


def test_format_duration_humanizes_seconds():
    assert cli._format_duration(7.2) == "7s"
    assert cli._format_duration(65.0) == "1m 5s"
    assert cli._format_duration(3661.0) == "1h 1m 1s"


def test_format_clock_duration_uses_compact_timer():
    assert cli._format_clock_duration(7.2) == "00:07"
    assert cli._format_clock_duration(65.0) == "01:05"
    assert cli._format_clock_duration(3661.0) == "1:01:01"


def test_format_failed_criteria_preview_compacts_long_lists():
    assert cli._format_failed_criteria_preview([]) == "-"
    assert (
        cli._format_failed_criteria_preview(
            [
                "trade_count>=5 x1 (observed 3)",
                "max_drawdown>=-45.00% x1 (observed -50.00%)",
            ]
        )
        == "trade_count>=5 x1 (observed 3), max_drawdown>=-45.00% x1 (observed -50.00%)"
    )
    assert (
        cli._format_failed_criteria_preview(
            [
                "trade_count>=5 x1 (observed 3)",
                "max_drawdown>=-45.00% x1 (observed -50.00%)",
                "trade_count>=5 x2 (observed 1)",
            ]
        )
        == "trade_count>=5 x1 (observed 3), max_drawdown>=-45.00% x1 (observed -50.00%), +1 more"
    )


@dataclass
class _DummyBestTrial:
    trial_id: int
    score: float
    mean_cagr_pct: float
    mean_max_drawdown_pct: float
    mean_sharpe: float
    mean_trade_count: float


def test_build_tuning_summary_payload_includes_metric_explanations():
    payload = cli._build_tuning_summary_payload(
        status="completed",
        policy_type="v1",
        trial_count=12,
        target_trial_count=20,
        window_count=22,
        best_trial=_DummyBestTrial(
            trial_id=7,
            score=123.45,
            mean_cagr_pct=19.2,
            mean_max_drawdown_pct=-11.3,
            mean_sharpe=1.42,
            mean_trade_count=8.5,
        ),
        final_test_stats={"cagr_pct": 15.0, "sharpe": 1.1},
        run_log_path="/tmp/tune.jsonl",
    )
    assert payload["status"] == "completed"
    assert payload["best_trial_id"] == 7
    assert payload["target_trial_count"] == 20
    assert payload["run_log_path"] == "/tmp/tune.jsonl"
    assert payload["metric_explanations"]["best_score"].startswith("Composite")
    assert (
        payload["metric_explanations"]["final_test_stats.cagr_pct"]
        == "Annualized return over the final test period."
    )


def test_build_tuning_summary_payload_accepts_cached_trial_dict():
    payload = cli._build_tuning_summary_payload(
        status="completed",
        policy_type="v1",
        trial_count=5,
        target_trial_count=10,
        window_count=22,
        best_trial={
            "trial_id": 3,
            "score": 77.7,
            "mean_cagr_pct": 21.0,
            "mean_max_drawdown_pct": -9.5,
            "mean_sharpe": 1.9,
            "mean_trade_count": 6.0,
        },
        final_test_stats=None,
        run_log_path="/tmp/tune.jsonl",
    )
    assert payload["best_trial_id"] == 3
    assert payload["best_score"] == 77.7


def test_trials_dataframe_from_cache_log_sorts_best_first():
    log_data = {
        "trial_events": [
            {
                "trial": {
                    "trial_id": 2,
                    "score": 9.0,
                    "feasible": True,
                    "window_count": 2,
                    "failed_window_count": 0,
                    "failed_windows": [],
                    "failed_criteria": [],
                    "mean_cagr_pct": 10.0,
                    "mean_max_drawdown_pct": -5.0,
                    "mean_annualized_vol_pct": 12.0,
                    "mean_sharpe": 1.2,
                    "mean_sortino": 1.6,
                    "mean_trade_count": 7.0,
                    "overrides": {},
                    "policy_payload": {"max_weight": 0.3},
                }
            },
            {
                "trial": {
                    "trial_id": 1,
                    "score": 12.0,
                    "feasible": True,
                    "window_count": 2,
                    "failed_window_count": 0,
                    "failed_windows": [],
                    "failed_criteria": [],
                    "mean_cagr_pct": 11.0,
                    "mean_max_drawdown_pct": -4.0,
                    "mean_annualized_vol_pct": 11.0,
                    "mean_sharpe": 1.4,
                    "mean_sortino": 1.8,
                    "mean_trade_count": 8.0,
                    "overrides": {},
                    "policy_payload": {"max_weight": 0.4},
                }
            },
        ]
    }
    trials_df = cli._trials_dataframe_from_cache_log(log_data)
    assert list(trials_df["trial_id"]) == [1, 2]


def test_tune_from_config_handles_interrupt_and_writes_cached_summary(
    monkeypatch, tmp_path
):
    cfg_path = tmp_path / "config.tune.json"
    cfg_path.write_text(
        json.dumps(
            {
                "data": {"tickers": ["AAA"]},
                "portfolio": {"cash": 1000},
                "tuning": {
                    "method": "random",
                    "trials": 2,
                    "search_space": {
                        "max_weight": {"type": "float", "low": 0.2, "high": 0.6}
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    class _DummyAdapter:
        policy_type = "v1"

    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")])
    df = pd.DataFrame({COL_CLOSE: [100.0, 101.0]}, index=idx)

    monkeypatch.setattr(cli, "validate_config", lambda payload, command: None)
    monkeypatch.setattr(cli, "_resolve_policy_adapter", lambda payload: _DummyAdapter())
    monkeypatch.setattr(
        cli, "_resolve_policy_payload", lambda payload, cfg_path, adapter: {}
    )
    monkeypatch.setattr(
        cli, "_load_market_data", lambda payload, cfg_path: ({"AAA": df}, df[COL_CLOSE])
    )
    monkeypatch.setattr(
        "trade_assist.tuning.engine._build_walk_forward_windows",
        lambda dates, split_cfg: [object(), object()],
    )
    monkeypatch.setattr(
        "trade_assist.tuning.engine._iter_overrides",
        lambda search_space, tuning_cfg: [{"max_weight": 0.2}, {"max_weight": 0.4}],
    )
    monkeypatch.setattr(
        cli,
        "tune_policy",
        lambda **kwargs: (_ for _ in ()).throw(
            cli.TuningInterrupted(partial_trials=[], total_trials=2)
        ),
    )

    rc = cli.tune_from_config(str(cfg_path))
    assert rc == 130

    output_dir = tmp_path / "outputs" / "tuning"
    summary_path = output_dir / "tuning_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "aborted"
    cache_logs = list((output_dir / ".cache" / "tuning_runs").glob("*.jsonl"))
    assert len(cache_logs) == 1
