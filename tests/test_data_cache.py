from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from trade_assist.ta.data import _cache_file_path, fetch_ohlcv


def test_fetch_ohlcv_uses_fresh_cache_without_download(tmp_path, ohlcv_factory):
    ticker = "AAA"
    period = "1y"
    interval = "1d"
    cache_dir = tmp_path / "cache"
    cache_path = _cache_file_path(cache_dir, ticker, period, interval)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ohlcv_factory(periods=30).to_csv(cache_path)

    with patch("trade_assist.ta.data.yf.download") as mock_download:
        out = fetch_ohlcv(
            ticker=ticker,
            period=period,
            interval=interval,
            use_cache=True,
            cache_dir=cache_dir,
            cache_ttl_hours=None,
            force_refresh=False,
        )

    mock_download.assert_not_called()
    assert len(out) == 30


def test_fetch_ohlcv_falls_back_to_cache_when_download_fails(tmp_path, ohlcv_factory):
    ticker = "AAA"
    period = "1y"
    interval = "1d"
    cache_dir = tmp_path / "cache"
    cache_path = _cache_file_path(cache_dir, ticker, period, interval)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = ohlcv_factory(periods=20)
    cached.to_csv(cache_path)

    with patch(
        "trade_assist.ta.data.yf.download", side_effect=RuntimeError("network down")
    ) as mock_download:
        out = fetch_ohlcv(
            ticker=ticker,
            period=period,
            interval=interval,
            use_cache=True,
            cache_dir=cache_dir,
            cache_ttl_hours=0,
            force_refresh=False,
        )

    mock_download.assert_called_once()
    assert len(out) == len(cached)


def test_fetch_ohlcv_force_refresh_downloads_and_writes_cache(tmp_path, ohlcv_factory):
    ticker = "AAA"
    period = "1y"
    interval = "1d"
    cache_dir = tmp_path / "cache"
    cache_path = _cache_file_path(cache_dir, ticker, period, interval)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    old = ohlcv_factory(periods=10, start_price=10.0)
    old.to_csv(cache_path)

    fresh = ohlcv_factory(periods=12, start_price=20.0)
    with patch("trade_assist.ta.data.yf.download", return_value=fresh) as mock_download:
        out = fetch_ohlcv(
            ticker=ticker,
            period=period,
            interval=interval,
            use_cache=True,
            cache_dir=cache_dir,
            cache_ttl_hours=None,
            force_refresh=True,
        )

    mock_download.assert_called_once()
    assert len(out) == len(fresh)
    assert Path(cache_path).exists()
