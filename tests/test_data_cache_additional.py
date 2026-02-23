from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from trade_assist.ta.data import _cache_file_path, fetch_ohlcv


def test_fetch_ohlcv_stale_cache_triggers_download(tmp_path, ohlcv_factory):
    ticker = "AAA"
    period = "1y"
    interval = "1d"
    cache_dir = tmp_path / "cache"
    cache_path = _cache_file_path(cache_dir, ticker, period, interval)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stale = ohlcv_factory(periods=10, start_price=10.0)
    stale.to_csv(cache_path)

    fresh = ohlcv_factory(periods=14, start_price=20.0)
    with patch("trade_assist.ta.data.yf.download", return_value=fresh) as mock_download:
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
    assert len(out) == len(fresh)


def test_fetch_ohlcv_corrupt_fresh_cache_downloads_successfully(
    tmp_path, ohlcv_factory
):
    ticker = "AAA"
    period = "1y"
    interval = "1d"
    cache_dir = tmp_path / "cache"
    cache_path = _cache_file_path(cache_dir, ticker, period, interval)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("not,a,valid,ohlcv\n1,2,3,4\n", encoding="utf-8")

    fresh = ohlcv_factory(periods=9, start_price=30.0)
    with patch("trade_assist.ta.data.yf.download", return_value=fresh) as mock_download:
        out = fetch_ohlcv(
            ticker=ticker,
            period=period,
            interval=interval,
            use_cache=True,
            cache_dir=cache_dir,
            cache_ttl_hours=None,
            force_refresh=False,
        )

    mock_download.assert_called_once()
    assert len(out) == len(fresh)


def test_fetch_ohlcv_corrupt_cache_and_failed_download_raises(tmp_path):
    ticker = "AAA"
    period = "1y"
    interval = "1d"
    cache_dir = tmp_path / "cache"
    cache_path = _cache_file_path(cache_dir, ticker, period, interval)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("bad,data\nx,y\n", encoding="utf-8")

    with patch(
        "trade_assist.ta.data.yf.download", side_effect=RuntimeError("download error")
    ) as mock_download:
        with pytest.raises(Exception):
            fetch_ohlcv(
                ticker=ticker,
                period=period,
                interval=interval,
                use_cache=True,
                cache_dir=cache_dir,
                cache_ttl_hours=0,
                force_refresh=False,
            )

    mock_download.assert_called_once()
    assert Path(cache_path).exists()
