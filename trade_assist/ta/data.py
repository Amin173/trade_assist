from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf


def _normalize_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = df.copy()

    # yfinance can return MultiIndex columns depending on version/options.
    if isinstance(out.columns, pd.MultiIndex):
        for level in range(out.columns.nlevels):
            values = set(out.columns.get_level_values(level).astype(str))
            if ticker in values:
                out = out.xs(ticker, axis=1, level=level, drop_level=True)
                break

        # Drop constant levels if still MultiIndex.
        while isinstance(out.columns, pd.MultiIndex):
            dropped = False
            for level in range(out.columns.nlevels):
                if out.columns.get_level_values(level).nunique() == 1:
                    out = out.droplevel(level, axis=1)
                    dropped = True
                    break
            if not dropped:
                out.columns = [
                    " ".join(str(part) for part in col if str(part).strip()).strip()
                    for col in out.columns
                ]
                break

    canonical = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adjclose": "Adj Close",
        "volume": "Volume",
    }
    out = out.rename(columns=lambda c: canonical.get(str(c).strip().lower(), c))

    # Fallback: map fuzzy names like "Close VRT" -> "Close".
    for wanted in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if wanted in out.columns:
            continue
        w = wanted.lower()
        for col in out.columns:
            c = str(col).strip().lower()
            if w in c:
                out = out.rename(columns={col: wanted})
                break

    return out


def _validate_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(
            f"Missing expected OHLCV columns for {ticker}: {missing}. Got columns: {list(df.columns)}"
        )
    out = df.dropna(subset=["Close"]).sort_index()
    if out.empty:
        raise RuntimeError(f"No usable OHLCV rows after cleanup for {ticker}.")
    return out


def _cache_file_path(cache_dir: str | Path, ticker: str, period: str, interval: str) -> Path:
    safe_period = period.replace("/", "_").replace(" ", "")
    safe_interval = interval.replace("/", "_").replace(" ", "")
    return Path(cache_dir) / f"{ticker.upper()}__{safe_period}__{safe_interval}.csv"


def _is_cache_fresh(path: Path, cache_ttl_hours: float | None) -> bool:
    if not path.exists():
        return False
    if cache_ttl_hours is None:
        return True
    if cache_ttl_hours <= 0:
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    return age_hours <= cache_ttl_hours


def _read_cached_ohlcv(path: Path, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = _normalize_columns(df, ticker=ticker)
    return _validate_ohlcv(df, ticker=ticker)


def fetch_ohlcv(
    ticker: str,
    period: str = "3y",
    interval: str = "1d",
    use_cache: bool = False,
    cache_dir: str | Path = ".cache/market_data",
    cache_ttl_hours: float | None = 24.0,
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache_path = _cache_file_path(cache_dir=cache_dir, ticker=ticker, period=period, interval=interval)
    if use_cache and not force_refresh and _is_cache_fresh(cache_path, cache_ttl_hours):
        try:
            return _read_cached_ohlcv(cache_path, ticker=ticker)
        except Exception:
            # Fall back to a fresh download if cache is unreadable/corrupt.
            pass

    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if df.empty:
            raise RuntimeError(f"No data returned for {ticker}.")
    except Exception:
        if use_cache and cache_path.exists():
            return _read_cached_ohlcv(cache_path, ticker=ticker)
        raise
    df = _normalize_columns(df, ticker=ticker)
    df = _validate_ohlcv(df, ticker=ticker)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=True)

    return df


def fetch_ohlcv_map(
    tickers: list[str],
    period: str = "3y",
    interval: str = "1d",
    use_cache: bool = False,
    cache_dir: str | Path = ".cache/market_data",
    cache_ttl_hours: float | None = 24.0,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    return {
        ticker: fetch_ohlcv(
            ticker=ticker,
            period=period,
            interval=interval,
            use_cache=use_cache,
            cache_dir=cache_dir,
            cache_ttl_hours=cache_ttl_hours,
            force_refresh=force_refresh,
        )
        for ticker in tickers
    }
