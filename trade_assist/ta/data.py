from __future__ import annotations

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


def fetch_ohlcv(ticker: str, period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")
    df = _normalize_columns(df, ticker=ticker)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(
            f"Missing expected OHLCV columns for {ticker}: {missing}. Got columns: {list(df.columns)}"
        )

    return df.dropna(subset=["Close"])


def fetch_ohlcv_map(
    tickers: list[str],
    period: str = "3y",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    return {ticker: fetch_ohlcv(ticker=ticker, period=period, interval=interval) for ticker in tickers}
