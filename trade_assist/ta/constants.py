from __future__ import annotations

# Market data columns
COL_OPEN = "Open"
COL_HIGH = "High"
COL_LOW = "Low"
COL_CLOSE = "Close"
COL_ADJ_CLOSE = "Adj Close"
COL_VOLUME = "Volume"

OHLCV_COLUMNS = (COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME)
FUZZY_COLUMN_ORDER = (COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_ADJ_CLOSE, COL_VOLUME)
CANONICAL_COLUMN_MAP = {
    "open": COL_OPEN,
    "high": COL_HIGH,
    "low": COL_LOW,
    "close": COL_CLOSE,
    "adj close": COL_ADJ_CLOSE,
    "adjclose": COL_ADJ_CLOSE,
    "volume": COL_VOLUME,
}

# Cache defaults
DEFAULT_CACHE_DIR = ".cache/market_data"
DEFAULT_CACHE_TTL_HOURS = 24.0

# Common historical settings
DEFAULT_PERIOD = "5y"
DEFAULT_INTERVAL = "1d"
DEFAULT_INDEX_TICKER = "SPY"
TRADING_DAYS_PER_YEAR = 252

# TA defaults
SMA_WINDOWS = (20, 50, 100, 200)
RSI_WINDOW = 14
ATR_WINDOW = 14
VOL_WINDOW = 20
LOOKBACK_SESSIONS = 252
PIVOT_LEFT = 3
PIVOT_RIGHT = 3
TOP_LEVEL_COUNT = 6
TOP_LEVEL_TOL = 0.010

# Signal thresholds
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Numeric helpers
EPSILON = 1e-12
