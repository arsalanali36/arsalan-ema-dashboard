"""Global configuration values for data fetch and pattern thresholds."""

import os

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv:
    # Load .env if present; safe if missing.
    load_dotenv()

ACCESS_TOKEN = os.getenv("DHAN_TOKEN", "<PASTE_DHAN_TOKEN_HERE>")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "")

DHAN_INTRADAY_URL = "https://api.dhan.co/v2/charts/intraday"

# Candlestick pattern sensitivity controls.
MIN_BODY_SIZE = 0.5
WICK_RATIO = 2.5
PREV_BODY_MIN_PTS = 0.5

# Expected keys in raw Dhan intraday response.
REQUIRED_KEYS = ["timestamp", "open", "high", "low", "close", "volume"]
