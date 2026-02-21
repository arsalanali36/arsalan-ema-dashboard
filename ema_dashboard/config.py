"""Global configuration values for data fetch and pattern thresholds."""

ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzcxNzMzNzUxLCJpYXQiOjE3NzE2NDczNTEsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAxMzEwOTc2In0.bH0tAmQ6i5ovAscFvbX2r9yakvOaD0mdvPY_0Fk-Tdkc87QoHcb_UdlTiH65tt6bhcw6fLSh43kkoMATHrJVvw"

DHAN_INTRADAY_URL = "https://api.dhan.co/v2/charts/intraday"

# Candlestick pattern sensitivity controls.
MIN_BODY_SIZE = 0.5
WICK_RATIO = 2.5
PREV_BODY_MIN_PTS = 0.5

# Expected keys in raw Dhan intraday response.
REQUIRED_KEYS = ["timestamp", "open", "high", "low", "close", "volume"]
