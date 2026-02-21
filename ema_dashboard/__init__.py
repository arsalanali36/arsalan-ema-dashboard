"""Public package surface for dashboard modules.

Importing from `ema_dashboard` gives a stable, high-level API so the app code
does not need to know each internal module path.
"""

# Chart capability flag.
from .charts import HAS_LWC

# Config constants.
from .config import ACCESS_TOKEN, MIN_BODY_SIZE, PREV_BODY_MIN_PTS, WICK_RATIO

# Data fetch + transform helpers.
from .data import build_intraday_df, fetch_dhan_data, validate_response

# HTML exporter.
from .exporters import build_tv_style_html_document

# Backtest engine and runners.
from .backtest import EmaCrossWithEma20Exit, run_ema20_exit_backtest, run_ema_variation_backtests

# Pattern and indicator preparation.
from .patterns import prepare_day_df

__all__ = [
    "ACCESS_TOKEN",
    "MIN_BODY_SIZE",
    "PREV_BODY_MIN_PTS",
    "WICK_RATIO",
    "HAS_LWC",
    "fetch_dhan_data",
    "validate_response",
    "build_intraday_df",
    "prepare_day_df",
    "EmaCrossWithEma20Exit",
    "run_ema20_exit_backtest",
    "run_ema_variation_backtests",
    "build_tv_style_html_document",
]
