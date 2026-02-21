# Dhan EMA Crossover Dashboard

This app fetches intraday index candles from Dhan, builds day-wise charts, and runs backtests in Streamlit.

## Recent Changes

### 1) Candle time alignment fixes
- Fixed first-candle alignment to market open (`09:15`) across 1m/3m/5m style views.
- Added robust per-day timestamp handling and minute normalization in `ema_dashboard/data.py`.

### 2) Chart layout and rendering improvements
- Moved **Data & Settings** into a vertical left panel (sidebar).
- Improved chart left-side spacing/padding so first candles and time labels are not clipped.
- Restored and validated Lightweight Charts integration (`streamlit-lightweight-charts`).

### 3) Continuation strategy port from Pine Script
- Implemented continuation-style entry logic in Python (`ema_dashboard/patterns.py`), including:
  - EMA touch + pattern-based zone confirmation
  - fresh-zone gating
  - zone width and big-candle filters
- Added position-state control so repeated entries do not fire while a trade is already open.

### 4) Exit engine enhancements (toggle-based)
- Added Pine-style toggleable exits:
  - `Fib Exit`
  - `Zone Exit`
  - `AtrExit`
- Added ATR trailing stop logic (long/short).
- Added exit reason tracking (`ExitReason`) and chart-level reason display (e.g., `LX_ATR`, `SX_ZONE`).

### 5) Strategy structure for future expansion
- Introduced strategy settings defaults and pass-through configuration model.
- Added **STR1 - Continuation** global enable toggle and grouped controls in sidebar.
- Structure prepared so new strategies (STR2/STR3...) can be added with their own grouped toggles.

### 6) Marker cleanup
- Reworked entry/exit markers to clear arrow icons with larger size.
- Removed garbled text from hammer/inverted-hammer markers; kept clean icon-only rendering.

## Main Files Updated
- `indexemaUserDefined.py`
- `ema_dashboard/data.py`
- `ema_dashboard/charts.py`
- `ema_dashboard/patterns.py`
- `ema_dashboard/__init__.py`

