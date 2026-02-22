# Dhan EMA Crossover Dashboard

This app fetches intraday index candles from Dhan, builds day-wise charts, and runs backtests in Streamlit.

## Recent Changes

### 0) Journal + Trade Management (Auto from Dhan)
- Added **Journal** page with auto-build from Dhan Trade History.
- Range, Daily, and Lifetime dashboards (tabs).
- Metrics + charts for points, net P/L, fixed costs, buy/sell averages, durations.
- Tagging (`STR Entry`) with persistent dropdown options and saved tags.
- Trade notes + multi-image uploads per trade; persisted to disk.
- Daily image gallery with click-to-open viewer and prev/next navigation.
- Import/Export tags and tag options; all data stored locally for session persistence.

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

## Local Persistent Data
- `journal_notes.json` (trade notes + image metadata)
- `journal_media/` (uploaded images)
- `journal_tags.csv` (saved STR Entry tags)
- `journal_tag_options.json` (custom tag options)
