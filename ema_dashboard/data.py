from __future__ import annotations

"""Data layer for Dhan intraday candles.

This module:
1) fetches raw JSON from API,
2) validates expected keys,
3) converts payload into analysis-ready OHLCV DataFrame.
"""

from datetime import datetime, time

import pandas as pd
import requests

from .config import DHAN_INTRADAY_URL, REQUIRED_KEYS

MARKET_OPEN_TIME = time(9, 15)


def fetch_dhan_data(
    access_token: str,
    security_id: str,
    interval: int,
    from_date_val,
    to_date_val,
    from_time_val: time,
    to_time_val: time,
) -> tuple[dict | None, str | None]:
    """Fetch intraday candles from Dhan API for the selected date-time window."""
    # Request full selected dates and apply intraday time filter locally.
    # This avoids API boundary quirks where the first session candle can be skipped.
    from_date = datetime.combine(from_date_val, time(0, 0)).strftime("%Y-%m-%d %H:%M:%S")
    to_date = datetime.combine(to_date_val, time(23, 59, 59)).strftime("%Y-%m-%d %H:%M:%S")

    # Dhan does not provide native 3-min interval for this endpoint.
    api_interval = 1 if interval == 3 else interval
    headers = {"access-token": access_token, "Content-Type": "application/json"}
    payload = {
        "securityId": security_id,
        "exchangeSegment": "IDX_I",
        "instrument": "INDEX",
        "interval": api_interval,
        "oi": False,
        "fromDate": from_date,
        "toDate": to_date,
    }

    response = requests.post(DHAN_INTRADAY_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return None, "API Error: " + response.text

    return response.json(), None


def validate_response(data: dict) -> list[str]:
    """Return missing required keys, if any."""
    return [k for k in REQUIRED_KEYS if k not in data]


def build_intraday_df(data: dict, interval: int, from_time_val: time, to_time_val: time) -> pd.DataFrame:
    """Convert raw response to local-time OHLCV DataFrame and apply time filter."""
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(data["timestamp"], unit="s", utc=True)
            .tz_convert("Asia/Kolkata")
            .tz_localize(None),
            "Open": data["open"],
            "High": data["high"],
            "Low": data["low"],
            "Close": data["close"],
            "Volume": data["volume"],
        }
    )

    if df.empty:
        return df

    # Normalize to minute boundary (some feeds emit :59 second close labels).
    df["Date"] = df["Date"].dt.floor("min")
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    if df.index.has_duplicates:
        # Keep one canonical OHLCV row per minute if duplicate timestamps appear.
        df = df.groupby(level=0).agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )

    def align_day_start_labels(frame: pd.DataFrame, step_minutes: int) -> pd.DataFrame:
        """Align each day's first candle label to market open when feed is close-time labelled."""
        adjusted_days = []
        max_shift = pd.Timedelta(minutes=max(1, int(step_minutes)))
        for day, day_df in frame.groupby(frame.index.date):
            session_start = pd.Timestamp.combine(pd.Timestamp(day).date(), MARKET_OPEN_TIME)
            first_ts = day_df.index.min()
            delta = first_ts - session_start
            if pd.Timedelta(0) < delta <= max_shift:
                day_df = day_df.copy()
                day_df.index = day_df.index - delta
            adjusted_days.append(day_df)
        return pd.concat(adjusted_days).sort_index() if adjusted_days else frame

    # Fix 1-minute close-time labels (e.g., first bar appears at 09:16).
    df = align_day_start_labels(df, step_minutes=1)

    if interval == 3:
        # Build 3-min candles from 1-min candles.
        df = df.resample("3min").agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        ).dropna(subset=["Open", "High", "Low", "Close"])
    elif interval > 1:
        # Fix native multi-minute close-time labels (e.g., first 5m bar at 09:20).
        df = align_day_start_labels(df, step_minutes=interval)

    return df.between_time(from_time_val.strftime("%H:%M"), to_time_val.strftime("%H:%M"))
