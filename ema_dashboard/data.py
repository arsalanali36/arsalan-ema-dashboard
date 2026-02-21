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
    from_date = datetime.combine(from_date_val, from_time_val).strftime("%Y-%m-%d %H:%M:%S")
    to_date = datetime.combine(to_date_val, to_time_val).strftime("%Y-%m-%d %H:%M:%S")

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

    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

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

    return df.between_time(from_time_val.strftime("%H:%M"), to_time_val.strftime("%H:%M"))
