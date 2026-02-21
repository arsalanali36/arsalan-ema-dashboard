from __future__ import annotations

"""Feature engineering for chart overlays and candle-pattern signals."""

import numpy as np
import pandas as pd


def prepare_day_df(day_df: pd.DataFrame, min_body_size: float, wick_ratio: float, prev_body_min_pts: float) -> pd.DataFrame:
    """Return day-level DataFrame enriched with EMAs, crossover signals and patterns."""
    day_df = day_df.copy()

    # Core trend indicators.
    day_df["EMA10"] = day_df["Close"].ewm(span=10, adjust=False).mean()
    day_df["EMA20"] = day_df["Close"].ewm(span=20, adjust=False).mean()
    e1 = day_df["Close"].ewm(span=100, adjust=False).mean()
    e2 = e1.ewm(span=100, adjust=False).mean()
    day_df["DEMA100"] = 2 * e1 - e2
    day_df["Signal"] = ""

    # EMA10/EMA20 crossover signal labels: B=buy, S=sell.
    for i in range(1, len(day_df)):
        pf = day_df["EMA10"].iloc[i - 1]
        ps = day_df["EMA20"].iloc[i - 1]
        cf = day_df["EMA10"].iloc[i]
        cs = day_df["EMA20"].iloc[i]
        if pf < ps and cf > cs:
            day_df.iloc[i, day_df.columns.get_loc("Signal")] = "B"
        elif pf > ps and cf < cs:
            day_df.iloc[i, day_df.columns.get_loc("Signal")] = "S"

    day_df["Buy_Plot"] = np.nan
    day_df["Sell_Plot"] = np.nan
    day_df.loc[day_df["Signal"] == "B", "Buy_Plot"] = day_df["Low"] * 0.995
    day_df.loc[day_df["Signal"] == "S", "Sell_Plot"] = day_df["High"] * 1.005

    body = (day_df["Close"] - day_df["Open"]).abs()
    lower_wick = day_df[["Open", "Close"]].min(axis=1) - day_df["Low"]
    upper_wick = day_df["High"] - day_df[["Open", "Close"]].max(axis=1)
    valid_body = body >= min_body_size

    # Hammer family patterns.
    day_df["GreenHammer"] = valid_body & (lower_wick >= wick_ratio * body) & (upper_wick <= body) & (day_df["Close"] > day_df["Open"])
    day_df["RedHammer"] = valid_body & (lower_wick >= wick_ratio * body) & (upper_wick <= body) & (day_df["Close"] < day_df["Open"])
    day_df["InvertedRedHammer"] = valid_body & (upper_wick >= wick_ratio * body) & (lower_wick <= body) & (day_df["Close"] < day_df["Open"])

    prev_open = day_df["Open"].shift(1)
    prev_close = day_df["Close"].shift(1)
    prev_body = (prev_open - prev_close).abs()
    curr_body = body

    # Two-candle reversal patterns.
    day_df["BullishEngulfing"] = (
        (prev_close < prev_open)
        & (day_df["Open"] < day_df["Close"])
        & (day_df["Open"] <= prev_close)
        & (day_df["Close"] >= prev_open)
        & (curr_body > (prev_open - prev_close))
        & (prev_body >= prev_body_min_pts)
    )
    day_df["BearishEngulfing"] = (
        (prev_close > prev_open)
        & (day_df["Open"] > day_df["Close"])
        & (day_df["Open"] >= prev_close)
        & (day_df["Close"] <= prev_open)
        & (curr_body > (prev_close - prev_open))
        & (prev_body >= prev_body_min_pts)
    )

    body_50pct = curr_body >= (prev_body * 0.5)
    day_df["BullishHarami"] = (
        (prev_close < prev_open)
        & (day_df["Open"] < day_df["Close"])
        & (day_df["Open"] > prev_close)
        & (day_df["Close"] < prev_open)
        & body_50pct
    )
    day_df["BearishHarami"] = (
        (prev_close > prev_open)
        & (day_df["Open"] > day_df["Close"])
        & (day_df["Open"] < prev_close)
        & (day_df["Close"] > prev_open)
        & body_50pct
    )

    day_df["BullishCandle"] = day_df["BullishEngulfing"] | day_df["BullishHarami"] | day_df["GreenHammer"]
    day_df["BearishCandle"] = day_df["BearishEngulfing"] | day_df["BearishHarami"] | day_df["InvertedRedHammer"] | day_df["RedHammer"]

    # Plot helper columns (NaN except where marker should be drawn).
    day_df["GreenHammer_Plot"] = np.nan
    day_df["RedHammer_Plot"] = np.nan
    day_df["InvRedHammer_Plot"] = np.nan
    day_df["BullishPattern_Plot"] = np.nan
    day_df["BearishPattern_Plot"] = np.nan
    day_df.loc[day_df["GreenHammer"], "GreenHammer_Plot"] = day_df["Low"]
    day_df.loc[day_df["RedHammer"], "RedHammer_Plot"] = day_df["Low"]
    day_df.loc[day_df["InvertedRedHammer"], "InvRedHammer_Plot"] = day_df["High"]
    day_df.loc[day_df["BullishCandle"], "BullishPattern_Plot"] = day_df["Low"] * 0.989
    day_df.loc[day_df["BearishCandle"], "BearishPattern_Plot"] = day_df["High"] * 1.011

    return day_df
