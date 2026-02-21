from __future__ import annotations

"""Feature engineering for chart overlays and candle-pattern signals."""

import numpy as np
import pandas as pd


def _default_strategy_settings() -> dict:
    """Defaults for continuation strategy; extend this dict for new strategy families."""
    return {
        "strategy_id": "continuation_v1",
        "enabled": True,
        "use_fresh_zone_only": True,
        "skip_big_candle": True,
        "max_zone_age": 2,
        "max_zone_distance": 25.0,
        "max_candle_size": 25.0,
        "exit_fib_enabled": True,
        "exit_zone_enabled": True,
        "exit_atr_enabled": True,
        "atr_len": 14,
        "atr_mult": 2.0,
    }


def prepare_day_df(
    day_df: pd.DataFrame,
    min_body_size: float,
    wick_ratio: float,
    prev_body_min_pts: float,
    strategy_settings: dict | None = None,
) -> pd.DataFrame:
    """Return day-level DataFrame enriched with indicators, patterns and continuation entry signals."""
    day_df = day_df.copy()

    # Core trend indicators.
    day_df["EMA10"] = day_df["Close"].ewm(span=10, adjust=False).mean()
    day_df["EMA20"] = day_df["Close"].ewm(span=20, adjust=False).mean()
    e1 = day_df["Close"].ewm(span=100, adjust=False).mean()
    e2 = e1.ewm(span=100, adjust=False).mean()
    day_df["DEMA100"] = 2 * e1 - e2

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

    # Three-candle star patterns used in continuation zone confirmation.
    body_2 = body.shift(2)
    body_1 = body.shift(1)
    first_midpoint = (day_df["Open"].shift(2) + day_df["Close"].shift(2)) / 2.0
    price_above_dema = day_df["Close"] > day_df["DEMA100"]
    price_below_dema = day_df["Close"] < day_df["DEMA100"]
    day_df["MorningStar"] = (
        (day_df["Close"].shift(2) < day_df["Open"].shift(2))
        & (body_2 > body_1)
        & (body_1 < body_2 * 0.5)
        & (day_df["Close"] > day_df["Open"])
        & (day_df["Close"] > first_midpoint)
        & price_above_dema
    )
    day_df["EveningStar"] = (
        (day_df["Close"].shift(2) > day_df["Open"].shift(2))
        & (body_2 > body_1)
        & (body_1 < body_2 * 0.5)
        & (day_df["Close"] < day_df["Open"])
        & (day_df["Close"] < first_midpoint)
        & price_below_dema
    )

    # Pine continuation entry logic port.
    ema_touch = (
        ((day_df["Low"] <= day_df["EMA10"]) & (day_df["High"] >= day_df["EMA10"]))
        | ((day_df["Low"] <= day_df["EMA20"]) & (day_df["High"] >= day_df["EMA20"]))
    )
    bullish_exit_zone = ema_touch & (day_df["BullishEngulfing"] | day_df["MorningStar"] | day_df["GreenHammer"])
    bearish_exit_zone = ema_touch & (day_df["BearishEngulfing"] | day_df["EveningStar"] | day_df["InvertedRedHammer"])

    cfg = _default_strategy_settings()
    if strategy_settings:
        cfg.update(strategy_settings)
    use_fresh_zone_only = bool(cfg["use_fresh_zone_only"])
    skip_big_candle = bool(cfg["skip_big_candle"])
    strategy_enabled = bool(cfg.get("enabled", True))
    max_zone_age = int(cfg["max_zone_age"])
    max_zone_distance = float(cfg["max_zone_distance"])
    max_candle_size = float(cfg["max_candle_size"])
    exit_fib_enabled = bool(cfg["exit_fib_enabled"])
    exit_zone_enabled = bool(cfg["exit_zone_enabled"])
    exit_atr_enabled = bool(cfg["exit_atr_enabled"])
    atr_len = max(1, int(cfg["atr_len"]))
    atr_mult = float(cfg["atr_mult"])

    signal_values = np.full(len(day_df), "", dtype=object)
    exit_signal_values = np.full(len(day_df), "", dtype=object)
    exit_reason_values = np.full(len(day_df), "", dtype=object)
    zone_upper_arr = np.full(len(day_df), np.nan, dtype=float)
    zone_lower_arr = np.full(len(day_df), np.nan, dtype=float)
    green_fresh_arr = np.zeros(len(day_df), dtype=bool)
    red_fresh_arr = np.zeros(len(day_df), dtype=bool)

    last_green_zone_bar = None
    last_red_zone_bar = None
    green_zone_touch_and_candle = False
    red_zone_touch_and_candle = False
    skip_big_candle_zone_exit_long = False
    skip_big_candle_zone_exit_short = False
    zone_upper = np.nan
    zone_lower = np.nan
    position_state = 0  # 0=flat, 1=long, -1=short
    entry_bar_index = -1

    opens = day_df["Open"].to_numpy()
    highs = day_df["High"].to_numpy()
    lows = day_df["Low"].to_numpy()
    closes = day_df["Close"].to_numpy()
    dema = day_df["DEMA100"].to_numpy()
    prev_close = day_df["Close"].shift(1)
    true_range = pd.concat(
        [
            day_df["High"] - day_df["Low"],
            (day_df["High"] - prev_close).abs(),
            (day_df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / float(atr_len), adjust=False).mean().to_numpy()
    atr_sl_long = np.nan
    atr_sl_short = np.nan
    entry_fib0 = np.nan
    entry_fib100 = np.nan
    running_day_high = np.nan
    running_day_low = np.nan

    for i in range(len(day_df)):
        if np.isnan(running_day_high) or highs[i] > running_day_high:
            running_day_high = highs[i]
        if np.isnan(running_day_low) or lows[i] < running_day_low:
            running_day_low = lows[i]

        if bool(bullish_exit_zone.iloc[i]):
            last_green_zone_bar = i
        if bool(bearish_exit_zone.iloc[i]):
            last_red_zone_bar = i

        green_zone_fresh = last_green_zone_bar is not None and (i - last_green_zone_bar <= max_zone_age)
        red_zone_fresh = last_red_zone_bar is not None and (i - last_red_zone_bar <= max_zone_age)
        green_fresh_arr[i] = green_zone_fresh
        red_fresh_arr[i] = red_zone_fresh

        if bool(bullish_exit_zone.iloc[i]) or bool(bearish_exit_zone.iloc[i]):
            zone_upper = float(highs[i])
            zone_lower = float(lows[i])
            green_zone_touch_and_candle = bool(bullish_exit_zone.iloc[i])
            red_zone_touch_and_candle = bool(bearish_exit_zone.iloc[i])

        zone_upper_arr[i] = zone_upper
        zone_lower_arr[i] = zone_lower

        close_above_green_zone = not np.isnan(zone_upper) and closes[i] > zone_upper
        close_below_green_zone = not np.isnan(zone_lower) and closes[i] < zone_lower
        close_above_red_zone = not np.isnan(zone_upper) and closes[i] > zone_upper
        close_below_red_zone = not np.isnan(zone_lower) and closes[i] < zone_lower
        zone_distance = abs(zone_upper - zone_lower) if (not np.isnan(zone_upper) and not np.isnan(zone_lower)) else np.nan
        zone_not_too_wide = (np.isnan(zone_distance)) or (zone_distance <= max_zone_distance)

        if skip_big_candle:
            candle_size = highs[i] - lows[i]
            if close_below_red_zone:
                skip_big_candle_zone_exit_short = candle_size > max_candle_size
            if close_above_green_zone:
                skip_big_candle_zone_exit_long = candle_size > max_candle_size
        else:
            skip_big_candle_zone_exit_long = False
            skip_big_candle_zone_exit_short = False

        prev_green = green_zone_touch_and_candle and i > 0 and closes[i - 1] > opens[i - 1]
        prev_red = red_zone_touch_and_candle and i > 0 and closes[i - 1] < opens[i - 1]
        is_green_candle = closes[i] > opens[i]
        is_red_candle = closes[i] < opens[i]
        price_above_dema_u = closes[i] > dema[i]
        price_below_dema_u = closes[i] < dema[i]

        long_gate = green_zone_fresh if use_fresh_zone_only else green_zone_touch_and_candle
        short_gate = red_zone_fresh if use_fresh_zone_only else red_zone_touch_and_candle
        index_long_signal = strategy_enabled and (
            long_gate
            and close_above_green_zone
            and prev_green
            and price_above_dema_u
            and zone_not_too_wide
            and (not skip_big_candle_zone_exit_long)
            and is_green_candle
        )
        index_short_signal = strategy_enabled and (
            short_gate
            and close_below_red_zone
            and prev_red
            and price_below_dema_u
            and zone_not_too_wide
            and (not skip_big_candle_zone_exit_short)
            and is_red_candle
        )

        if index_long_signal:
            if position_state == 0:
                signal_values[i] = "B"
                position_state = 1
                entry_bar_index = i
                atr_sl_long = closes[i] - (atr[i] * atr_mult)
                atr_sl_short = np.nan
                entry_fib0 = running_day_high
                entry_fib100 = running_day_low
        elif index_short_signal:
            if position_state == 0:
                signal_values[i] = "S"
                position_state = -1
                entry_bar_index = i
                atr_sl_short = closes[i] + (atr[i] * atr_mult)
                atr_sl_long = np.nan
                entry_fib0 = running_day_high
                entry_fib100 = running_day_low

        long_exit_gate = red_zone_fresh if use_fresh_zone_only else red_zone_touch_and_candle
        short_exit_gate = green_zone_fresh if use_fresh_zone_only else green_zone_touch_and_candle
        index_long_exit_condition = close_below_red_zone and closes[i] < opens[i] and long_exit_gate
        index_short_exit_condition = close_above_red_zone and closes[i] > opens[i] and short_exit_gate
        fib_long_exit_condition = not np.isnan(entry_fib100) and closes[i] < entry_fib100
        fib_short_exit_condition = not np.isnan(entry_fib0) and closes[i] > entry_fib0
        if position_state == 1:
            new_long_sl = closes[i] - (atr[i] * atr_mult)
            if np.isnan(atr_sl_long) or new_long_sl > atr_sl_long:
                atr_sl_long = new_long_sl
        elif position_state == -1:
            new_short_sl = closes[i] + (atr[i] * atr_mult)
            if np.isnan(atr_sl_short) or new_short_sl < atr_sl_short:
                atr_sl_short = new_short_sl
        sl_long_atr = position_state == 1 and (not np.isnan(atr_sl_long)) and closes[i] < atr_sl_long
        sl_short_atr = position_state == -1 and (not np.isnan(atr_sl_short)) and closes[i] > atr_sl_short

        if position_state == 1 and i > entry_bar_index and strategy_enabled and exit_fib_enabled and fib_long_exit_condition:
            exit_signal_values[i] = "LX"
            exit_reason_values[i] = "FIB_LONG"
            position_state = 0
            entry_bar_index = -1
            atr_sl_long = np.nan
            atr_sl_short = np.nan
            entry_fib0 = np.nan
            entry_fib100 = np.nan
        elif position_state == 1 and i > entry_bar_index and strategy_enabled and exit_atr_enabled and sl_long_atr:
            exit_signal_values[i] = "LX"
            exit_reason_values[i] = "ATR_LONG"
            position_state = 0
            entry_bar_index = -1
            atr_sl_long = np.nan
            atr_sl_short = np.nan
            entry_fib0 = np.nan
            entry_fib100 = np.nan
        elif position_state == 1 and i > entry_bar_index and strategy_enabled and exit_zone_enabled and index_long_exit_condition:
            exit_signal_values[i] = "LX"
            exit_reason_values[i] = "ZONE_LONG"
            position_state = 0
            entry_bar_index = -1
            atr_sl_long = np.nan
            atr_sl_short = np.nan
            entry_fib0 = np.nan
            entry_fib100 = np.nan
        elif position_state == -1 and i > entry_bar_index and strategy_enabled and exit_fib_enabled and fib_short_exit_condition:
            exit_signal_values[i] = "SX"
            exit_reason_values[i] = "FIB_SHORT"
            position_state = 0
            entry_bar_index = -1
            atr_sl_long = np.nan
            atr_sl_short = np.nan
            entry_fib0 = np.nan
            entry_fib100 = np.nan
        elif position_state == -1 and i > entry_bar_index and strategy_enabled and exit_atr_enabled and sl_short_atr:
            exit_signal_values[i] = "SX"
            exit_reason_values[i] = "ATR_SHORT"
            position_state = 0
            entry_bar_index = -1
            atr_sl_long = np.nan
            atr_sl_short = np.nan
            entry_fib0 = np.nan
            entry_fib100 = np.nan
        elif position_state == -1 and i > entry_bar_index and strategy_enabled and exit_zone_enabled and index_short_exit_condition:
            exit_signal_values[i] = "SX"
            exit_reason_values[i] = "ZONE_SHORT"
            position_state = 0
            entry_bar_index = -1
            atr_sl_long = np.nan
            atr_sl_short = np.nan
            entry_fib0 = np.nan
            entry_fib100 = np.nan

    day_df["Signal"] = signal_values
    day_df["ExitSignal"] = exit_signal_values
    day_df["ExitReason"] = exit_reason_values
    day_df["StrategyId"] = str(cfg["strategy_id"])
    day_df["ZoneUpper"] = zone_upper_arr
    day_df["ZoneLower"] = zone_lower_arr
    day_df["GreenZoneFresh"] = green_fresh_arr
    day_df["RedZoneFresh"] = red_fresh_arr

    day_df["Buy_Plot"] = np.nan
    day_df["Sell_Plot"] = np.nan
    day_df["LongExit_Plot"] = np.nan
    day_df["ShortExit_Plot"] = np.nan
    day_df.loc[day_df["Signal"] == "B", "Buy_Plot"] = day_df["Low"] * 0.995
    day_df.loc[day_df["Signal"] == "S", "Sell_Plot"] = day_df["High"] * 1.005
    day_df.loc[day_df["ExitSignal"] == "LX", "LongExit_Plot"] = day_df["High"] * 1.004
    day_df.loc[day_df["ExitSignal"] == "SX", "ShortExit_Plot"] = day_df["Low"] * 0.996

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
