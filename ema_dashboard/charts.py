from __future__ import annotations

"""Chart rendering helpers (mplfinance + lightweight-charts)."""

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

try:
    from streamlit_lightweight_charts import renderLightweightCharts
    HAS_LWC = True
except ImportError:
    HAS_LWC = False


def make_day_figure(day: str, day_df: pd.DataFrame, interval: int, show_volume: bool, price_bar_ratio: float):
    """Build matplotlib/mplfinance day chart with overlays and signal markers."""
    # Base indicator overlays.
    ema10_plot = mpf.make_addplot(day_df["EMA10"], color="blue", width=1)
    ema20_plot = mpf.make_addplot(day_df["EMA20"], color="red")
    dema_plot = mpf.make_addplot(day_df["DEMA100"], color="black", width=3)
    add_plots = [dema_plot, ema10_plot, ema20_plot]

    # Conditional markers only when at least one value exists.
    if day_df["Buy_Plot"].notna().any():
        add_plots.append(mpf.make_addplot(day_df["Buy_Plot"], type="scatter", marker="$B$", markersize=180, color="green"))
    if day_df["Sell_Plot"].notna().any():
        add_plots.append(mpf.make_addplot(day_df["Sell_Plot"], type="scatter", marker="$S$", markersize=180, color="red"))
    if day_df["GreenHammer_Plot"].notna().any():
        add_plots.append(mpf.make_addplot(day_df["GreenHammer_Plot"], type="scatter", marker="^", markersize=90, color="green"))
    if day_df["RedHammer_Plot"].notna().any():
        add_plots.append(mpf.make_addplot(day_df["RedHammer_Plot"], type="scatter", marker="s", markersize=90, color="red"))
    if day_df["InvRedHammer_Plot"].notna().any():
        add_plots.append(mpf.make_addplot(day_df["InvRedHammer_Plot"], type="scatter", marker="D", markersize=90, color="#7f1e1e"))

    chart_height_ratio = 5.0 + (float(price_bar_ratio) - 3.0) * 0.4
    chart_height_ratio = max(4.0, min(6.5, chart_height_ratio))

    hammer_mask = (
        day_df.get("GreenHammer", False).astype(bool)
        | day_df.get("RedHammer", False).astype(bool)
        | day_df.get("InvertedRedHammer", False).astype(bool)
    )
    hammer_lines = []
    if hammer_mask.any():
        seg_minutes = max(1, int(interval))
        for ts, row in day_df.loc[hammer_mask, ["High", "Low"]].iterrows():
            seg_end = pd.Timestamp(ts) + pd.Timedelta(minutes=seg_minutes * 3)
            hammer_lines.append([(pd.Timestamp(ts), float(row["High"])), (seg_end, float(row["High"]))])
            hammer_lines.append([(pd.Timestamp(ts), float(row["Low"])), (seg_end, float(row["Low"]))])

    plot_kwargs = {}
    if hammer_lines:
        plot_kwargs["alines"] = dict(alines=hammer_lines, colors="#000000", linewidths=2.1, alpha=0.9)

    fig, _ = mpf.plot(
        day_df,
        type="candle",
        style="yahoo",
        volume=show_volume,
        title=f"{day} | EMA Crossover | TF: {interval} min",
        addplot=add_plots,
        figratio=(8, chart_height_ratio),
        figscale=1.2,
        tight_layout=False,
        xrotation=0,
        xlim=(day_df.index.min(), day_df.index.max() + pd.Timedelta(minutes=int(interval))),
        returnfig=True,
        **plot_kwargs,
    )
    fig.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.12)
    return fig


def build_lwc_payload(day_df: pd.DataFrame):
    """Convert enriched DataFrame into lightweight-charts series + markers payload."""
    candle_data, ema10_data, ema20_data, dema_data, volume_data, markers = [], [], [], [], [], []
    for idx, row in day_df.iterrows():
        ts = int(pd.Timestamp(idx).timestamp())
        is_up = row["Close"] >= row["Open"]
        candle_data.append({"time": ts, "open": float(row["Open"]), "high": float(row["High"]), "low": float(row["Low"]), "close": float(row["Close"])})
        ema10_data.append({"time": ts, "value": float(row["EMA10"])})
        ema20_data.append({"time": ts, "value": float(row["EMA20"])})
        dema_data.append({"time": ts, "value": float(row["DEMA100"])})
        volume_data.append({"time": ts, "value": float(row["Volume"]), "color": "rgba(38,166,154,0.45)" if is_up else "rgba(239,83,80,0.45)"})

        if bool(row.get("GreenHammer", False)):
            markers.append({"time": ts, "position": "belowBar", "color": "#00a651", "shape": "circle", "text": "🔨"})
        if bool(row.get("RedHammer", False)):
            markers.append({"time": ts, "position": "belowBar", "color": "#d32f2f", "shape": "circle", "text": "🔨"})
        if bool(row.get("InvertedRedHammer", False)):
            markers.append({"time": ts, "position": "aboveBar", "color": "#7f1e1e", "shape": "arrowDown", "text": "🔨ᶦ"})
        if row["Signal"] == "B":
            markers.append({"time": ts, "position": "belowBar", "color": "#2e7d32", "shape": "arrowUp", "text": "B"})
        elif row["Signal"] == "S":
            markers.append({"time": ts, "position": "aboveBar", "color": "#c62828", "shape": "arrowDown", "text": "S"})

    return candle_data, ema10_data, ema20_data, dema_data, volume_data, markers


def render_lwc_day_chart(day: str, day_df: pd.DataFrame, key_suffix: str, interval: int, show_volume: bool, price_bar_ratio: float, st_module):
    """Render one day chart in Streamlit using lightweight-charts."""
    candle_data, ema10_data, ema20_data, dema_data, volume_data, markers = build_lwc_payload(day_df)

    series = [
        {
            "type": "Candlestick",
            "data": candle_data,
            "options": {
                "upColor": "#26a69a",
                "downColor": "#ef5350",
                "borderVisible": False,
                "wickUpColor": "#26a69a",
                "wickDownColor": "#ef5350",
            },
            "markers": markers,
        },
        {"type": "Line", "data": dema_data, "options": {"color": "#000000", "lineWidth": 3, "lineType": 1, "priceLineVisible": False}},
        {"type": "Line", "data": ema10_data, "options": {"color": "#2962FF", "lineWidth": 1, "priceLineVisible": False}},
        {"type": "Line", "data": ema20_data, "options": {"color": "#FF1744", "lineWidth": 2, "priceLineVisible": False}},
    ]

    hammer_mask = (
        day_df.get("GreenHammer", False).astype(bool)
        | day_df.get("RedHammer", False).astype(bool)
        | day_df.get("InvertedRedHammer", False).astype(bool)
    )
    if hammer_mask.any():
        seg_seconds = max(60, int(interval) * 60)
        for idx, row in day_df.loc[hammer_mask, ["High", "Low"]].iterrows():
            ts = int(pd.Timestamp(idx).timestamp())
            seg_end = ts + (seg_seconds * 3)

            series.append(
                {
                    "type": "Line",
                    "data": [{"time": ts, "value": float(row["High"])}, {"time": seg_end, "value": float(row["High"])}],
                    "options": {"color": "#000000", "lineWidth": 3, "priceLineVisible": False, "lastValueVisible": False},
                }
            )
            series.append(
                {
                    "type": "Line",
                    "data": [{"time": ts, "value": float(row["Low"])}, {"time": seg_end, "value": float(row["Low"])}],
                    "options": {"color": "#000000", "lineWidth": 3, "priceLineVisible": False, "lastValueVisible": False},
                }
            )

    if show_volume:
        series.append({"type": "Histogram", "data": volume_data, "options": {"priceFormat": {"type": "volume"}, "priceScaleId": ""}})

    height_px = int(max(360, min(760, 520 + (float(price_bar_ratio) - 3.0) * 35)))
    chart_config = {
        "chart": {
            "height": height_px,
            "layout": {"background": {"type": "solid", "color": "#f5f5f5"}, "textColor": "#222"},
            "grid": {"vertLines": {"color": "#e0e0e0"}, "horzLines": {"color": "#e0e0e0"}},
            "handleScroll": {"mouseWheel": True, "pressedMouseMove": True, "horzTouchDrag": True, "vertTouchDrag": True},
            "handleScale": {"axisPressedMouseMove": True, "mouseWheel": True, "pinch": True},
            "rightPriceScale": {"borderVisible": False, "minimumWidth": 62},
            "leftPriceScale": {"visible": False},
            "timeScale": {
                "borderVisible": False,
                "timeVisible": True,
                "secondsVisible": False,
                "rightOffset": 4,
                "fixLeftEdge": False,
                "fixRightEdge": False,
                "lockVisibleTimeRangeOnResize": False,
            },
            "crosshair": {"mode": 0},
        },
        "series": series,
    }

    st_module.markdown(f"### {day} | EMA Crossover | TF: {interval} min")
    renderLightweightCharts([chart_config], key=f"lwc_{key_suffix}_{day}")


def render_lwc_trade_chart(day_df: pd.DataFrame, trades_df: pd.DataFrame, interval: int, price_bar_ratio: float, st_module, key_suffix: str = "bt"):
    """Render trade entries/exits on top of candle chart."""
    candle_data, ema10_data, ema20_data, _, _, _ = build_lwc_payload(day_df)

    markers = []
    if {"EntryTime", "ExitTime", "EntryPrice", "ExitPrice", "Size"}.issubset(trades_df.columns):
        for idx, row in trades_df.iterrows():
            entry_ts = int(pd.Timestamp(row["EntryTime"]).timestamp())
            exit_ts = int(pd.Timestamp(row["ExitTime"]).timestamp())
            is_long = float(row["Size"]) > 0
            trade_id = row.get("TradeID", idx + 1)
            trade_id_text = f"T{int(trade_id)}" if pd.notna(trade_id) else f"T{idx + 1}"

            markers.append(
                {
                    "time": entry_ts,
                    "position": "belowBar" if is_long else "aboveBar",
                    "color": "#2e7d32" if is_long else "#c62828",
                    "shape": "arrowUp" if is_long else "arrowDown",
                    "text": f"{'LE' if is_long else 'SE'} {trade_id_text}",
                }
            )
            markers.append(
                {
                    "time": exit_ts,
                    "position": "aboveBar" if is_long else "belowBar",
                    "color": "#1b5e20" if is_long else "#8e0000",
                    "shape": "circle",
                    "text": f"{'LX' if is_long else 'SX'} {trade_id_text}",
                }
            )

    # Price + EMA overlays with trade markers.
    series = [
        {
            "type": "Candlestick",
            "data": candle_data,
            "options": {
                "upColor": "#26a69a",
                "downColor": "#ef5350",
                "borderVisible": False,
                "wickUpColor": "#26a69a",
                "wickDownColor": "#ef5350",
            },
            "markers": markers,
        },
        {"type": "Line", "data": ema10_data, "options": {"color": "#2962FF", "lineWidth": 1, "priceLineVisible": False}},
        {"type": "Line", "data": ema20_data, "options": {"color": "#FF1744", "lineWidth": 2, "priceLineVisible": False}},
    ]

    height_px = int(max(380, min(780, 560 + (float(price_bar_ratio) - 3.0) * 35)))
    chart_config = {
        "chart": {
            "height": height_px,
            "layout": {"background": {"type": "solid", "color": "#f5f5f5"}, "textColor": "#222"},
            "grid": {"vertLines": {"color": "#e0e0e0"}, "horzLines": {"color": "#e0e0e0"}},
            "handleScroll": {"mouseWheel": True, "pressedMouseMove": True, "horzTouchDrag": True, "vertTouchDrag": True},
            "handleScale": {"axisPressedMouseMove": True, "mouseWheel": True, "pinch": True},
            "rightPriceScale": {"borderVisible": False, "minimumWidth": 62},
            "leftPriceScale": {"visible": False},
            "timeScale": {
                "borderVisible": False,
                "timeVisible": True,
                "secondsVisible": False,
                "rightOffset": 4,
                "fixLeftEdge": False,
                "fixRightEdge": False,
                "lockVisibleTimeRangeOnResize": False,
            },
            "crosshair": {"mode": 0},
        },
        "series": series,
    }

    renderLightweightCharts([chart_config], key=f"lwc_{key_suffix}_trades")


__all__ = ["HAS_LWC", "make_day_figure", "build_lwc_payload", "render_lwc_day_chart", "render_lwc_trade_chart", "plt"]
