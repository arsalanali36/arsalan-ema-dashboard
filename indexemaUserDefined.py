"""Main Streamlit app for EMA crossover charting, scan, and backtest workflow.

High-level flow:
1) Data & Settings: fetch candles and enrich with indicators/patterns.
2) Charts: visualize day-wise charts and prepare downloadable artifact.
3) Backtest: scan EMA pairs, run strategy, and inspect trade-level details.
"""

from datetime import time
from pathlib import Path
import base64
import hashlib
import json
from numbers import Number

import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

from ema_dashboard import (
    _default_strategy_settings,
    ACCESS_TOKEN,
    DHAN_CLIENT_ID,
    MIN_BODY_SIZE,
    PREV_BODY_MIN_PTS,
    WICK_RATIO,
    build_intraday_df,
    build_tv_style_html_document,
    fetch_dhan_data,
    prepare_day_df,
    run_str1_signal_backtest,
    run_str1_variation_backtests,
    validate_response,
)
from ema_dashboard.charts import HAS_LWC, make_day_figure, render_lwc_day_chart, render_lwc_trade_chart

from ema_dashboard.dhan_api import (
    fetch_option_chain,
    fetch_option_expiries,
    fetch_order_book,
    fetch_trade_book,
    fetch_trade_history_all,
)


st.set_page_config(page_title="Dhan EMA Crossover Dashboard", layout="wide")
st.title("Dhan EMA Crossover Dashboard")


# ---------- Session state bootstrap ----------
def init_state():
    strategy_defaults = _default_strategy_settings()
    defaults = {
        "security_id": "13",
        "from_date_val": pd.to_datetime("2026-02-17").date(),
        "to_date_val": pd.to_datetime("2026-02-19").date(),
        "from_time_val": time(9, 15),
        "to_time_val": time(15, 30),
        "interval": 3,
        "show_volume": False,
        "dhan_client_id": DHAN_CLIENT_ID,
        "dhan_token": "",
        "use_lightweight": True,
        "price_bar_ratio": 3.0,
        "day_results": [],
        "final_df": None,
        "download_bytes": None,
        "pages_added": 0,
        "range_caption": "",
        "download_mode": "lightweight-html" if HAS_LWC else "mplfinance",
        "download_ready": False,
        "download_name": "ema_multi_day_charts.pdf",
        "download_mime": "application/pdf",
        "backtest_stats": None,
        "backtest_trades": None,
        "backtest_error": "",
        "ema_scan_results": None,
        "bt_fast_ema": 10,
        "bt_slow_ema": 20,
        "graph_selected_trade_id": None,
        "strategy_id": strategy_defaults["strategy_id"],
        "str1_continuation_enabled": strategy_defaults["enabled"],
        "max_trades_per_day": strategy_defaults["max_trades_per_day"],
        "use_fresh_zone_only": strategy_defaults["use_fresh_zone_only"],
        "exit_fib_enabled": strategy_defaults["exit_fib_enabled"],
        "exit_zone_enabled": strategy_defaults["exit_zone_enabled"],
        "exit_atr_enabled": strategy_defaults["exit_atr_enabled"],
        "skip_big_candle": strategy_defaults["skip_big_candle"],
        "max_zone_age": strategy_defaults["max_zone_age"],
        "max_zone_distance": strategy_defaults["max_zone_distance"],
        "max_candle_size": strategy_defaults["max_candle_size"],
        "atr_len": strategy_defaults["atr_len"],
        "atr_mult": strategy_defaults["atr_mult"],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _effective_token() -> str:
    return st.session_state.get("dhan_token") or ACCESS_TOKEN


def _effective_client_id() -> str:
    return st.session_state.get("dhan_client_id") or DHAN_CLIENT_ID


def _fmt_value(metric_name: str, val) -> str:
    """Format metric values for UI cards/tables."""
    if isinstance(val, Number):
        num = float(val)
        if metric_name == "# Trades":
            return f"{int(round(num))}"
        if "[%]" in metric_name:
            return f"{num:.2f}%"
        if "[$]" in metric_name:
            return f"${num:.2f}"
        return f"{num:.2f}"
    return str(val)


def fetch_and_prepare():
    """Fetch API data, build DataFrame, enrich day-wise features, reset downstream state."""
    effective_token = _effective_token()
    if (not effective_token) or ("<PASTE_DHAN_TOKEN_HERE>" in effective_token):
        st.error("Please paste your Dhan access token in the sidebar (session-only) or set DHAN_TOKEN in .env.")
        return

    from_date_val = st.session_state["from_date_val"]
    to_date_val = st.session_state["to_date_val"]
    from_time_val = st.session_state["from_time_val"]
    to_time_val = st.session_state["to_time_val"]

    if from_date_val > to_date_val:
        st.error("From Date cannot be greater than To Date.")
        return

    with st.spinner("Fetching data from Dhan..."):
        data, error = fetch_dhan_data(
            effective_token,
            str(st.session_state["security_id"]),
            int(st.session_state["interval"]),
            from_date_val,
            to_date_val,
            from_time_val,
            to_time_val,
        )

    if error:
        st.error(error)
        return

    missing = validate_response(data)
    if missing:
        st.error(f"Unexpected API response. Missing keys: {missing}")
        st.write(data)
        return

    df = build_intraday_df(data, int(st.session_state["interval"]), from_time_val, to_time_val)
    if df.empty:
        st.error("No data available in selected time window.")
        return

    first_times = ", ".join(ts.strftime("%H:%M") for ts in df.index[:5])
    day_firsts = []
    for day, day_df in df.groupby(df.index.date):
        if not day_df.empty:
            day_firsts.append(f"{day}: {day_df.index.min().strftime('%H:%M')}")
    day_firsts_text = "; ".join(day_firsts)
    st.session_state["range_caption"] = (
        f"Fetched candles: {len(df)} | Range: {df.index.min()} to {df.index.max()} (Asia/Kolkata)"
        f" | First candles: {first_times} | Day starts: {day_firsts_text}"
    )

    strategy_settings = {
        "strategy_id": st.session_state["strategy_id"],
        "enabled": bool(st.session_state["str1_continuation_enabled"]),
        "max_trades_per_day": int(st.session_state["max_trades_per_day"]),
        "use_fresh_zone_only": bool(st.session_state["use_fresh_zone_only"]),
        "exit_fib_enabled": bool(st.session_state["exit_fib_enabled"]),
        "exit_zone_enabled": bool(st.session_state["exit_zone_enabled"]),
        "exit_atr_enabled": bool(st.session_state["exit_atr_enabled"]),
        "skip_big_candle": bool(st.session_state["skip_big_candle"]),
        "max_zone_age": int(st.session_state["max_zone_age"]),
        "max_zone_distance": float(st.session_state["max_zone_distance"]),
        "max_candle_size": float(st.session_state["max_candle_size"]),
        "atr_len": int(st.session_state["atr_len"]),
        "atr_mult": float(st.session_state["atr_mult"]),
    }

    day_results = []
    for day, day_df in df.groupby(df.index.date):
        if day_df.empty:
            continue
        ready_day_df = prepare_day_df(
            day_df,
            MIN_BODY_SIZE,
            WICK_RATIO,
            PREV_BODY_MIN_PTS,
            strategy_settings=strategy_settings,
        )
        day_results.append({"day": str(day), "df": ready_day_df})

    if not day_results:
        st.error("No chart pages generated for selected range.")
        return

    st.session_state["day_results"] = day_results
    st.session_state["final_df"] = pd.concat([d["df"] for d in day_results]).sort_index()
    st.session_state["pages_added"] = len(day_results)
    st.session_state["download_bytes"] = None
    st.session_state["download_ready"] = False
    st.session_state["download_mode"] = "lightweight-html" if (st.session_state["use_lightweight"] and HAS_LWC) else "mplfinance"
    st.session_state["download_name"] = "ema_multi_day_charts.pdf"
    st.session_state["download_mime"] = "application/pdf"
    st.session_state["backtest_stats"] = None
    st.session_state["backtest_trades"] = None
    st.session_state["backtest_error"] = ""
    st.session_state["ema_scan_results"] = None
    st.session_state["bt_fast_ema"] = 10
    st.session_state["bt_slow_ema"] = 20
    st.success("Data fetch ho gaya. Ab Charts page aur Backtest page use karein.")


def apply_scan_best_str1():
    scan_results = st.session_state.get("ema_scan_results")
    if isinstance(scan_results, pd.DataFrame) and not scan_results.empty:
        best = scan_results.iloc[0]
        st.session_state["exit_fib_enabled_pending"] = bool(int(best["Fib Exit"]))
        st.session_state["exit_zone_enabled_pending"] = bool(int(best["Zone Exit"]))
        st.session_state["exit_atr_enabled_pending"] = bool(int(best["AtrExit"]))
        st.session_state["atr_len_pending"] = int(best["ATR Len"])
        st.session_state["atr_mult_pending"] = float(best["ATR Mult"])
        st.rerun()
    else:
        st.session_state["scan_pair_warning"] = "Pehle STR1 Variation Scan run karein."


init_state()
if "exit_fib_enabled_pending" in st.session_state:
    st.session_state["exit_fib_enabled"] = bool(st.session_state.pop("exit_fib_enabled_pending"))
if "exit_zone_enabled_pending" in st.session_state:
    st.session_state["exit_zone_enabled"] = bool(st.session_state.pop("exit_zone_enabled_pending"))
if "exit_atr_enabled_pending" in st.session_state:
    st.session_state["exit_atr_enabled"] = bool(st.session_state.pop("exit_atr_enabled_pending"))
if "atr_len_pending" in st.session_state:
    st.session_state["atr_len"] = int(st.session_state.pop("atr_len_pending"))
if "atr_mult_pending" in st.session_state:
    st.session_state["atr_mult"] = float(st.session_state.pop("atr_mult_pending"))

with st.sidebar:
    st.subheader("Data & Settings")
    st.text_input("Dhan Token (Today)", key="dhan_token", type="password", help="Session-only; not saved to disk.")
    st.text_input("Dhan Client ID", key="dhan_client_id", help="Required for Option Chain/Live OI.")
    st.selectbox("Strategy", options=["continuation_v1"], key="strategy_id", help="Future me yahan multiple strategies add hongi.")
    st.text_input("Security ID (Index)", key="security_id")
    st.time_input("Entry Time", key="from_time_val")
    st.time_input("Exit Time", key="to_time_val")
    st.selectbox("Interval (Minutes)", [1, 3, 5, 15, 25, 60], key="interval")
    st.number_input("Price to Bar", min_value=0.5, max_value=10.0, step=0.1, key="price_bar_ratio")
    st.checkbox("Use Lightweight Charts", key="use_lightweight")
    st.date_input("From Date", key="from_date_val")
    st.date_input("To Date", key="to_date_val")
    st.checkbox("Show Volume Panel", key="show_volume")
    with st.expander("STR1 - Continuation", expanded=True):
        st.checkbox("Enable STR1 - Continuation", key="str1_continuation_enabled")
        st.number_input("Max Trades Per Day", min_value=1, max_value=20, step=1, key="max_trades_per_day")
        st.markdown("`Entry`")
        st.checkbox("Use Fresh Zone Only", key="use_fresh_zone_only")
        st.checkbox("Skip Big Candle", key="skip_big_candle")
        st.number_input("Zone Fresh Age", min_value=1, max_value=20, step=1, key="max_zone_age")
        st.number_input("Max Zone Distance", min_value=1.0, max_value=500.0, step=1.0, key="max_zone_distance")
        st.number_input("Max Candle Size", min_value=1.0, max_value=500.0, step=1.0, key="max_candle_size")
        st.markdown("`Exit`")
        st.checkbox("Fib Exit", key="exit_fib_enabled")
        st.checkbox("Zone Exit", key="exit_zone_enabled")
        st.checkbox("AtrExit", key="exit_atr_enabled")
        st.number_input("ATR Length", min_value=1, max_value=100, step=1, key="atr_len")
        st.number_input("ATR Mult", min_value=0.5, max_value=10.0, step=0.1, key="atr_mult")

    if st.button("Fetch Data", use_container_width=True):
        fetch_and_prepare()

    if st.session_state["range_caption"]:
        st.caption(st.session_state["range_caption"])

# ---------- Top-level page router ----------
page = st.radio(
    "Select Page",
    ["Charts", "Backtest", "Live Data", "Dhan Trades", "Journal"],
    horizontal=True,
)

if page == "Charts":
    # ---------- Page 2: chart rendering + download ----------
    st.subheader("Charts")

    if st.session_state["use_lightweight"] and not HAS_LWC:
        st.warning("lightweight-charts not installed. Run: `python -m pip install streamlit-lightweight-charts`")

    if not st.session_state["day_results"]:
        st.info("Upar Data & Settings panel se data fetch karein.")
    else:
        if st.session_state["range_caption"]:
            st.caption(st.session_state["range_caption"])

        if not st.session_state["download_ready"]:
            if st.button(f"Prepare Download ({st.session_state['pages_added']} pages)"):
                if st.session_state["use_lightweight"] and HAS_LWC:
                    height_px = int(max(360, min(760, 500 + (float(st.session_state["price_bar_ratio"]) - 3.0) * 25)))
                    html_doc = build_tv_style_html_document(st.session_state["day_results"], int(st.session_state["interval"]), height_px)
                    st.session_state["download_bytes"] = html_doc.encode("utf-8")
                    st.session_state["download_mode"] = "lightweight-html"
                    st.session_state["download_name"] = "ema_multi_day_tv_style.html"
                    st.session_state["download_mime"] = "text/html"
                    st.session_state["download_ready"] = True
                    st.rerun()

                if not st.session_state["download_ready"]:
                    pdf_buffer = io.BytesIO()
                    with PdfPages(pdf_buffer) as pdf:
                        for day_entry in st.session_state["day_results"]:
                            fig = make_day_figure(
                                day_entry["day"],
                                day_entry["df"],
                                int(st.session_state["interval"]),
                                bool(st.session_state["show_volume"]),
                                float(st.session_state["price_bar_ratio"]),
                            )
                            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)
                    st.session_state["download_bytes"] = pdf_buffer.getvalue()
                    st.session_state["download_mode"] = "mplfinance"
                    st.session_state["download_name"] = "ema_multi_day_charts.pdf"
                    st.session_state["download_mime"] = "application/pdf"
                    st.session_state["download_ready"] = True
                    st.rerun()
        else:
            st.download_button(
                label=f"Download File ({st.session_state['pages_added']} pages)",
                data=st.session_state["download_bytes"],
                file_name=st.session_state["download_name"],
                mime=st.session_state["download_mime"],
            )

        for i, day_entry in enumerate(st.session_state["day_results"]):
            day = day_entry["day"]
            day_df = day_entry["df"]
            st.caption(
                f"{day} -> GH: {int(day_df['GreenHammer'].sum())}, "
                f"RH: {int(day_df['RedHammer'].sum())}, "
                f"IRH: {int(day_df['InvertedRedHammer'].sum())}"
            )

            if st.session_state["use_lightweight"] and HAS_LWC:
                render_lwc_day_chart(
                    day,
                    day_df,
                    str(i),
                    int(st.session_state["interval"]),
                    bool(st.session_state["show_volume"]),
                    float(st.session_state["price_bar_ratio"]),
                    st,
                )
            else:
                fig = make_day_figure(
                    day,
                    day_df,
                    int(st.session_state["interval"]),
                    bool(st.session_state["show_volume"]),
                    float(st.session_state["price_bar_ratio"]),
                )
                st.pyplot(fig, use_container_width=False)

        st.success(
            f"Prepared {st.session_state['pages_added']} chart page(s). "
            f"Download mode: {st.session_state['download_mode']}."
        )

        st.markdown("#### Combined Data")
        st.dataframe(st.session_state["final_df"].reset_index(), use_container_width=True)

elif page == "Backtest":
    # ---------- Page 3: STR1 scan + backtest + reporting ----------
    st.subheader("Backtest (STR1 - Continuation)")

    if st.session_state["final_df"] is None or st.session_state["final_df"].empty:
        st.info("Upar Data & Settings panel se data fetch karein.")
    else:
        st.markdown("### STR1 Exit-Criteria Variation Scan")
        sc1, sc2, sc3 = st.columns(3)
        fib_on = sc1.checkbox("Scan Fib Exit ON", value=True, key="scan_fib_on")
        zone_on = sc2.checkbox("Scan Zone Exit ON", value=True, key="scan_zone_on")
        atr_on = sc3.checkbox("Scan AtrExit ON", value=True, key="scan_atr_on")
        sc4, sc5 = st.columns(2)
        scan_atr_len_min = sc4.number_input("ATR Len Min", min_value=1, max_value=100, value=10, step=1)
        scan_atr_len_max = sc5.number_input("ATR Len Max", min_value=1, max_value=100, value=20, step=1)
        sc6, sc7 = st.columns(2)
        scan_atr_mult_min = sc6.number_input("ATR Mult Min", min_value=0.5, max_value=10.0, value=1.5, step=0.1)
        scan_atr_mult_max = sc7.number_input("ATR Mult Max", min_value=0.5, max_value=10.0, value=3.0, step=0.1)

        if st.button("Run STR1 Variation Scan"):
            try:
                if scan_atr_len_min > scan_atr_len_max or scan_atr_mult_min > scan_atr_mult_max:
                    raise ValueError("Min value max se bada nahi ho sakta.")
                atr_len_values = list(range(int(scan_atr_len_min), int(scan_atr_len_max) + 1))
                atr_mult_values = [round(x, 2) for x in pd.Series(np.arange(float(scan_atr_mult_min), float(scan_atr_mult_max) + 0.001, 0.1)).tolist()]
                base_settings = {
                    "strategy_id": st.session_state["strategy_id"],
                    "enabled": bool(st.session_state["str1_continuation_enabled"]),
                    "max_trades_per_day": int(st.session_state["max_trades_per_day"]),
                    "use_fresh_zone_only": bool(st.session_state["use_fresh_zone_only"]),
                    "skip_big_candle": bool(st.session_state["skip_big_candle"]),
                    "max_zone_age": int(st.session_state["max_zone_age"]),
                    "max_zone_distance": float(st.session_state["max_zone_distance"]),
                    "max_candle_size": float(st.session_state["max_candle_size"]),
                    "exit_fib_enabled": bool(st.session_state["exit_fib_enabled"]),
                    "exit_zone_enabled": bool(st.session_state["exit_zone_enabled"]),
                    "exit_atr_enabled": bool(st.session_state["exit_atr_enabled"]),
                    "atr_len": int(st.session_state["atr_len"]),
                    "atr_mult": float(st.session_state["atr_mult"]),
                }
                with st.spinner("STR1 combinations scan ho raha hai..."):
                    st.session_state["ema_scan_results"] = run_str1_variation_backtests(
                        st.session_state["final_df"],
                        min_body_size=MIN_BODY_SIZE,
                        wick_ratio=WICK_RATIO,
                        prev_body_min_pts=PREV_BODY_MIN_PTS,
                        base_settings=base_settings,
                        fib_options=[False, True] if fib_on else [base_settings["exit_fib_enabled"]],
                        zone_options=[False, True] if zone_on else [base_settings["exit_zone_enabled"]],
                        atr_options=[False, True] if atr_on else [base_settings["exit_atr_enabled"]],
                        atr_len_values=atr_len_values,
                        atr_mult_values=atr_mult_values,
                    )
            except Exception as exc:
                st.session_state["ema_scan_results"] = None
                st.error(f"STR1 scan failed: {exc}")

        scan_results = st.session_state.get("ema_scan_results")
        scan_pair_warning = st.session_state.pop("scan_pair_warning", "")
        if scan_pair_warning:
            st.warning(scan_pair_warning)
        if isinstance(scan_results, pd.DataFrame) and not scan_results.empty:
            # Show best setup summary and full ranked table.
            best_row = scan_results.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best Setup", f"F{int(best_row['Fib Exit'])}/Z{int(best_row['Zone Exit'])}/A{int(best_row['AtrExit'])}")
            c2.metric("Score", f"{float(best_row['Score']):.2f}")
            c3.metric("Return [%]", f"{float(best_row['Return [%]']):.2f}%")
            c4.metric("Sharpe", f"{float(best_row['Sharpe Ratio']):.2f}")
            st.caption("Score = Return + (20 x Sharpe) + (0.5 x Win Rate) - (0.7 x |Max Drawdown|), low-trade strategies penalized.")
            st.dataframe(
                scan_results.style.format(
                    {
                        "Score": "{:.2f}",
                        "Return [%]": "{:.2f}",
                        "Sharpe Ratio": "{:.2f}",
                        "Win Rate [%]": "{:.2f}",
                        "Max. Drawdown [%]": "{:.2f}",
                        "# Trades": "{:.0f}",
                        "Equity Final [$]": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )

            if st.button("Use Scan Best Setup"):
                apply_scan_best_str1()
                st.success("Best STR1 setup sidebar toggles par apply kar diya gaya.")

        st.markdown("---")
        st.markdown("### Run Backtest with Selected STR1 Toggles")

        if st.button("Run Backtest"):
            try:
                # Keep backtest aligned with what is currently visible on day charts.
                # This avoids any chart/backtest mismatch from re-computation differences.
                if not st.session_state["day_results"]:
                    raise ValueError("No prepared day results found. Please fetch data first.")
                bt_df = pd.concat([d["df"] for d in st.session_state["day_results"]]).sort_index()
                _, stats = run_str1_signal_backtest(bt_df)
                st.session_state["backtest_stats"] = stats
                st.session_state["backtest_trades"] = stats["_trades"].copy()
                st.session_state["backtest_error"] = ""
                st.session_state["selected_trade_id"] = None
                st.session_state["graph_selected_trade_id"] = None
            except Exception as exc:
                st.session_state["backtest_stats"] = None
                st.session_state["backtest_trades"] = None
                st.session_state["backtest_error"] = str(exc)

        if st.session_state["backtest_error"]:
            st.error(f"Backtest failed: {st.session_state['backtest_error']}")
        elif st.session_state["backtest_stats"] is not None:
            # Backtest outputs and derived views.
            st.caption(
                "Applied STR1: "
                f"MaxTrades/Day={int(st.session_state['max_trades_per_day'])}, "
                f"Fib={int(bool(st.session_state['exit_fib_enabled']))}, "
                f"Zone={int(bool(st.session_state['exit_zone_enabled']))}, "
                f"Atr={int(bool(st.session_state['exit_atr_enabled']))}, "
                f"ATR({int(st.session_state['atr_len'])},{float(st.session_state['atr_mult']):.1f})"
            )
            stats = st.session_state["backtest_stats"]
            stats_view = stats.drop(labels=["_strategy", "_equity_curve", "_trades"], errors="ignore")

            trades_df = st.session_state["backtest_trades"].copy()
            for col in trades_df.select_dtypes(include="number").columns:
                trades_df[col] = trades_df[col].round(2)
            if "TradeID" not in trades_df.columns:
                trades_df.insert(0, "TradeID", range(1, len(trades_df) + 1))

            req_cols = {"EntryTime", "ExitTime", "EntryPrice", "ExitPrice", "Size"}
            chart_ready = req_cols.issubset(trades_df.columns)
            chart_trades_df = trades_df.copy()
            if chart_ready:
                chart_trades_df["EntryTime"] = pd.to_datetime(chart_trades_df["EntryTime"])
                chart_trades_df["ExitTime"] = pd.to_datetime(chart_trades_df["ExitTime"])

            chart_price_df = st.session_state["final_df"]
            chart_markers_df = chart_trades_df

            st.markdown("### Strategy Chart")

            if chart_ready:
                if HAS_LWC:
                    st.caption(f"TV Style | TF: {int(st.session_state['interval'])} min")
                    render_lwc_trade_chart(
                        chart_price_df,
                        chart_markers_df,
                        int(st.session_state["interval"]),
                        float(st.session_state["price_bar_ratio"]),
                        st,
                        key_suffix="backtest",
                    )
                else:
                    price_df = chart_price_df[["Close"]].copy()
                    fig, ax = plt.subplots(figsize=(14, 5))
                    ax.plot(price_df.index, price_df["Close"], color="#1f77b4", linewidth=1.2, label="Close")
                    long_trades = chart_markers_df[chart_markers_df["Size"] > 0].copy()
                    short_trades = chart_markers_df[chart_markers_df["Size"] < 0].copy()
                    ax.scatter(long_trades["EntryTime"], long_trades["EntryPrice"], marker="^", s=45, color="green", label="Long Entry")
                    ax.scatter(long_trades["ExitTime"], long_trades["ExitPrice"], marker="v", s=45, color="#0b6623", label="Long Exit")
                    ax.scatter(short_trades["EntryTime"], short_trades["EntryPrice"], marker="v", s=45, color="red", label="Short Entry")
                    ax.scatter(short_trades["ExitTime"], short_trades["ExitPrice"], marker="^", s=45, color="#8b0000", label="Short Exit")
                    ax.set_title("Price with All Trade Entries/Exits")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Price")
                    ax.grid(alpha=0.25)
                    ax.legend(loc="best", ncol=2)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            else:
                st.info("Trade chart unavailable: required trade columns not found.")

            st.markdown("### Strategy Report")
            # Four report surfaces: key metrics, trade list, per-trade graph, and summary table.
            metrics_tab, trades_tab, graph_tab, summary_tab = st.tabs(
                ["Metrics", "List of trades", "Graph of trade", "Backtest Summary"]
            )

            with metrics_tab:
                st.markdown("#### Performance Dashboard")
                metric_keys = [
                    "Equity Final [$]",
                    "Return [%]",
                    "# Trades",
                    "Win Rate [%]",
                    "Max. Drawdown [%]",
                    "Sharpe Ratio",
                ]
                metric_cols = st.columns(3)
                for idx, key in enumerate(metric_keys):
                    if key in stats.index:
                        metric_cols[idx % 3].metric(key, _fmt_value(key, stats[key]))

                eq_curve = stats.get("_equity_curve")
                if isinstance(eq_curve, pd.DataFrame) and "Equity" in eq_curve.columns:
                    st.markdown("#### Equity Curve")
                    eq_plot_df = eq_curve[["Equity"]].copy().dropna()
                    if not eq_plot_df.empty:
                        eq_min = float(eq_plot_df["Equity"].min())
                        eq_max = float(eq_plot_df["Equity"].max())
                        pad = max(10.0, (eq_max - eq_min) * 0.2)
                        y_range = [eq_min - pad, eq_max + pad]
                        eq_fig = go.Figure()
                        eq_fig.add_trace(
                            go.Scatter(
                                x=eq_plot_df.index,
                                y=eq_plot_df["Equity"],
                                mode="lines",
                                line=dict(color="#1f77b4", width=2),
                                hovertemplate="Time: %{x}<br>Equity: %{y:.2f}<extra></extra>",
                                name="Equity",
                            )
                        )
                        eq_fig.update_layout(
                            height=320,
                            margin=dict(l=20, r=20, t=10, b=20),
                            yaxis=dict(title="Equity", range=y_range),
                            xaxis=dict(title="Time"),
                            dragmode="zoom",
                            plot_bgcolor="#ffffff",
                            paper_bgcolor="#ffffff",
                            showlegend=False,
                        )
                        st.plotly_chart(eq_fig, use_container_width=True, config={"displaylogo": False})

            with graph_tab:
                st.markdown("#### Graph of Trade")
                with st.container(border=True):
                    if chart_ready:
                        trade_ids = sorted(chart_trades_df["TradeID"].astype(int).tolist())
                        if trade_ids and st.session_state.get("graph_selected_trade_id") not in trade_ids:
                            st.session_state["graph_selected_trade_id"] = trade_ids[0]
                    preview_trade_id = st.session_state.get("graph_selected_trade_id")
                    preview_chart_price_df = st.session_state["final_df"]
                    preview_chart_markers_df = chart_trades_df
                    if chart_ready and preview_trade_id is not None and preview_trade_id in chart_trades_df["TradeID"].values:
                        selected_trade = chart_trades_df[chart_trades_df["TradeID"] == preview_trade_id].iloc[0]
                        focus_day = pd.Timestamp(selected_trade["EntryTime"]).date()
                        sliced_df = preview_chart_price_df[preview_chart_price_df.index.date == focus_day]
                        if not sliced_df.empty:
                            preview_chart_price_df = sliced_df
                        preview_chart_markers_df = chart_trades_df[
                            (chart_trades_df["EntryTime"].dt.date == focus_day)
                            | (chart_trades_df["ExitTime"].dt.date == focus_day)
                        ]

                    if preview_trade_id is not None and chart_ready:
                        preview_left, preview_right = st.columns(
                            [80, 20],
                            vertical_alignment="top",
                        )

                        with preview_left:
                            st.markdown("##### A. Trade OHLC Chart")
                            if HAS_LWC:
                                render_lwc_trade_chart(
                                    preview_chart_price_df,
                                    preview_chart_markers_df,
                                    int(st.session_state["interval"]),
                                    float(st.session_state["price_bar_ratio"]),
                                    st,
                                    key_suffix="backtest_preview",
                                )
                            else:
                                preview_price_df = preview_chart_price_df[["Close"]].copy()
                                preview_fig, preview_ax = plt.subplots(figsize=(14, 4.5))
                                preview_ax.plot(preview_price_df.index, preview_price_df["Close"], color="#1f77b4", linewidth=1.2, label="Close")
                                preview_long = preview_chart_markers_df[preview_chart_markers_df["Size"] > 0].copy()
                                preview_short = preview_chart_markers_df[preview_chart_markers_df["Size"] < 0].copy()
                                preview_ax.scatter(preview_long["EntryTime"], preview_long["EntryPrice"], marker="^", s=45, color="green", label="Long Entry")
                                preview_ax.scatter(preview_long["ExitTime"], preview_long["ExitPrice"], marker="v", s=45, color="#0b6623", label="Long Exit")
                                preview_ax.scatter(preview_short["EntryTime"], preview_short["EntryPrice"], marker="v", s=45, color="red", label="Short Entry")
                                preview_ax.scatter(preview_short["ExitTime"], preview_short["ExitPrice"], marker="^", s=45, color="#8b0000", label="Short Exit")
                                preview_ax.set_title("Selected Trade Preview")
                                preview_ax.grid(alpha=0.25)
                                preview_ax.legend(loc="best", ncol=2)
                                st.pyplot(preview_fig, use_container_width=True)
                                plt.close(preview_fig)

                            list_df = trades_df[["TradeID"]].copy()
                            pnl_col_list = next((c for c in trades_df.columns if c.lower() in {"pnl", "pnl [$]", "profit", "profit/loss"}), None)
                            if pnl_col_list is None:
                                pnl_col_list = next((c for c in trades_df.columns if "pnl" in c.lower()), None)
                            if pnl_col_list is not None:
                                list_df["PnL"] = pd.to_numeric(
                                    trades_df[pnl_col_list]
                                    .astype(str)
                                    .str.replace(",", "", regex=False)
                                    .str.replace("$", "", regex=False)
                                    .str.replace("₹", "", regex=False)
                                    .str.strip(),
                                    errors="coerce",
                                ).round(2)
                            else:
                                list_df["PnL"] = 0.0

                            list_df = list_df.sort_values("TradeID")
                            list_df["TradeID"] = pd.to_numeric(list_df["TradeID"], errors="coerce").fillna(0).astype(int)
                            list_df["PnL"] = pd.to_numeric(list_df["PnL"], errors="coerce").fillna(0.0)
                            list_df["Color"] = list_df["PnL"].apply(lambda x: "#1b8f4e" if x >= 0 else "#d64545")

                            focus_fig = go.Figure()
                            focus_fig.add_trace(
                                go.Bar(
                                    x=list_df["TradeID"].astype(str),
                                    y=list_df["PnL"],
                                    marker_color=list_df["Color"],
                                    hovertemplate="Trade ID: %{x}<br>PnL: %{y:.2f}<extra></extra>",
                                )
                            )
                            focus_fig.add_hline(y=0, line_dash="dash", line_color="#666", line_width=1)
                            focus_fig.update_layout(
                                height=170,
                                margin=dict(l=10, r=10, t=6, b=10),
                                xaxis_title="",
                                yaxis_title="PnL",
                                plot_bgcolor="#ffffff",
                                paper_bgcolor="#ffffff",
                                showlegend=False,
                            )
                            focus_event = st.plotly_chart(
                                focus_fig,
                                use_container_width=True,
                                key="trade_focus_chart_bottom",
                                on_select="rerun",
                                selection_mode="points",
                                config={"displaylogo": False},
                            )
                            focus_points = []
                            if focus_event is not None:
                                if hasattr(focus_event, "selection"):
                                    focus_points = focus_event.selection.get("points", [])
                                elif isinstance(focus_event, dict):
                                    focus_points = focus_event.get("selection", {}).get("points", [])
                            if focus_points:
                                selected_x = focus_points[0].get("x")
                                try:
                                    focused_trade_id = int(str(selected_x))
                                    if st.session_state.get("graph_selected_trade_id") != focused_trade_id:
                                        st.session_state["graph_selected_trade_id"] = focused_trade_id
                                        st.rerun()
                                except Exception:
                                    pass
                        with preview_right:
                            st.markdown("##### B. Trade Snapshot")
                            if chart_ready and trade_ids:
                                selected_trade_ui = st.selectbox(
                                    "Trade ID",
                                    options=trade_ids,
                                    index=trade_ids.index(int(preview_trade_id)),
                                    key="graph_trade_selector",
                                )
                                if int(selected_trade_ui) != int(preview_trade_id):
                                    st.session_state["graph_selected_trade_id"] = int(selected_trade_ui)
                                    st.rerun()
                            selected_trade_df = trades_df[trades_df["TradeID"] == preview_trade_id]
                            if not selected_trade_df.empty:
                                tr = selected_trade_df.iloc[0]
                                pnl_col_preview = next(
                                    (c for c in trades_df.columns if c.lower() in {"pnl", "pnl [$]", "profit", "profit/loss"}),
                                    None,
                                )
                                if pnl_col_preview is None:
                                    pnl_col_preview = next((c for c in trades_df.columns if "pnl" in c.lower()), None)

                                snapshot_items = [
                                    ("PnL", tr.get(pnl_col_preview) if pnl_col_preview is not None else None),
                                    ("Size", tr.get("Size")),
                                    ("Entry Price", tr.get("EntryPrice")),
                                    ("Exit Price", tr.get("ExitPrice")),
                                ]
                                dd_val = None
                                run_up_val = None
                                if "EntryTime" in tr.index and "ExitTime" in tr.index:
                                    et_trade = pd.to_datetime(tr.get("EntryTime"), errors="coerce")
                                    xt_trade = pd.to_datetime(tr.get("ExitTime"), errors="coerce")
                                    ep_trade = pd.to_numeric(tr.get("EntryPrice"), errors="coerce")
                                    sz_trade = pd.to_numeric(tr.get("Size"), errors="coerce")
                                    if pd.notna(et_trade) and pd.notna(xt_trade) and pd.notna(ep_trade) and pd.notna(sz_trade):
                                        trade_slice = st.session_state["final_df"].loc[
                                            (st.session_state["final_df"].index >= min(et_trade, xt_trade))
                                            & (st.session_state["final_df"].index <= max(et_trade, xt_trade))
                                        ]
                                        if not trade_slice.empty and "High" in trade_slice.columns and "Low" in trade_slice.columns:
                                            max_high = float(trade_slice["High"].max())
                                            min_low = float(trade_slice["Low"].min())
                                            if float(sz_trade) >= 0:
                                                run_up_val = max_high - float(ep_trade)
                                                dd_val = float(ep_trade) - min_low
                                            else:
                                                run_up_val = float(ep_trade) - min_low
                                                dd_val = max_high - float(ep_trade)

                                snapshot_items.append(("DD", dd_val))
                                snapshot_items.append(("Run up", run_up_val))
                                snap_df = pd.DataFrame(snapshot_items, columns=["Metric", "Value"])
                                snap_df["Value"] = pd.to_numeric(snap_df["Value"], errors="coerce")
                                snap_df = snap_df.dropna(subset=["Value"])
                                for _, metric_row in snap_df.iterrows():
                                    st.metric(str(metric_row["Metric"]), f"{float(metric_row['Value']):.2f}")

                                if "EntryTime" in tr.index and "ExitTime" in tr.index:
                                    et = pd.to_datetime(tr["EntryTime"], errors="coerce")
                                    xt = pd.to_datetime(tr["ExitTime"], errors="coerce")
                                    if pd.notna(et) and pd.notna(xt):
                                        dur_min = max(0.0, (xt - et).total_seconds() / 60.0)
                                        st.metric("Duration (min)", f"{dur_min:.1f}")

                                # Print-ready HTML export for Ctrl+P style high-quality output.
                                panel_df = preview_chart_price_df.copy()
                                x_start = panel_df.index.min()
                                x_end = panel_df.index.max()
                                if pd.notna(x_start) and pd.notna(x_end):
                                    x_pad = pd.Timedelta(minutes=max(2, int(st.session_state["interval"]) * 3))
                                    x_range = [x_start - x_pad, x_end + x_pad]
                                else:
                                    x_range = None

                                export_fig = go.Figure()
                                export_fig.add_trace(
                                    go.Candlestick(
                                        x=panel_df.index,
                                        open=panel_df["Open"],
                                        high=panel_df["High"],
                                        low=panel_df["Low"],
                                        close=panel_df["Close"],
                                        name="OHLC",
                                        increasing_line_color="#26a69a",
                                        decreasing_line_color="#ef5350",
                                    )
                                )
                                if "EMA10" in panel_df.columns:
                                    export_fig.add_trace(
                                        go.Scatter(
                                            x=panel_df.index,
                                            y=panel_df["EMA10"],
                                            mode="lines",
                                            name="EMA10",
                                            line=dict(color="#2962FF", width=1),
                                        )
                                    )
                                if "EMA20" in panel_df.columns:
                                    export_fig.add_trace(
                                        go.Scatter(
                                            x=panel_df.index,
                                            y=panel_df["EMA20"],
                                            mode="lines",
                                            name="EMA20",
                                            line=dict(color="#FF1744", width=2),
                                        )
                                    )

                                long_marks = preview_chart_markers_df[preview_chart_markers_df["Size"] > 0].copy()
                                short_marks = preview_chart_markers_df[preview_chart_markers_df["Size"] < 0].copy()
                                if x_range is not None:
                                    long_marks = long_marks[
                                        (long_marks["EntryTime"] >= x_range[0]) & (long_marks["EntryTime"] <= x_range[1])
                                    ]
                                    short_marks = short_marks[
                                        (short_marks["EntryTime"] >= x_range[0]) & (short_marks["EntryTime"] <= x_range[1])
                                    ]
                                if not long_marks.empty:
                                    export_fig.add_trace(
                                        go.Scatter(
                                            x=long_marks["EntryTime"],
                                            y=long_marks["EntryPrice"],
                                            mode="markers",
                                            name="Long Entry",
                                            marker=dict(symbol="triangle-up", color="#2e7d32", size=8),
                                        )
                                    )
                                    export_fig.add_trace(
                                        go.Scatter(
                                            x=long_marks["ExitTime"],
                                            y=long_marks["ExitPrice"],
                                            mode="markers",
                                            name="Long Exit",
                                            marker=dict(symbol="triangle-down", color="#1b5e20", size=8),
                                        )
                                    )
                                if not short_marks.empty:
                                    export_fig.add_trace(
                                        go.Scatter(
                                            x=short_marks["EntryTime"],
                                            y=short_marks["EntryPrice"],
                                            mode="markers",
                                            name="Short Entry",
                                            marker=dict(symbol="triangle-down", color="#c62828", size=8),
                                        )
                                    )
                                    export_fig.add_trace(
                                        go.Scatter(
                                            x=short_marks["ExitTime"],
                                            y=short_marks["ExitPrice"],
                                            mode="markers",
                                            name="Short Exit",
                                            marker=dict(symbol="triangle-up", color="#8e0000", size=8),
                                        )
                                    )

                                export_fig.update_layout(
                                    template="plotly_white",
                                    xaxis_rangeslider_visible=False,
                                    margin=dict(l=20, r=20, t=30, b=20),
                                    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
                                )
                                if x_range is not None:
                                    export_fig.update_xaxes(range=x_range, constrain="domain")

                                snapshot_rows = "".join(
                                    f"<tr><td>{metric_row['Metric']}</td><td>{float(metric_row['Value']):.2f}</td></tr>"
                                    for _, metric_row in snap_df.iterrows()
                                )
                                if "dur_min" in locals():
                                    snapshot_rows += f"<tr><td>Duration (min)</td><td>{dur_min:.1f}</td></tr>"

                                fig_json = export_fig.to_json()
                                html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Trade {int(preview_trade_id)} Print View</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 14px; color: #111; }}
    .toolbar {{ margin-bottom: 10px; }}
    .layout {{ display: grid; grid-template-columns: 72% 28%; gap: 12px; align-items: start; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #fff; }}
    .title {{ font-size: 18px; margin: 0 0 8px 0; }}
    .snap-table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    .snap-table td {{ padding: 7px 4px; border-bottom: 1px solid #f0f0f0; font-size: 15px; color: #111; }}
    .snap-table td:last-child {{ text-align: right; font-weight: 700; }}
    .snap-table tr:last-child td {{ border-bottom: none; }}
    @media print {{
      @page {{ size: A4 landscape; margin: 8mm; }}
      .toolbar {{ display: none; }}
      body {{ margin: 0; }}
      .layout {{ grid-template-columns: 72% 28%; gap: 10px; }}
      .card {{ border: none; }}
      #chart {{ height: 540px !important; }}
    }}
  </style>
</head>
<body>
  <div class="toolbar">
    <button onclick="window.print()">Print / Save PDF (Ctrl+P)</button>
  </div>
  <div class="layout">
    <div class="card">
      <h3 class="title">A. Trade OHLC Chart | Trade ID {int(preview_trade_id)}</h3>
      <div id="chart" style="height:620px;"></div>
    </div>
    <div class="card">
      <h3 class="title">B. Trade Snapshot</h3>
      <table class="snap-table">
        <tbody>
          {snapshot_rows}
        </tbody>
      </table>
    </div>
  </div>
  <script>
    const fig = {fig_json};
    Plotly.newPlot('chart', fig.data, fig.layout, {{displaylogo: false, responsive: true}});
  </script>
</body>
</html>"""
                                st.download_button(
                                    "Download Print HTML (Ctrl+P)",
                                    data=html_doc.encode("utf-8"),
                                    file_name=f"trade_{int(preview_trade_id)}_print_view.html",
                                    mime="text/html",
                                    key=f"download_ab_print_html_{int(preview_trade_id)}",
                                )
                    else:
                        st.info("Graph of trade tab open karte hi first trade load ho jayega. B section se trade change karein.")

            with summary_tab:
                st.markdown("#### Backtest Summary")
                summary_df = stats_view.rename_axis("Metric").reset_index(name="Value")
                summary_df["Value"] = summary_df.apply(lambda row: _fmt_value(row["Metric"], row["Value"]), axis=1)
                st.dataframe(summary_df, use_container_width=True)

            with trades_tab:
                title_col, options_col = st.columns([8, 1])
                title_col.markdown("#### List of Trades")
                with options_col:
                    with st.popover("Options"):
                        st.markdown("##### View Settings")
                        all_cols = trades_df.columns.tolist()
                        default_cols = [c for c in all_cols if c not in {"SL", "TP", "Tag"}]
                        visible_cols = st.multiselect(
                            "Visible Columns",
                            options=all_cols,
                            default=default_cols,
                            key="trades_visible_cols",
                        )
                        latest_first = st.checkbox("Latest Trade First", value=True, key="trades_latest_first")
                        sort_col = st.selectbox("Sort By", options=all_cols, index=0, key="trades_sort_col")
                        sort_asc = st.checkbox("Sort Ascending", value=False, key="trades_sort_asc")

                work_df = trades_df.copy()
                if latest_first:
                    work_df = work_df.sort_values("TradeID", ascending=False)
                if sort_col in work_df.columns:
                    work_df = work_df.sort_values(sort_col, ascending=sort_asc, na_position="last")
                if visible_cols:
                    display_cols = [c for c in visible_cols if c in work_df.columns]
                else:
                    display_cols = ["TradeID"]
                st.dataframe(work_df[display_cols], use_container_width=True, height=420)




elif page == "Live Data":
    st.subheader("Live Data (NIFTY OI)")

    st.caption("Note: Option Chain API is rate-limited (1 request / 3 seconds). OI updates slower than LTP.")

    token = _effective_token()
    client_id = _effective_client_id()
    if (not token) or ("<PASTE_DHAN_TOKEN_HERE>" in token):
        st.error("Please paste your Dhan access token in the sidebar (session-only) or set DHAN_TOKEN in .env.")
        st.stop()
    if not client_id:
        st.error("Please enter your Dhan Client ID in the sidebar (DHAN_CLIENT_ID).")
        st.stop()

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        with col1:
            underlying_scrip = st.number_input("Underlying Scrip (NIFTY)", min_value=1, step=1, value=int(st.session_state.get("security_id", 13)))
        with col2:
            underlying_seg = st.selectbox("Underlying Segment", options=["IDX_I", "NSE_FNO"], index=0)
        with col3:
            refresh = st.button("Load / Refresh OI", use_container_width=True)
        with col4:
            max_rows = st.number_input("Max Strikes", min_value=10, max_value=500, step=10, value=60)

    if refresh or ("oi_option_chain" not in st.session_state):
        with st.spinner("Fetching option chain..."):
            exp_data, exp_error = fetch_option_expiries(token, client_id, int(underlying_scrip), underlying_seg)
            if exp_error:
                st.error(exp_error)
                st.stop()
            expiries = exp_data.get("data", []) if isinstance(exp_data, dict) else []
            if not expiries:
                st.error("No expiries returned by Dhan API.")
                st.stop()
            st.session_state["oi_expiries"] = expiries
            st.session_state["oi_selected_expiry"] = expiries[0]

    expiries = st.session_state.get("oi_expiries", [])
    if expiries:
        selected_expiry = st.selectbox("Expiry", options=expiries, key="oi_selected_expiry")
    else:
        st.info("Click 'Load / Refresh OI' to fetch expiries.")
        st.stop()

    if refresh or ("oi_option_chain" not in st.session_state) or (st.session_state.get("oi_loaded_expiry") != selected_expiry):
        with st.spinner("Loading option chain..."):
            oc_data, oc_error = fetch_option_chain(token, client_id, int(underlying_scrip), underlying_seg, selected_expiry)
            if oc_error:
                st.error(oc_error)
                st.stop()
            st.session_state["oi_option_chain"] = oc_data
            st.session_state["oi_loaded_expiry"] = selected_expiry

    oc_payload = st.session_state.get("oi_option_chain", {})
    oc_data = oc_payload.get("data", {}) if isinstance(oc_payload, dict) else {}
    oc_map = oc_data.get("oc", {}) if isinstance(oc_data, dict) else {}
    if not oc_map:
        st.error("Option chain data not available.")
        st.stop()

    rows = []
    for strike_key, legs in oc_map.items():
        try:
            strike = float(strike_key)
        except Exception:
            continue
        ce = legs.get("ce", {}) if isinstance(legs, dict) else {}
        pe = legs.get("pe", {}) if isinstance(legs, dict) else {}
        ce_oi = ce.get("oi") or ce.get("open_interest") or 0
        pe_oi = pe.get("oi") or pe.get("open_interest") or 0
        ce_prev_oi = ce.get("previous_oi") or 0
        pe_prev_oi = pe.get("previous_oi") or 0
        ce_oi_change = ce.get("oi_change") or ce.get("oiChange")
        pe_oi_change = pe.get("oi_change") or pe.get("oiChange")
        if ce_oi_change is None:
            ce_oi_change = ce_oi - ce_prev_oi
        if pe_oi_change is None:
            pe_oi_change = pe_oi - pe_prev_oi
        rows.append({
            "Strike": strike,
            "CE_OI": ce_oi,
            "PE_OI": pe_oi,
            "CE_OI_Change": ce_oi_change,
            "PE_OI_Change": pe_oi_change,
            "CE_LTP": ce.get("ltp") or ce.get("last_price") or 0,
            "PE_LTP": pe.get("ltp") or pe.get("last_price") or 0,
            "CE_Vol": ce.get("volume") or 0,
            "PE_Vol": pe.get("volume") or 0,
        })

    oi_df = pd.DataFrame(rows)
    if oi_df.empty:
        st.error("No OI rows parsed from option chain.")
        st.stop()

    oi_df = oi_df.sort_values("Strike")
    for col in ["CE_OI", "PE_OI", "CE_OI_Change", "PE_OI_Change", "CE_LTP", "PE_LTP", "CE_Vol", "PE_Vol"]:
        oi_df[col] = pd.to_numeric(oi_df[col], errors="coerce").fillna(0)

    total_ce_oi = float(oi_df["CE_OI"].sum())
    total_pe_oi = float(oi_df["PE_OI"].sum())
    pcr = (total_pe_oi / total_ce_oi) if total_ce_oi else 0.0

    max_ce_row = oi_df.loc[oi_df["CE_OI"].idxmax()] if total_ce_oi else None
    max_pe_row = oi_df.loc[oi_df["PE_OI"].idxmax()] if total_pe_oi else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total CE OI", f"{total_ce_oi:,.0f}")
    c2.metric("Total PE OI", f"{total_pe_oi:,.0f}")
    c3.metric("PCR", f"{pcr:.2f}")
    if max_ce_row is not None:
        c4.metric("Max CE OI Strike", f"{max_ce_row['Strike']:.0f}")

    if max_pe_row is not None:
        st.caption(f"Max PE OI Strike: {max_pe_row['Strike']:.0f}")

    last_price = oc_data.get("last_price") or oc_data.get("underlyingPrice") or None
    if last_price is not None:
        st.caption(f"Underlying LTP: {last_price}")

    if last_price is not None:
        oi_df["ATM_Dist"] = (oi_df["Strike"] - float(last_price)).abs()
        view_df = oi_df.sort_values("ATM_Dist").head(int(max_rows)).sort_values("Strike")
    else:
        view_df = oi_df.tail(int(max_rows))

    st.markdown("### Option Chain (OI Focus)")
    st.dataframe(view_df.drop(columns=["ATM_Dist"], errors="ignore"), use_container_width=True, height=420)


elif page == "Dhan Trades":
    st.subheader("Dhan Trades / Orders")

    token = _effective_token()
    if (not token) or ("<PASTE_DHAN_TOKEN_HERE>" in token):
        st.error("Please paste your Dhan access token in the sidebar (session-only) or set DHAN_TOKEN in .env.")
        st.stop()

    st.caption("Note: Dhan order/trade book returns only current-day data per API docs.")
    use_history = st.checkbox("Use Trade History API (Date Range)", value=True, help="Uses Dhan trade history endpoint for past dates.")
    max_pages = st.number_input("History Pages (max)", min_value=1, max_value=200, step=1, value=20)
    st.caption("Today (local): February 22, 2026.")

    col1, col2, col3 = st.columns(3)
    with col1:
        from_date = st.date_input("From Date", value=pd.Timestamp.today().date(), key="orders_from_date")
    with col2:
        to_date = st.date_input("To Date", value=pd.Timestamp.today().date(), key="orders_to_date")
    with col3:
        refresh = st.button("Load Orders + Trades", use_container_width=True)

    if refresh or ("order_book_raw" not in st.session_state) or ("trade_book_raw" not in st.session_state):
        with st.spinner("Fetching order & trade book..."):
            orders_raw, orders_error = fetch_order_book(token)
            trades_raw = []
            trades_error = None
            if use_history:
                from_str = pd.to_datetime(from_date).strftime('%Y-%m-%d')
                to_str = pd.to_datetime(to_date).strftime('%Y-%m-%d')
                trades_raw, trades_error = fetch_trade_history_all(token, from_str, to_str, max_pages=int(max_pages))
            else:
                trades_raw, trades_error = fetch_trade_book(token)
            if orders_error:
                st.error(orders_error)
            if trades_error:
                st.error(trades_error)
            st.session_state["order_book_raw"] = orders_raw if orders_raw is not None else []
            st.session_state["trade_book_raw"] = trades_raw if trades_raw is not None else []

    orders_raw = st.session_state.get("order_book_raw", [])
    trades_raw = st.session_state.get("trade_book_raw", [])

    def _extract_list(payload):
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if "data" in payload:
                data = payload.get("data")
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    for key in ["trades", "orders", "records", "orderList", "tradeList"]:
                        if key in data and isinstance(data[key], list):
                            return data[key]
            for key in ["trades", "orders", "records", "orderList", "tradeList"]:
                if key in payload and isinstance(payload[key], list):
                    return payload[key]
            # numeric-key dict -> list of values
            if payload and all(str(k).isdigit() for k in payload.keys()):
                return list(payload.values())
        return []

    orders_list = _extract_list(orders_raw)
    trades_list = _extract_list(trades_raw)

    orders_df = pd.DataFrame(orders_list) if orders_list else pd.DataFrame()
    trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()

    def _parse_dt(df, cols):
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    orders_df = _parse_dt(orders_df, ["createdTime", "createTime", "orderTime", "exchangeTime"])
    trades_df = _parse_dt(trades_df, ["tradeTime", "exchangeTime", "createTime", "createdTime"])

    # Filters
    def _filter_by_date(df):
        date_cols = [c for c in ["createdTime", "createTime", "orderTime", "exchangeTime", "tradeTime"] if c in df.columns]
        if not date_cols:
            return df
        if not (from_date and to_date and from_date <= to_date):
            return df

        # Pick the date column that has the most valid timestamps (avoid "NA" fields).
        best_col = None
        best_series = None
        best_count = -1
        for col in date_cols:
            series = pd.to_datetime(df[col], errors="coerce")
            count = int(series.notna().sum())
            if count > best_count:
                best_col = col
                best_series = series
                best_count = count

        if best_col is None or best_count == 0:
            return df

        left = from_date
        right = to_date
        series_date = best_series.dt.date
        return df[(series_date >= left) & (series_date <= right)]

    orders_df = _filter_by_date(orders_df)
    trades_df = _filter_by_date(trades_df)

    with st.expander("Filters", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sym_opts = sorted(set(orders_df.get("tradingSymbol", [])).union(set(trades_df.get("tradingSymbol", []))))
            sym_filter = st.multiselect("Symbol", options=[s for s in sym_opts if s])
        with c2:
            status_opts = sorted(set(orders_df.get("orderStatus", [])))
            status_filter = st.multiselect("Order Status", options=[s for s in status_opts if s])
        with c3:
            product_opts = sorted(set(orders_df.get("productType", [])).union(set(trades_df.get("productType", []))))
            product_filter = st.multiselect("Product Type", options=[s for s in product_opts if s])
        with c4:
            side_opts = sorted(set(orders_df.get("transactionType", [])).union(set(trades_df.get("transactionType", []))))
            side_filter = st.multiselect("Side", options=[s for s in side_opts if s])

    def _apply_filters(df):
        out = df.copy()
        if sym_filter and "tradingSymbol" in out.columns:
            out = out[out["tradingSymbol"].isin(sym_filter)]
        if status_filter and "orderStatus" in out.columns:
            out = out[out["orderStatus"].isin(status_filter)]
        if product_filter and "productType" in out.columns:
            out = out[out["productType"].isin(product_filter)]
        if side_filter and "transactionType" in out.columns:
            out = out[out["transactionType"].isin(side_filter)]
        return out

    orders_df = _apply_filters(orders_df)
    trades_df = _apply_filters(trades_df)

    # Quick dashboard
    total_orders = len(orders_df)
    total_trades = len(trades_df)
    buy_qty = 0.0
    sell_qty = 0.0
    turnover = 0.0
    if not trades_df.empty:
        qty_col = next((c for c in trades_df.columns if c.lower() in {"quantity", "qty", "tradedquantity"}), None)
        price_col = next((c for c in trades_df.columns if c.lower() in {"price", "tradeprice", "tradedprice"}), None)
        side_col = next((c for c in trades_df.columns if c.lower() in {"transactiontype", "side"}), None)
        if qty_col:
            trades_df[qty_col] = pd.to_numeric(trades_df[qty_col], errors="coerce").fillna(0)
        if price_col:
            trades_df[price_col] = pd.to_numeric(trades_df[price_col], errors="coerce").fillna(0)
        if qty_col and price_col:
            turnover = float((trades_df[qty_col] * trades_df[price_col]).sum())
        if side_col and qty_col:
            buy_qty = float(trades_df[trades_df[side_col].astype(str).str.upper().str.contains("BUY")][qty_col].sum())
            sell_qty = float(trades_df[trades_df[side_col].astype(str).str.upper().str.contains("SELL")][qty_col].sum())

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Orders", f"{total_orders}")
    d2.metric("Trades", f"{total_trades}")
    d3.metric("Buy Qty", f"{buy_qty:,.0f}")
    d4.metric("Turnover", f"{turnover:,.2f}")

    if not trades_df.empty:
        time_col = next((c for c in ["tradeTime", "exchangeTime", "createdTime", "createTime"] if c in trades_df.columns), None)
        qty_col = next((c for c in trades_df.columns if c.lower() in {"quantity", "qty", "tradedquantity"}), None)
        price_col = next((c for c in trades_df.columns if c.lower() in {"price", "tradeprice", "tradedprice"}), None)
        side_col = next((c for c in trades_df.columns if c.lower() in {"transactiontype", "side"}), None)
        if time_col and qty_col and price_col:
            chart_df = trades_df.copy()
            chart_df[time_col] = pd.to_datetime(chart_df[time_col], errors="coerce")
            chart_df[qty_col] = pd.to_numeric(chart_df[qty_col], errors="coerce").fillna(0)
            chart_df[price_col] = pd.to_numeric(chart_df[price_col], errors="coerce").fillna(0)
            chart_df["Turnover"] = chart_df[qty_col] * chart_df[price_col]
            chart_df["TradeDate"] = chart_df[time_col].dt.date
            if side_col and side_col in chart_df.columns:
                side_upper = chart_df[side_col].astype(str).str.upper()
                chart_df["SignedValue"] = chart_df["Turnover"] * side_upper.apply(
                    lambda s: 1 if "SELL" in s else (-1 if "BUY" in s else 0)
                )
            else:
                chart_df["SignedValue"] = 0.0
            daily = chart_df.groupby("TradeDate").agg(Trades=(time_col, "count"), Turnover=("Turnover", "sum")).reset_index()
            daily_pnl = chart_df.groupby("TradeDate").agg(PnL=("SignedValue", "sum")).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=daily["TradeDate"].astype(str), y=daily["Trades"], name="Trades"))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title="Trades Per Day")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            fig2 = go.Figure()
            fig2.add_trace(
                go.Bar(
                    x=daily_pnl["TradeDate"].astype(str),
                    y=daily_pnl["PnL"],
                    name="PnL",
                    marker_color=daily_pnl["PnL"].apply(lambda x: "#1b8f4e" if x >= 0 else "#d64545"),
                )
            )
            fig2.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), title="PnL Per Day")
            st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})
        if side_col and qty_col:
            side_df = trades_df.copy()
            side_df[qty_col] = pd.to_numeric(side_df[qty_col], errors="coerce").fillna(0)
            side_df["Side"] = side_df[side_col].astype(str).str.upper()
            side_sum = side_df.groupby("Side")[qty_col].sum().reset_index()
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=side_sum["Side"], y=side_sum[qty_col], name="Qty"))
            fig3.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), title="Qty by Side")
            st.plotly_chart(fig3, use_container_width=True, config={"displaylogo": False})
    with st.expander("Raw API Response", expanded=False):
        st.json({"orders_raw": st.session_state.get("order_book_raw"), "trades_raw": st.session_state.get("trade_book_raw")})

    tab_orders, tab_trades = st.tabs(["Orders", "Trades"])
    with tab_orders:
        st.markdown("### Order Book")
        if orders_df.empty:
            st.info("No orders returned.")
        else:
            st.dataframe(orders_df, use_container_width=True, height=420)

    with tab_trades:
        st.markdown("### Trade Book")
        if trades_df.empty:
            st.info("No trades returned.")
        else:
            st.dataframe(trades_df, use_container_width=True, height=420)


elif page == "Journal":
    st.subheader("Journal (Auto from Dhan Trades)")
    st.caption("Auto-build journal from Dhan Trade History. Missing fields can be added later.")

    token = _effective_token()
    if (not token) or ("<PASTE_DHAN_TOKEN_HERE>" in token):
        st.error("Please paste your Dhan access token in the sidebar (session-only) or set DHAN_TOKEN in .env.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        j_from = st.date_input("From Date", value=pd.Timestamp.today().date(), key="journal_from_date")
    with col2:
        j_to = st.date_input("To Date", value=pd.Timestamp.today().date(), key="journal_to_date")
    with col3:
        j_pages = st.number_input("History Pages (max)", min_value=1, max_value=200, step=1, value=20, key="journal_pages")

    show_lifetime = st.checkbox("Show Lifetime Graphs", value=True, help="Fetch full history for lifetime cumulative graphs.")

    # Auto-reload when range changes
    last_range = st.session_state.get("journal_last_range")
    current_range = (j_from, j_to, int(j_pages))
    if last_range != current_range:
        st.session_state["journal_reload"] = True
        st.session_state["journal_last_range"] = current_range

    # Tags storage path (shared for export/import + table tagging)
    tags_path = Path(__file__).resolve().parent / "journal_tags.csv"

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
    with btn_col1:
        if st.button("Load Journal from Dhan"):
            st.session_state["journal_reload"] = True
            st.session_state["journal_lifetime_reload"] = True
    with btn_col2:
        if tags_path.exists():
            st.download_button(
                "Export Tags CSV",
                data=tags_path.read_bytes(),
                file_name="journal_tags.csv",
                mime="text/csv",
            )
    with btn_col3:
        up = st.file_uploader("Import Tags CSV", type=["csv"], accept_multiple_files=False, key="tags_upload")
        if up is not None:
            try:
                import_df = pd.read_csv(up)
                if "_tag_key" in import_df.columns and "STR Entry" in import_df.columns:
                    import_df[["_tag_key", "STR Entry"]].drop_duplicates().to_csv(tags_path, index=False)
                    st.success("Tags imported. Reload the page to see updates.")
                else:
                    st.error("Invalid tags file. Expected columns: _tag_key, STR Entry")
            except Exception as exc:
                st.error(f"Import failed: {exc}")

    # Backward compatible: if button not used, no reload.

    if st.session_state.get("journal_reload") or ("journal_trades_raw" not in st.session_state):
        with st.spinner("Fetching trade history..."):
            from_str = pd.to_datetime(j_from).strftime('%Y-%m-%d')
            to_str = pd.to_datetime(j_to).strftime('%Y-%m-%d')
            trades_raw, trades_error = fetch_trade_history_all(token, from_str, to_str, max_pages=int(j_pages))
            if trades_error:
                st.error(trades_error)
                trades_raw = []
            st.session_state["journal_trades_raw"] = trades_raw
            st.session_state["journal_reload"] = False

    if show_lifetime and (st.session_state.get("journal_lifetime_reload") or ("journal_trades_raw_lifetime" not in st.session_state)):
        with st.spinner("Fetching lifetime trade history..."):
            lifetime_from = "2000-01-01"
            lifetime_to = pd.Timestamp.today().strftime('%Y-%m-%d')
            trades_raw_life, trades_error_life = fetch_trade_history_all(token, lifetime_from, lifetime_to, max_pages=int(j_pages))
            if trades_error_life:
                st.error(trades_error_life)
                trades_raw_life = []
            st.session_state["journal_trades_raw_lifetime"] = trades_raw_life
            st.session_state["journal_lifetime_range"] = (lifetime_from, lifetime_to)
            st.session_state["journal_lifetime_reload"] = False

    trades_raw = st.session_state.get("journal_trades_raw", [])
    if not trades_raw:
        st.info("No trades returned for selected range.")
        st.stop()

    trades_df = pd.DataFrame(trades_raw)

    journal_cols = [
        "Date", "Week", "unique ID", "BUY Time", "SELL Time", "Strike", "L/S", "STR Entry", "Trade", "Qty", "BUY", "SELL",
        "Pt", "Invested", "% Return", "C%", "C Pt", "Dur", "Brokerage", "Other Charges", "Gross P/L", "Net P/L", "Total FC"
    ]

    def _build_journal_df(source_df: pd.DataFrame) -> pd.DataFrame:
        if source_df is None or source_df.empty:
            return pd.DataFrame(columns=journal_cols)

        time_col = next((c for c in ["exchangeTime", "tradeTime", "createdTime", "createTime"] if c in source_df.columns), None)
        qty_col = next((c for c in source_df.columns if c.lower() in {"tradedquantity", "quantity", "qty"}), None)
        price_col = next((c for c in source_df.columns if c.lower() in {"tradedprice", "price", "tradeprice"}), None)
        side_col = next((c for c in source_df.columns if c.lower() in {"transactiontype", "side"}), None)

        rows = []
        if time_col and qty_col and price_col and side_col:
            work = source_df.copy()
            work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
            work[qty_col] = pd.to_numeric(work[qty_col], errors="coerce")
            work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
            work["__side"] = work[side_col].astype(str).str.upper()
            work = work.sort_values(time_col)

            # FIFO pairing by date + symbol + qty
            from collections import defaultdict, deque

            buy_q = defaultdict(deque)
            sell_q = defaultdict(deque)
            trade_counter = defaultdict(int)

            for _, row in work.iterrows():
                dt = row.get(time_col)
                if pd.isna(dt):
                    continue
                date_key = dt.date()
                date_str = dt.strftime("%a %d - %b")
                symbol = row.get("customSymbol", row.get("drvStrikePrice", ""))
                qty = row.get(qty_col)
                price = row.get(price_col)
                side = row.get("__side", "")
                key = (date_key, symbol, qty)

                if "BUY" in side:
                    if sell_q[key]:
                        # Close short: sell entry first, then buy exit
                        entry = sell_q[key].popleft()
                        entry_dt, entry_price, entry_row = entry
                        trade_counter[date_key] += 1
                        trade_id = trade_counter[date_key]
                        rows.append({
                            "Date": date_str,
                            "Week": '',
                            "unique ID": f"{date_str} | {entry_dt.strftime('%H:%M')} | Short | {trade_id}",
                            "BUY Time": entry_dt.strftime("%H:%M"),
                            "SELL Time": dt.strftime("%H:%M"),
                            "Strike": symbol,
                            "L/S": "Short",
                            "STR Entry": '',
                            "Trade": trade_id,
                            "Qty": qty,
                            "BUY": entry_price,
                            "SELL": price,
                            "Pt": '',
                            "Invested": (float(qty) * float(price)) if pd.notna(qty) and pd.notna(price) else '',
                            "% Return": '',
                            "C%": '',
                            "C Pt": '',
                            "Dur": '',
                            "Brokerage": row.get('brokerageCharges', ''),
                            "Other Charges": row.get('serviceTax', ''),
                            "Gross P/L": '',
                            "Net P/L": '',
                            "Total FC": '',
                            "_entry_dt": entry_dt,
                        })
                    else:
                        buy_q[key].append((dt, price, row))
                elif "SELL" in side:
                    if buy_q[key]:
                        # Close long: buy entry first, then sell exit
                        entry = buy_q[key].popleft()
                        entry_dt, entry_price, entry_row = entry
                        trade_counter[date_key] += 1
                        trade_id = trade_counter[date_key]
                        rows.append({
                            "Date": date_str,
                            "Week": '',
                            "unique ID": f"{date_str} | {entry_dt.strftime('%H:%M')} | Long | {trade_id}",
                            "BUY Time": entry_dt.strftime("%H:%M"),
                            "SELL Time": dt.strftime("%H:%M"),
                            "Strike": symbol,
                            "L/S": "Long",
                            "STR Entry": '',
                            "Trade": trade_id,
                            "Qty": qty,
                            "BUY": entry_price,
                            "SELL": price,
                            "Pt": '',
                            "Invested": (float(qty) * float(entry_price)) if pd.notna(qty) and pd.notna(entry_price) else '',
                            "% Return": '',
                            "C%": '',
                            "C Pt": '',
                            "Dur": '',
                            "Brokerage": row.get('brokerageCharges', ''),
                            "Other Charges": row.get('serviceTax', ''),
                            "Gross P/L": '',
                            "Net P/L": '',
                            "Total FC": '',
                            "_entry_dt": entry_dt,
                        })
                    else:
                        sell_q[key].append((dt, price, row))

        if not rows:
            return pd.DataFrame(columns=journal_cols)

        journal_df = pd.DataFrame(rows)
        # Order like the sheet: Date -> BUY Time -> Trade
        if "Date" in journal_df.columns and "BUY Time" in journal_df.columns and "Trade" in journal_df.columns:
            date_parsed = pd.to_datetime(journal_df["Date"].astype(str) + " 2026", format="%a %d - %b %Y", errors="coerce")
            buy_time_parsed = pd.to_datetime(journal_df["BUY Time"], format="%H:%M", errors="coerce")
            journal_df = journal_df.assign(_date_sort=date_parsed, _time_sort=buy_time_parsed)
            journal_df = journal_df.sort_values(["_date_sort", "_time_sort", "Trade"]).reset_index(drop=True)
        # Keep _entry_dt for plotting; drop from table later
        return journal_df

    journal_df = _build_journal_df(trades_df)
    if journal_df.empty:
        st.info("No paired trades formed yet. Check if BUY/SELL pairs match by date + symbol + qty.")
    else:
        # Table view without helper columns
        journal_df_table = journal_df[[c for c in journal_cols if c in journal_df.columns]]

    def _apply_journal_calcs(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        # Compute Pt = SELL - BUY (where both are present).
        if "BUY" in df.columns and "SELL" in df.columns:
            buy_num = pd.to_numeric(df["BUY"], errors="coerce")
            sell_num = pd.to_numeric(df["SELL"], errors="coerce")
            df["Pt"] = (sell_num - buy_num).where(sell_num.notna() & buy_num.notna())

        # Charges & P/L formulas (as per provided sheet).
        if {"BUY", "SELL", "Qty"}.issubset(df.columns):
            buy_num = pd.to_numeric(df["BUY"], errors="coerce")
            sell_num = pd.to_numeric(df["SELL"], errors="coerce")
            qty_num = pd.to_numeric(df["Qty"], errors="coerce")

            brokerage = qty_num * 0 + 40
            stt = (sell_num * qty_num) * 0.001
            exch = (buy_num * qty_num + sell_num * qty_num) * 0.0003503
            stamp = (buy_num * qty_num) * 0.00003
            gst = (brokerage + exch + stamp) * 0.18
            other_charges = stt + exch + stamp + gst
            gross_pl = (sell_num - buy_num) * qty_num
            net_pl = gross_pl - (brokerage + other_charges)

            df["Brokerage"] = brokerage.round(2)
            df["Other Charges"] = other_charges.round(2)
            df["Gross P/L"] = gross_pl.round(2)
            df["Net P/L"] = net_pl.round(2)
            df["Total FC"] = (df["Brokerage"] + df["Other Charges"]).round(2)

            # % Return = (Gross P/L / BUY) * 100
            df["% Return"] = ((gross_pl / buy_num) * 100).replace([pd.NA, pd.NaT], 0).round(4)
            # C% cumulative of % Return
            df["C%"] = pd.to_numeric(df["% Return"], errors="coerce").fillna(0).cumsum().round(4)
            # C Pt cumulative of Pt (sheet behavior)
            df["C Pt"] = pd.to_numeric(df["Pt"], errors="coerce").fillna(0).cumsum().round(2)

        # Dur = SELL Time - BUY Time
        if "BUY Time" in df.columns and "SELL Time" in df.columns:
            buy_t = pd.to_datetime(df["BUY Time"], format="%H:%M", errors="coerce")
            sell_t = pd.to_datetime(df["SELL Time"], format="%H:%M", errors="coerce")
            dur = sell_t - buy_t
            mins = (dur.dt.total_seconds() / 60.0).round().astype("Int64")
            df["Dur"] = mins.apply(lambda m: f"{int(m)//60:02d}:{int(m)%60:02d}" if pd.notna(m) else "")
        return df

    journal_df = _apply_journal_calcs(journal_df)
    if not journal_df.empty:
        journal_df_table = journal_df[[c for c in journal_cols if c in journal_df.columns]]

    journal_df_life = None
    if show_lifetime:
        trades_raw_life = st.session_state.get("journal_trades_raw_lifetime", [])
        if trades_raw_life:
            journal_df_life = _build_journal_df(pd.DataFrame(trades_raw_life))
            journal_df_life = _apply_journal_calcs(journal_df_life)

    # ---------- Dashboard ----------
    st.markdown("### Dashboard")

    if journal_df.empty:
        st.info("No journal data for selected range.")
    else:
        tab_daily, tab_range, tab_life = st.tabs(["Daily", "Range", "Lifetime"])

        def _metrics_panel(df, title_label: str):
            if df is None or df.empty:
                st.info("No data available.")
                return

            def _num(col):
                return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([])

            total_trades = len(df)
            total_pt = _num("Pt").sum() if "Pt" in df.columns else 0
            total_net = _num("Net P/L").sum() if "Net P/L" in df.columns else 0
            avg_buy = _num("BUY").mean() if "BUY" in df.columns else 0
            avg_sell = _num("SELL").mean() if "SELL" in df.columns else 0
            avg_inv = _num("Invested").mean() if "Invested" in df.columns else 0
            total_fc = _num("Total FC").sum() if "Total FC" in df.columns else 0
            avg_return = _num("% Return").mean() if "% Return" in df.columns else 0

            conti_cnt = int((df.get("STR Entry", "") == "Conti").sum()) if "STR Entry" in df.columns else 0
            reversal_cnt = int((df.get("STR Entry", "") == "Reversal").sum()) if "STR Entry" in df.columns else 0
            manual_cnt = int((df.get("STR Entry", "") == "Manual").sum()) if "STR Entry" in df.columns else 0
            long_cnt = int((df.get("L/S", "") == "Long").sum()) if "L/S" in df.columns else 0
            short_cnt = int((df.get("L/S", "") == "Short").sum()) if "L/S" in df.columns else 0

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"**{title_label}**")
                st.metric("Total Trades", f"{total_trades}")
                st.metric("Avg Buy", f"{avg_buy:.2f}")
                st.metric("Avg Sell", f"{avg_sell:.2f}")
                st.metric("% Return (Avg)", f"{avg_return:.2f}%")
            with c2:
                st.metric("Total Point", f"{total_pt:.2f}")
                st.metric("Total Net P/L", f"{total_net:,.2f}")
                st.metric("T FC (Tax)", f"{total_fc:,.2f}")
                st.metric("Avg Invested", f"{avg_inv:,.2f}")
            with c3:
                st.metric("Conti Trade", f"{conti_cnt}")
                st.metric("Reversal Trade", f"{reversal_cnt}")
                st.metric("Manual Trade", f"{manual_cnt}")
                st.metric("Long Trade", f"{long_cnt}")
                st.metric("Short Trade", f"{short_cnt}")

        with tab_daily:
            day_options = journal_df["Date"].dropna().unique().tolist() if "Date" in journal_df.columns else []
            selected_day = st.selectbox("Select Day", options=day_options, index=(len(day_options) - 1 if day_options else 0))
            day_df = journal_df[journal_df["Date"] == selected_day] if selected_day else journal_df
            _metrics_panel(day_df, selected_day or "Daily")

            if day_df is not None and not day_df.empty and "Pt" in day_df.columns:
                st.markdown("#### Selected Day Breakdown")
                labels = day_df["unique ID"].astype(str) if "unique ID" in day_df.columns else [str(i + 1) for i in range(len(day_df))]
                trade_points = pd.to_numeric(day_df["Pt"], errors="coerce").fillna(0)
                trade_dur = day_df["Dur"].astype(str) if "Dur" in day_df.columns else ["" for _ in range(len(day_df))]

                d1, d2 = st.columns(2)
                with d1:
                    fig_day = go.Figure()
                    fig_day.add_trace(
                        go.Bar(
                            x=labels,
                            y=trade_points,
                            marker_color=trade_points.apply(lambda v: "#1b8f4e" if v >= 0 else "#d64545"),
                            name="Trade Pt",
                        )
                    )
                    fig_day.update_layout(height=220, margin=dict(l=10, r=10, t=20, b=10), title="Individual Trade Points")
                    st.plotly_chart(fig_day, use_container_width=True, config={"displaylogo": False})

                with d2:
                    def _dur_to_min(val):
                        if not isinstance(val, str) or ":" not in val:
                            return 0
                        try:
                            h, m = val.split(":")
                            return int(h) * 60 + int(m)
                        except Exception:
                            return 0
                    dur_min = [_dur_to_min(v) for v in trade_dur]
                    fig_dur = go.Figure()
                    fig_dur.add_trace(
                        go.Bar(
                            x=labels,
                            y=dur_min,
                            marker_color="#6c8ebf",
                            name="Trade Duration (min)",
                        )
                    )
                    fig_dur.update_layout(height=220, margin=dict(l=10, r=10, t=20, b=10), title="Individual Trade Duration (min)")
                    st.plotly_chart(fig_dur, use_container_width=True, config={"displaylogo": False})

            # ---- Notes & Images (persistent) ----
            with st.expander("Trade Notes & Images (Selected Day)", expanded=False):
                notes_path = Path(__file__).resolve().parent / "journal_notes.json"
                media_dir = Path(__file__).resolve().parent / "journal_media"
                media_dir.mkdir(exist_ok=True)

                notes_data = {}
                if notes_path.exists():
                    try:
                        notes_data = json.loads(notes_path.read_text(encoding="utf-8"))
                    except Exception:
                        notes_data = {}

                key_cols = ["unique ID", "Date", "BUY Time", "SELL Time", "Strike", "Qty"]
                key_cols = [c for c in key_cols if c in day_df.columns]
                if key_cols:
                    day_df = day_df.copy()
                    day_df["_tag_key"] = day_df[key_cols].astype(str).agg("|".join, axis=1)
                else:
                    day_df["_tag_key"] = day_df.index.astype(str)

                all_day_images = []
                for _, row in day_df.iterrows():
                    tkey = row["_tag_key"]
                    entry = notes_data.get(tkey, {"note": "", "images": []})
                    note_key = f"note_{hashlib.sha1(tkey.encode()).hexdigest()}"
                    img_key = f"img_{hashlib.sha1(tkey.encode()).hexdigest()}"

                    st.markdown(f"**Trade:** {row.get('unique ID', '')}")
                    obs_col, up_col = st.columns([3, 2])
                    with obs_col:
                        note_val = st.text_area("Observation", value=entry.get("note", ""), key=note_key, height=80)
                    with up_col:
                        st.markdown(
                            """
                            <style>
                            [data-testid="stFileUploaderDropzone"] {
                                padding: 0.2rem !important;
                                border: 1px dashed #d0d7de !important;
                                background: #fff !important;
                            }
                            [data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
                            [data-testid="stFileUploaderDropzone"] svg { display: none !important; }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )
                        uploads = st.file_uploader(
                            "Upload Images",
                            type=["png", "jpg", "jpeg", "webp"],
                            accept_multiple_files=True,
                            key=img_key,
                            label_visibility="collapsed",
                        )

                    if note_val != entry.get("note", ""):
                        entry["note"] = note_val
                        notes_data[tkey] = entry

                    if uploads:
                        for up in uploads:
                            content = up.read()
                            h = hashlib.sha1(content).hexdigest()[:10]
                            filename = f"{h}_{up.name}"
                            filepath = media_dir / filename
                            if filename not in entry.get("images", []):
                                filepath.write_bytes(content)
                                entry.setdefault("images", []).append(filename)
                        notes_data[tkey] = entry

                    for img in entry.get("images", []):
                        img_path = media_dir / img
                        if img_path.exists():
                            all_day_images.append(str(img_path))

                    st.divider()

                try:
                    notes_path.write_text(json.dumps(notes_data, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    st.warning("Could not save notes/images metadata.")

            # Daily gallery (outside expander)
            if all_day_images:
                st.markdown("#### Daily Image Gallery")
                if "gallery_idx" not in st.session_state:
                    st.session_state["gallery_idx"] = None
                max_idx = len(all_day_images) - 1
                def _img_html(img_path, height_px):
                    ext = Path(img_path).suffix.lower().lstrip(".")
                    mime = "image/png" if ext not in {"jpg", "jpeg", "webp"} else f"image/{ext}"
                    data = base64.b64encode(Path(img_path).read_bytes()).decode("utf-8")
                    return f'<img src="data:{mime};base64,{data}" style="height:{height_px}px; width:100%; object-fit:contain; border-radius:8px;" />'
                st.markdown(
                    """
                    <style>
                    button[aria-label^="open-thumb-"] {
                        margin-top: -96px !important;
                        height: 90px !important;
                        width: 100% !important;
                        opacity: 0 !important;
                        padding: 0 !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                if st.session_state["gallery_idx"] is not None:
                    if st.session_state["gallery_idx"] < 0:
                        st.session_state["gallery_idx"] = 0
                    if st.session_state["gallery_idx"] > max_idx:
                        st.session_state["gallery_idx"] = max_idx

                    nav_left, nav_img, nav_right = st.columns([1, 8, 1])
                    with nav_left:
                        if st.button("◀", disabled=st.session_state["gallery_idx"] <= 0):
                            st.session_state["gallery_idx"] -= 1
                    with nav_img:
                        st.markdown(_img_html(all_day_images[st.session_state["gallery_idx"]], 420), unsafe_allow_html=True)
                    with nav_right:
                        if st.button("▶", disabled=st.session_state["gallery_idx"] >= max_idx):
                            st.session_state["gallery_idx"] += 1

                st.markdown("##### Thumbnails (Click to Open)")
                cols = st.columns(6)
                for idx, img_path in enumerate(all_day_images):
                    with cols[idx % 6]:
                        st.markdown(_img_html(img_path, 90), unsafe_allow_html=True)
                        if st.button(f"open-thumb-{idx}", key=f"open_img_{idx}"):
                            st.session_state["gallery_idx"] = idx

        with tab_range:
            _metrics_panel(journal_df, "Range")

        with tab_life:
            if journal_df_life is None or journal_df_life.empty:
                st.info("Lifetime data not loaded.")
            else:
                if "_entry_dt" in journal_df_life.columns:
                    min_dt = pd.to_datetime(journal_df_life["_entry_dt"], errors="coerce").min()
                    max_dt = pd.to_datetime(journal_df_life["_entry_dt"], errors="coerce").max()
                    if pd.notna(min_dt) and pd.notna(max_dt):
                        st.caption(f"Lifetime range: {min_dt.date()} to {max_dt.date()}")
                    else:
                        life_range = st.session_state.get("journal_lifetime_range")
                        if life_range:
                            st.caption(f"Lifetime range: {life_range[0]} to {life_range[1]}")
                _metrics_panel(journal_df_life, "Lifetime")
    st.markdown("### Journal Table")
    # Tag options (persistent)
    tag_defaults = ["", "Conti", "Reversal", "Manual", "Other"]
    tag_options_path = Path(__file__).resolve().parent / "journal_tag_options.json"
    tag_options = tag_defaults.copy()
    if tag_options_path.exists():
        try:
            import json
            loaded = json.loads(tag_options_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                tag_options = tag_defaults + [t for t in loaded if t not in tag_defaults]
        except Exception:
            pass

    if "journal_df_table" in locals():
        # Add custom tag
        st.markdown("#### Add Tag")
        new_tag = st.text_input("New Tag", key="journal_new_tag", placeholder="e.g., Breakout")
        if st.button("Add Tag"):
            if new_tag:
                existing = [t for t in tag_options if t]
                if new_tag not in existing:
                    try:
                        import json
                        custom = [t for t in existing if t not in tag_defaults]
                        custom.append(new_tag)
                        tag_options_path.write_text(json.dumps(custom, ensure_ascii=False, indent=2), encoding="utf-8")
                        st.success("Tag added. Reload the page to see it in dropdown.")
                    except Exception:
                        st.error("Could not save tag.")
        # Allow tagging via editable STR Entry column
        editable_df = journal_df_table.copy()
        if "STR Entry" not in editable_df.columns:
            editable_df["STR Entry"] = ""
        # Build a stable key for persistence
        key_cols = ["unique ID", "Date", "BUY Time", "SELL Time", "Strike", "Qty"]
        key_cols = [c for c in key_cols if c in editable_df.columns]
        if key_cols:
            editable_df["_tag_key"] = editable_df[key_cols].astype(str).agg("|".join, axis=1)
        else:
            editable_df["_tag_key"] = editable_df.index.astype(str)

        if tags_path.exists():
            try:
                tags_df = pd.read_csv(tags_path)
                if "_tag_key" in tags_df.columns and "STR Entry" in tags_df.columns:
                    editable_df = editable_df.merge(tags_df[["_tag_key", "STR Entry"]], on="_tag_key", how="left", suffixes=("", "_saved"))
                    editable_df["STR Entry"] = editable_df["STR Entry_saved"].fillna(editable_df["STR Entry"])
                    editable_df = editable_df.drop(columns=["STR Entry_saved"])
            except Exception:
                pass

        display_df = editable_df.drop(columns=["_tag_key"], errors="ignore")
        edited = st.data_editor(
            display_df,
            use_container_width=True,
            height=420,
            key="journal_table_editor",
            column_config={
                "STR Entry": st.column_config.SelectboxColumn(
                    "STR Entry",
                    options=tag_options,
                    required=False,
                )
            },
            disabled=[c for c in display_df.columns if c not in {"STR Entry"}],
        )
        journal_df_table = edited

        # Persist tags
        if "_tag_key" in editable_df.columns:
            save_df = pd.DataFrame({
                "_tag_key": editable_df["_tag_key"],
                "STR Entry": edited["STR Entry"].values if "STR Entry" in edited.columns else ""
            }).drop_duplicates()
            try:
                save_df.to_csv(tags_path, index=False)
            except Exception:
                st.warning("Could not save tags to disk.")
    else:
        st.dataframe(journal_df, use_container_width=True, height=420)
