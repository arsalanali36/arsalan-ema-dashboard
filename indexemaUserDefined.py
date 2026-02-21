"""Main Streamlit app for EMA crossover charting, scan, and backtest workflow.

High-level flow:
1) Data & Settings: fetch candles and enrich with indicators/patterns.
2) Charts: visualize day-wise charts and prepare downloadable artifact.
3) Backtest: scan EMA pairs, run strategy, and inspect trade-level details.
"""

from datetime import time
from numbers import Number

import io
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

from ema_dashboard import (
    _default_strategy_settings,
    ACCESS_TOKEN,
    MIN_BODY_SIZE,
    PREV_BODY_MIN_PTS,
    WICK_RATIO,
    build_intraday_df,
    build_tv_style_html_document,
    fetch_dhan_data,
    prepare_day_df,
    run_ema20_exit_backtest,
    run_ema_variation_backtests,
    validate_response,
)
from ema_dashboard.charts import HAS_LWC, make_day_figure, render_lwc_day_chart, render_lwc_trade_chart


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
    if "<PASTE_DHAN_TOKEN_HERE>" in ACCESS_TOKEN:
        st.error("Please paste your real Dhan access token in ACCESS_TOKEN.")
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
            ACCESS_TOKEN,
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


def apply_backtest_pair(fast: int, slow: int):
    """Apply selected EMA pair through pending keys before widgets instantiate."""
    st.session_state["bt_fast_ema_pending"] = int(fast)
    st.session_state["bt_slow_ema_pending"] = int(slow)


def apply_scan_best_pair():
    scan_results = st.session_state.get("ema_scan_results")
    if isinstance(scan_results, pd.DataFrame) and not scan_results.empty:
        apply_backtest_pair(scan_results.iloc[0]["Fast EMA"], scan_results.iloc[0]["Slow EMA"])
    else:
        st.session_state["scan_pair_warning"] = "Pehle EMA Variation Scan run karein."


init_state()

with st.sidebar:
    st.subheader("Data & Settings")
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
    ["Charts", "Backtest"],
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
    # ---------- Page 3: EMA scan + backtest + reporting ----------
    st.subheader("Backtest (EMA Cross + EMA20 Exit)")

    if st.session_state["final_df"] is None or st.session_state["final_df"].empty:
        st.info("Upar Data & Settings panel se data fetch karein.")
    else:
        st.markdown("### EMA Variation Scan")
        # Grid-scan candidate fast/slow EMA combinations.
        sc1, sc2, sc3 = st.columns(3)
        scan_fast_min = sc1.number_input("Fast EMA Min", min_value=2, max_value=100, value=8, step=1)
        scan_fast_max = sc2.number_input("Fast EMA Max", min_value=3, max_value=150, value=15, step=1)
        scan_fast_step = sc3.number_input("Fast EMA Step", min_value=1, max_value=20, value=1, step=1)
        sc4, sc5, sc6 = st.columns(3)
        scan_slow_min = sc4.number_input("Slow EMA Min", min_value=3, max_value=200, value=15, step=1)
        scan_slow_max = sc5.number_input("Slow EMA Max", min_value=4, max_value=250, value=30, step=1)
        scan_slow_step = sc6.number_input("Slow EMA Step", min_value=1, max_value=25, value=1, step=1)

        if st.button("Run EMA Variation Scan"):
            try:
                if scan_fast_min > scan_fast_max or scan_slow_min > scan_slow_max:
                    raise ValueError("Min value max se bada nahi ho sakta.")
                fast_periods = list(range(int(scan_fast_min), int(scan_fast_max) + 1, int(scan_fast_step)))
                slow_periods = list(range(int(scan_slow_min), int(scan_slow_max) + 1, int(scan_slow_step)))
                with st.spinner("EMA combinations scan ho raha hai..."):
                    st.session_state["ema_scan_results"] = run_ema_variation_backtests(
                        st.session_state["final_df"],
                        fast_periods=fast_periods,
                        slow_periods=slow_periods,
                    )
            except Exception as exc:
                st.session_state["ema_scan_results"] = None
                st.error(f"EMA scan failed: {exc}")

        scan_results = st.session_state.get("ema_scan_results")
        if "bt_fast_ema_pending" in st.session_state:
            st.session_state["bt_fast_ema"] = int(st.session_state.pop("bt_fast_ema_pending"))
        if "bt_slow_ema_pending" in st.session_state:
            st.session_state["bt_slow_ema"] = int(st.session_state.pop("bt_slow_ema_pending"))
        scan_pair_warning = st.session_state.pop("scan_pair_warning", "")
        if scan_pair_warning:
            st.warning(scan_pair_warning)
        if isinstance(scan_results, pd.DataFrame) and not scan_results.empty:
            # Show best pair summary and full ranked table.
            best_row = scan_results.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best Pair", f"{int(best_row['Fast EMA'])}/{int(best_row['Slow EMA'])}")
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

            st.markdown("### Heatmap Optimiser")
            # Visual optimizer to pick best pair by a chosen metric.
            metric_options = {
                "Score (Composite)": "Score",
                "Return [%]": "Return [%]",
                "Sharpe Ratio": "Sharpe Ratio",
                "Win Rate [%]": "Win Rate [%]",
                "Equity Final [$]": "Equity Final [$]",
            }
            metric_label = st.selectbox("Optimise by", list(metric_options.keys()), index=0)
            metric_col = metric_options[metric_label]

            heatmap_df = scan_results.pivot(index="Fast EMA", columns="Slow EMA", values=metric_col).sort_index()
            if not heatmap_df.empty:
                hm_best_idx = scan_results[metric_col].astype(float).idxmax()
                hm_best = scan_results.loc[hm_best_idx]
                hm_fig = go.Figure(
                    data=go.Heatmap(
                        z=heatmap_df.values,
                        x=heatmap_df.columns.tolist(),
                        y=heatmap_df.index.tolist(),
                        colorscale="RdYlGn",
                        colorbar=dict(title=metric_col),
                        hovertemplate="Fast EMA: %{y}<br>Slow EMA: %{x}<br>Value: %{z:.2f}<extra></extra>",
                    )
                )
                hm_fig.add_trace(
                    go.Scatter(
                        x=[int(hm_best["Slow EMA"])],
                        y=[int(hm_best["Fast EMA"])],
                        mode="markers+text",
                        marker=dict(symbol="x", size=14, color="#111111", line=dict(width=1, color="#ffffff")),
                        text=["Best"],
                        textposition="top center",
                        showlegend=False,
                        hovertemplate=(
                            f"Best Pair: {int(hm_best['Fast EMA'])}/{int(hm_best['Slow EMA'])}"
                            f"<br>{metric_col}: {float(hm_best[metric_col]):.2f}<extra></extra>"
                        ),
                    )
                )
                hm_fig.update_layout(
                    height=420,
                    margin=dict(l=20, r=20, t=10, b=20),
                    xaxis=dict(title="Slow EMA"),
                    yaxis=dict(title="Fast EMA"),
                )
                st.plotly_chart(hm_fig, use_container_width=True, config={"displaylogo": False})

                if st.button("Use Heatmap Best Pair"):
                    apply_backtest_pair(int(hm_best["Fast EMA"]), int(hm_best["Slow EMA"]))
                    st.success(
                        f"Backtest pair set to {int(hm_best['Fast EMA'])}/{int(hm_best['Slow EMA'])} by {metric_col}."
                    )

        st.markdown("---")
        st.markdown("### Run Backtest with Selected EMA Pair")
        # Single-run backtest controls.
        b1, b2, b3 = st.columns([1, 1, 1.3])
        b1.number_input("Backtest Fast EMA", min_value=2, max_value=100, step=1, key="bt_fast_ema")
        b2.number_input("Backtest Slow EMA", min_value=3, max_value=250, step=1, key="bt_slow_ema")
        b3.button("Use Scan Best Pair", on_click=apply_scan_best_pair)

        if st.button("Run Backtest"):
            try:
                fast = int(st.session_state["bt_fast_ema"])
                slow = int(st.session_state["bt_slow_ema"])
                if fast >= slow:
                    raise ValueError("Backtest Fast EMA hamesha Slow EMA se chhota hona chahiye.")
                _, stats = run_ema20_exit_backtest(st.session_state["final_df"], fast=fast, slow=slow)
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
                f"Applied EMA Pair: {int(st.session_state['bt_fast_ema'])}/{int(st.session_state['bt_slow_ema'])}"
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


