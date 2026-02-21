"""Legacy/alternate Streamlit app for quick multi-day option OHLC view."""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tempfile
import os

# ---------------- CONFIG ----------------
BASE_URL = "https://api.dhan.co/v2/charts/rollingoption"
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJkaGFuIiwicGFydG5lcklkIjoiIiwiZXhwIjoxNzcxNzMzNzUxLCJpYXQiOjE3NzE2NDczNTEsInRva2VuQ29uc3VtZXJUeXBlIjoiU0VMRiIsIndlYmhvb2tVcmwiOiIiLCJkaGFuQ2xpZW50SWQiOiIxMTAxMzEwOTc2In0.bH0tAmQ6i5ovAscFvbX2r9yakvOaD0mdvPY_0Fk-Tdkc87QoHcb_UdlTiH65tt6bhcw6fLSh43kkoMATHrJVvw"

headers = {
    "Content-Type": "application/json",
    "access-token": ACCESS_TOKEN
}

# ---------------- UI ----------------
st.title("Options OHLC Multi-Day Viewer")

from_date = st.date_input("From Date")
to_date = st.date_input("To Date")
interval = st.selectbox("Time Interval (minutes)", ["1", "5", "15", "25", "60"])

if st.button("Load Charts"):

    # API request payload based on selected UI values.
    body = {
        "exchangeSegment": "NSE_FNO",
        "interval": interval,
        "securityId": 13,
        "instrument": "OPTIDX",
        "expiryFlag": "WEEK",
        "expiryCode": 1,
        "strike": "ATM",
        "drvOptionType": "CALL",
        "requiredData": ["open", "high", "low", "close", "volume"],
        "fromDate": str(from_date),
        "toDate": str(to_date)
    }

    response = requests.post(BASE_URL, headers=headers, json=body)
    data = response.json().get("data", {})
    ce_data = data.get("ce", {})

    ts = ce_data.get("timestamp", [])
    opens = ce_data.get("open", [])
    highs = ce_data.get("high", [])
    lows = ce_data.get("low", [])
    closes = ce_data.get("close", [])
    vols = ce_data.get("volume", [])

    df = pd.DataFrame({
        "Date": [datetime.fromtimestamp(t) for t in ts],
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": vols
    })

    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # ---------------- GROUP BY DAY ----------------
    grouped = df.groupby(df.index.date)

    # Temporary PDF file to store one page per trading day.
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = temp_pdf.name
    temp_pdf.close()

    with PdfPages(pdf_path) as pdf:

        for day, day_df in grouped:

            if len(day_df) == 0:
                continue

            # 🔥 IMPROVED CHART SETTINGS (Only change)
            # Chart settings tuned for readability (larger figure and candles).
            fig, axlist = mpf.plot(
                day_df,
                type="candle",
                volume=True,
                title=f"{day} | Interval: {interval} min",
                style="yahoo",
                figsize=(18, 9),   # Bigger figure
                scale_width_adjustment=dict(
                    candle=1.6,    # Thicker candles
                    volume=1.3
                ),
                returnfig=True
            )

            st.pyplot(fig, use_container_width=True)
            pdf.savefig(fig)
            plt.close(fig)

    # ---------------- DOWNLOAD BUTTON ----------------
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Download All Charts as PDF",
            data=f,
            file_name="multi_day_option_charts.pdf",
            mime="application/pdf"
        )

    os.unlink(pdf_path)
