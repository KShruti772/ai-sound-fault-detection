"""
Streamlit dashboard for live audio fault detection.

- Reads data from `data/live_predictions.csv` and displays the latest status and history.
- Auto-refreshes every 2 seconds.
"""
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------- Configuration ----------------
PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "data" / "live_predictions.csv"
REFRESH_SECONDS = 2
HISTORY_ROWS = 20

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="WhisperNet", layout="wide")
# Centered brand title + subtitle for a clean, professional look
st.markdown(
    """
    <div style="text-align:center; margin-bottom: 0.5rem;">
        <h1 style="margin:0; font-weight:700;">WhisperNet</h1>
        <h4 style="margin:0; color:#6c757d;">AI‑Based Machine Fault Detection System</h4>
    </div>
    """,
    unsafe_allow_html=True,
)

def load_data():
    if CSV_PATH.exists():
        try:
            df = pd.read_csv(CSV_PATH)
            return df
        except Exception:
            return pd.DataFrame(columns=["time", "rms", "status", "cause", "solution"])
    else:
        return pd.DataFrame(columns=["time", "rms", "status", "cause", "solution"])

df = load_data()

if df.empty:
    st.warning("No predictions yet. Start the live predictor to generate data.")
else:
    latest = df.iloc[-1]
    status = str(latest.get("status", "")).upper()
    rms = latest.get("rms", "")
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Current Machine Status")
        if status == "NORMAL":
            st.markdown(f"<h1 style='color:green'>{status}</h1>", unsafe_allow_html=True)
        elif status == "FAULTY":
            st.markdown(f"<h1 style='color:red'>{status}</h1>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h1>{status}</h1>", unsafe_allow_html=True)
        st.write(f"Latest RMS: {rms}")

    with col2:
        st.subheader("Explanation & Suggested Action")
        if status == "FAULTY":
            st.markdown(f"**Detected cause:** {latest.get('cause','Unknown')}")
            st.markdown(f"**Suggested solution:** {latest.get('solution','Perform inspection')}")
        elif status == "NORMAL":
            st.markdown("Machine operating normally.")
        else:
            st.markdown("Waiting for data...")

    st.subheader("Recent Predictions")
    df_display = df.tail(HISTORY_ROWS).copy()
    st.dataframe(df_display.reset_index(drop=True), use_container_width=True)

# Auto-refresh every REFRESH_SECONDS seconds
# Sleep a short time then rerun to refresh UI (Streamlit will re-run script)
time.sleep(REFRESH_SECONDS)
st.experimental_rerun()
