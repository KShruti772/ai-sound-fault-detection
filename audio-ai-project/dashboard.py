"""
Streamlit dashboard for live audio fault detection.

- Reads data from `data/live_predictions.csv` and displays the latest status and history.
- Auto-refreshes every 2 seconds.
"""
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------- Configuration ----------------
PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "data" / "live_predictions.csv"
REFRESH_SECONDS = 2
HISTORY_ROWS = 20
SAMPLE_DEMO_ROWS = 60  # number of demo rows when no live file

# ---------------- Page & Branding ----------------
st.set_page_config(page_title="WhisperNet", layout="wide")
st.markdown(
    """
    <div style="text-align:center; margin-bottom: 0.25rem;">
        <h1 style="margin:0; font-weight:800; letter-spacing: -1px;">WhisperNet</h1>
        <div style="color:#6c757d; font-size:16px;">AI‑Based Machine Fault Detection System</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Helpers ----------------
def load_data():
    """Load CSV if available, otherwise return empty DataFrame."""
    if CSV_PATH.exists():
        try:
            df = pd.read_csv(CSV_PATH)
            # ensure expected columns exist
            expected = ["time", "rms", "status", "cause", "solution"]
            for c in expected:
                if c not in df.columns:
                    df[c] = ""
            # parse time if possible
            try:
                df["time"] = pd.to_datetime(df["time"])
            except Exception:
                df["time"] = pd.to_datetime(df["time"], errors="coerce")
            return df
        except Exception:
            return pd.DataFrame(columns=["time", "rms", "status", "cause", "solution"])
    else:
        return pd.DataFrame(columns=["time", "rms", "status", "cause", "solution"])

def demo_data(n=SAMPLE_DEMO_ROWS):
    """Generate demo data (alternating normal/faulty) for offline presentation."""
    times = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="S")
    rms = np.abs(np.random.normal(loc=0.02, scale=0.01, size=n))
    # simulate occasional faults
    status = []
    cause = []
    solution = []
    for i in range(n):
        if np.random.rand() > 0.9:
            status.append("FAULTY")
            cause.append("High vibration / bearing wear")
            solution.append("Check bearings; lubricate or replace")
            rms[i] = 0.06 + np.random.rand() * 0.1
        else:
            status.append("NORMAL")
            cause.append("")
            solution.append("")
    return pd.DataFrame({"time": times, "rms": rms, "status": status, "cause": cause, "solution": solution})

def system_health_label(df_recent):
    """Return a simple system health label based on recent fault rate."""
    if df_recent.empty:
        return "Unknown", "grey"
    faulty_pct = (df_recent["status"] == "FAULTY").mean()
    if faulty_pct >= 0.5:
        return "Critical", "red"
    if faulty_pct >= 0.2:
        return "Warning", "orange"
    return "Stable", "green"

# ---------------- Main UI Loop ----------------
placeholder = st.empty()

# Allow demo mode toggle (works when CSV missing)
st.sidebar.title("Controls")
demo_mode = st.sidebar.checkbox("Demo mode (no ESP32)", value=not CSV_PATH.exists())
refresh_sec = st.sidebar.number_input("Refresh interval (s)", min_value=1, max_value=10, value=REFRESH_SECONDS)

while True:
    df = load_data()
    if df.empty and demo_mode:
        df = demo_data()

    # Prepare derived values
    latest = df.iloc[-1] if not df.empty else None
    last_updated = latest["time"] if latest is not None and not pd.isna(latest["time"]) else None
    status_str = str(latest["status"]).upper() if latest is not None else "NO DATA"
    rms_latest = latest["rms"] if latest is not None else None

    # Recent slice for charts & health
    df_recent = df.tail(200).copy() if not df.empty else pd.DataFrame(columns=df.columns)

    # Build UI inside placeholder so it can be cleared/redrawn
    with placeholder.container():
        # Top row: status card, RMS metric, last updated, health
        col_status, col_rms, col_health = st.columns([2, 1, 1], gap="large")

        with col_status:
            st.markdown("### Machine Status")
            if status_str == "NORMAL":
                st.markdown(f"<div style='font-size:32px; color:green;'>✅ NORMAL</div>", unsafe_allow_html=True)
            elif status_str == "FAULTY":
                st.markdown(f"<div style='font-size:32px; color:red;'>⚠️ FAULTY</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='font-size:28px; color:gray;'>⏳ No data</div>", unsafe_allow_html=True)

            if last_updated is not None:
                st.caption(f"Last updated: {pd.to_datetime(last_updated).strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.caption("Last updated: —")

        with col_rms:
            st.metric(label="Latest RMS", value=f"{float(rms_latest):.4f}" if rms_latest is not None else "—", delta=None)
            # small sparkline for recent RMS
            if not df_recent.empty:
                st.line_chart(df_recent["rms"].reset_index(drop=True), height=120)

        with col_health:
            st.markdown("### System Health")
            health_label, health_color = system_health_label(df_recent)
            st.markdown(f"<div style='font-size:20px; color:{health_color}; font-weight:600'>{health_label}</div>", unsafe_allow_html=True)
            # show counts
            cnt_normal = int((df_recent["status"] == "NORMAL").sum()) if not df_recent.empty else 0
            cnt_faulty = int((df_recent["status"] == "FAULTY").sum()) if not df_recent.empty else 0
            st.write(f"Normal: {cnt_normal}  •  Faulty: {cnt_faulty}")

        st.markdown("---")

        # Middle row: charts
        c1, c2 = st.columns([2, 1], gap="large")

        with c1:
            st.subheader("RMS Over Time")
            if df_recent.empty:
                st.info("No RMS data available.")
            else:
                chart = alt.Chart(df_recent.reset_index()).mark_line(point=False).encode(
                    x=alt.X("time:T", title="Time"),
                    y=alt.Y("rms:Q", title="RMS"),
                    color=alt.value("#1f77b4")
                ).properties(height=250)
                st.altair_chart(chart, use_container_width=True)

        with c2:
            st.subheader("Status Distribution")
            if df_recent.empty:
                st.info("No status data.")
            else:
                dist = df_recent["status"].value_counts().reindex(["NORMAL", "FAULTY"]).fillna(0)
                dist_df = pd.DataFrame({"status": dist.index, "count": dist.values})
                bar = alt.Chart(dist_df).mark_bar().encode(
                    x=alt.X("status:N"),
                    y=alt.Y("count:Q"),
                    color=alt.Color("status:N", scale=alt.Scale(domain=["NORMAL","FAULTY"], range=["#2ca02c", "#d62728"]))
                )
                st.altair_chart(bar.properties(height=250), use_container_width=True)

        st.markdown("---")

        # Explanation & Actions
        st.subheader("Fault Explanation & Recommended Action")
        if latest is None or status_str == "NO DATA":
            st.info("No prediction available. Start the live predictor to generate data or enable Demo mode.")
        elif status_str == "NORMAL":
            st.success("Machine operating normally. No immediate action required.")
        else:
            # show human-readable explanation and solution
            cause_text = latest.get("cause", "") or "Abnormal acoustic signature detected."
            solution_text = latest.get("solution", "") or "Inspect mechanical components, bearings, and alignment."
            st.markdown(f"**Cause:** {cause_text}")
            st.markdown(f"**Recommended action:** {solution_text}")

        st.markdown("---")
        # History table
        st.subheader("Recent Predictions")
        if df.empty:
            st.write("No history available.")
        else:
            df_table = df.tail(HISTORY_ROWS).copy()
            # format time column for readability
            if "time" in df_table.columns:
                df_table["time"] = pd.to_datetime(df_table["time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(df_table.reset_index(drop=True), use_container_width=True)

        # Footer / quick tips
        st.markdown(
            """
            <div style="color:#6c757d; font-size:12px;">
            Tip: Use <strong>Demo mode</strong> in the sidebar to demo the dashboard without an ESP32 connection.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Wait then redraw
    time.sleep(refresh_sec)
