"""
Streamlit dashboard for live audio fault detection.

- Reads data from `data/live_predictions.csv` and displays the latest status and history.
- Auto-refreshes every 2 seconds.
"""
import time
from pathlib import Path
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import streamlit.components.v1 as components

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

# New: short pitch and demo hint (helps judges get it in 10s)
st.markdown(
    "<div style='text-align:center; color:#4b5563; margin-bottom:1rem;'>"
    "<strong>Real-time audio-based anomaly detection for rotating machinery.</strong> "
    "Connect the ESP32, run the predictor, and open this dashboard to visualize RMS, predictions and suggested actions."
    "</div>",
    unsafe_allow_html=True,
)

# ---------------- Helpers ----------------
def load_csv_df():
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

# ---------------- session_state initialization ----------------
if "history" not in st.session_state:
    st.session_state.history: List[dict] = []      # persistent in-memory history during session
if "seen_keys" not in st.session_state:
    st.session_state.seen_keys = set()            # set of (time,rms,status) to avoid duplicates
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = not CSV_PATH.exists()

# --- Auto-save / restore configuration ---
RMS_CSV = PROJECT_ROOT / "data" / "rms_log.csv"
RMS_CSV.parent.mkdir(parents=True, exist_ok=True)

def load_rms_csv():
    """Load persisted RMS/prediction CSV (if exists) and return DataFrame."""
    if RMS_CSV.exists():
        try:
            df = pd.read_csv(RMS_CSV)
            # ensure expected columns
            for c in ["time", "rms", "status", "cause", "solution"]:
                if c not in df.columns:
                    df[c] = ""
            try:
                df["time"] = pd.to_datetime(df["time"])
            except Exception:
                df["time"] = pd.to_datetime(df["time"], errors="coerce")
            return df
        except Exception:
            return pd.DataFrame(columns=["time", "rms", "status", "cause", "solution"])
    return pd.DataFrame(columns=["time", "rms", "status", "cause", "solution"])

def save_history_to_rms_csv():
    """Persist session_state.history to RMS_CSV, avoiding duplicates."""
    if not st.session_state.history:
        return
    df = pd.DataFrame(st.session_state.history)
    # normalize time to string for stable storage
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    # drop exact duplicates
    df = df.drop_duplicates(subset=["time", "rms", "status"])
    try:
        df.to_csv(RMS_CSV, index=False)
    except Exception as e:
        # non-fatal: show in app logs
        st.error(f"Failed to save history CSV: {e}")

# --- Restore any existing persisted rows into session_state (once at startup) ---
_restored = False
if RMS_CSV.exists():
    df_rest = load_rms_csv()
    if not df_rest.empty:
        for _, row in df_rest.iterrows():
            key = (str(row.get("time")), str(row.get("rms")), str(row.get("status")))
            if key not in st.session_state.seen_keys:
                rec = {
                    "time": pd.to_datetime(row.get("time")) if not pd.isna(row.get("time")) else datetime.now(),
                    "rms": float(row.get("rms") if row.get("rms") is not None else 0.0),
                    "status": str(row.get("status") or "").upper(),
                    "cause": str(row.get("cause") or ""),
                    "solution": str(row.get("solution") or ""),
                }
                st.session_state.history.append(rec)
                st.session_state.seen_keys.add(key)
                _restored = True
if _restored:
    st.success(f"Restored {len(st.session_state.history)} entries from rms_log.csv")

# Sidebar controls
st.sidebar.title("Controls")
st.session_state.demo_mode = st.sidebar.checkbox("Demo mode (no ESP32)", value=st.session_state.demo_mode)
refresh_sec = st.sidebar.number_input("Refresh interval (s)", min_value=1, max_value=10, value=REFRESH_SECONDS)

# ---------------- Load CSV and update session history ----------------
df_csv = load_csv_df()
if df_csv.empty and st.session_state.demo_mode:
    df_csv = demo_data()

# Append new rows from csv to session_state.history (persist across reruns)
if not df_csv.empty:
    # iterate rows in order and add unseen
    for _, row in df_csv.iterrows():
        key = (str(row.get("time")), str(row.get("rms")), str(row.get("status")))
        if key not in st.session_state.seen_keys:
            rec = {
                "time": pd.to_datetime(row.get("time")) if not pd.isna(row.get("time")) else datetime.now(),
                "rms": float(row.get("rms") if row.get("rms") is not None else 0.0),
                "status": str(row.get("status") or "").upper(),
                "cause": str(row.get("cause") or ""),
                "solution": str(row.get("solution") or ""),
            }
            st.session_state.history.append(rec)
            st.session_state.seen_keys.add(key)
            # Persist immediately
            save_history_to_rms_csv()

# Prepare DataFrame from session history for rendering
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    # ensure time column is datetime
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
else:
    df = pd.DataFrame(columns=["time", "rms", "status", "cause", "solution"])

# ---------------- UI Rendering (single-run) ----------------
latest = df.iloc[-1] if not df.empty else None
last_updated = latest["time"] if latest is not None and not pd.isna(latest["time"]) else None
status_str = str(latest["status"]).upper() if latest is not None else "NO DATA"
rms_latest = latest["rms"] if latest is not None else None
df_recent = df.tail(200).copy() if not df.empty else pd.DataFrame(columns=df.columns)

# Enhanced top status card + KPIs and export controls
# Replace existing top row rendering with the following improved block:
col_status, col_kpis, col_actions = st.columns([2, 1, 1], gap="large")

with col_status:
    # large colored card for immediate visual cue
    if latest is None:
        st.markdown("<div style='padding:18px; background:#f3f4f6; border-radius:8px; text-align:center;'>"
                    "<h2 style='margin:0; color:#6b7280;'>No data</h2>"
                    "<div style='color:#6b7280;'>Start the predictor to stream live results</div>"
                    "</div>", unsafe_allow_html=True)
    else:
        if status_str == "NORMAL":
            bg = "#ecfdf5"; color="#065f46"; emoji="✅"
        else:
            bg = "#fff1f2"; color="#7f1d1d"; emoji="⚠️"
        st.markdown(
            f"<div style='padding:18px; background:{bg}; border-radius:8px; text-align:center;'>"
            f"<div style='font-size:28px; color:{color}; font-weight:700;'>{emoji} {status_str}</div>"
            f"<div style='margin-top:6px; color:{color};'>RMS: <strong>{float(rms_latest):.4f}</strong></div>"
            "</div>",
            unsafe_allow_html=True
        )

with col_kpis:
    # Fault rate and uptime KPI
    if not df_recent.empty:
        fault_rate = (df_recent["status"] == "FAULTY").mean()
        first_time = df["time"].min()
        uptime = ""
        try:
            uptime = str(pd.Timestamp.now() - pd.to_datetime(first_time)).split('.')[0]
        except Exception:
            uptime = "—"
    else:
        fault_rate = 0.0
        uptime = "—"

    st.metric("Fault Rate (recent)", f"{fault_rate*100:.1f}%")
    st.metric("Uptime (since first sample)", uptime)

with col_actions:
    st.markdown("### Controls")
    # export/download
    if not df.empty:
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download history CSV", csv_bytes, file_name="whispernet_history.csv", mime="text/csv")
    # snapshot button to save current snapshot for demo evidence
    if st.button("Save snapshot (demo)"):
        snap_path = PROJECT_ROOT / "data" / f"snapshot_{int(time.time())}.csv"
        try:
            df.to_csv(snap_path, index=False)
            st.success(f"Snapshot saved: {snap_path.name}")
        except Exception as e:
            st.error(f"Failed to save snapshot: {e}")

st.markdown("---")

# How it works and judge checklist (storytelling & credibility)
with st.expander("How it works — quick overview (for judges)"):
    st.markdown(
        """
        1. ESP32 streams raw audio; host program buffers windows and extracts 20 MFCC features.
        2. Pre-trained Random Forest classifies windows as NORMAL or FAULTY.
        3. Dashboard displays latest RMS, human-readable explanation and suggested action.
        """
    )
with st.expander("Judge checklist"):
    st.write("- See NORMAL baseline, then trigger a fault or play a sample → observe FAULTY.")
    st.write("- Check RMS rise and explanation matches audible behavior.")
    st.write("- Use the 'Save snapshot' and 'Download history' controls to capture evidence for judging.")

# Charts and explanation
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
st.subheader("Fault Explanation & Recommended Action")
if latest is None or status_str == "NO DATA":
    st.info("No prediction available. Start the live predictor to generate data or enable Demo mode.")
elif status_str == "NORMAL":
    st.success("Machine operating normally. No immediate action required.")
else:
    cause_text = latest.get("cause", "") or "Abnormal acoustic signature detected."
    solution_text = latest.get("solution", "") or "Inspect mechanical components, bearings, and alignment."
    st.markdown(f"**Cause:** {cause_text}")
    st.markdown(f"**Recommended action:** {solution_text}")

st.markdown("---")
st.subheader("Recent Predictions")
if df.empty:
    st.write("No history available.")
else:
    df_table = df.tail(HISTORY_ROWS).copy()
    if "time" in df_table.columns:
        df_table["time"] = pd.to_datetime(df_table["time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(df_table.reset_index(drop=True), use_container_width=True)

st.caption("Tip: Enable Demo mode to preview the dashboard without a live ESP32 connection.")

# ---------------- Auto-refresh (client-side) ----------------
# Inject a small JS snippet to reload the page every refresh_sec seconds.
# This avoids server-side infinite loops and works on Streamlit Cloud.
components.html(
    f"""
    <script>
        const t = {refresh_sec * 1000};
        setTimeout(() => location.reload(), t);
    </script>
    """,
    height=0,
)

# Optionally expose a manual refresh button
if st.button("Refresh now"):
    st.experimental_rerun()
