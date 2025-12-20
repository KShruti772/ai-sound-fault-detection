import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# ensure backend package imports work when running this script from project root
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# import helper to get Firestore client
from backend import firebase_db

logger = logging.getLogger(__name__)
st.set_page_config(page_title="Machine Audio Health Monitoring Dashboard", layout="wide")


def _ts_to_datetime(ts) -> Optional[datetime]:
    if ts is None:
        return None
    try:
        if isinstance(ts, datetime):
            return ts
        if hasattr(ts, "ToDatetime"):
            return ts.ToDatetime()
        if hasattr(ts, "to_datetime"):
            return ts.to_datetime()
    except Exception:
        pass
    try:
        return datetime.fromtimestamp(float(ts))
    except Exception:
        return None


def _label_name(label: Any) -> str:
    if label is None:
        return "unknown"
    s = str(label).strip().lower()
    if s in ("0", "normal", "ok", "healthy"):
        return "Normal"
    if s in ("1", "faulty", "fault", "anomaly"):
        return "Faulty"
    return label  # preserve if already descriptive


def _conf_str(confidence: Any) -> str:
    try:
        c = float(confidence)
        if 0.0 <= c <= 1.0:
            return f"{c*100:.0f}%"
        return f"{c:.0f}%"
    except Exception:
        return str(confidence or "—")


def fetch_predictions(limit: int = 200) -> List[Dict[str, Any]]:
    client = firebase_db.get_firestore_client()
    if client is None:
        raise RuntimeError("Firestore client not available (see logs).")

    # try both collection names: "predictions" then fallback to "machine_predictions"
    for coll_name in ("predictions", "machine_predictions"):
        try:
            coll = client.collection(coll_name)
            docs = []
            for doc in coll.stream():
                try:
                    data = doc.to_dict() or {}
                    data["_id"] = doc.id
                    docs.append(data)
                except Exception:
                    logger.debug("Skipping a malformed document", exc_info=True)
            if docs:
                # sort by timestamp desc in Python for robustness
                for d in docs:
                    d["_ts"] = _ts_to_datetime(d.get("timestamp"))
                docs.sort(key=lambda x: (x.get("_ts") is None, x.get("_ts")), reverse=True)
                return docs[:max(0, int(limit))]
        except Exception:
            logger.debug("Failed to read collection %s, trying next.", coll_name, exc_info=True)

    return []


def render_latest(preds: List[Dict[str, Any]]):
    st.header("🔎 Latest Prediction")
    if not preds:
        st.info("No prediction records found in Firestore.")
        return
    latest = preds[0]
    metadata = latest.get("metadata") or {}
    file_name = metadata.get("file") or metadata.get("file_name") or metadata.get("audio_source") or "—"
    pred = _label_name(latest.get("prediction"))
    conf = _conf_str(latest.get("confidence"))
    expl = latest.get("explanation") or latest.get("gemini_explanation") or "No explanation available."
    ts = latest.get("_ts")
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "—"

    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader(f"📂 File: {file_name}")
        st.write(f"**Prediction:** {'🟢' if pred=='Normal' else '🔴'} {pred}")
        st.write(f"**Confidence:** {conf}")
    with col2:
        st.subheader("💬 Gemini explanation")
        st.write(expl)
        st.caption(f"Timestamp: {ts_str}")


def render_history_and_chart(preds: List[Dict[str, Any]]):
    st.header("📚 Prediction History")
    if not preds:
        st.info("No history to show.")
        return
    rows = []
    for d in preds:
        md = d.get("metadata") or {}
        file_name = md.get("file") or md.get("file_name") or md.get("audio_source") or "—"
        label = _label_name(d.get("prediction"))
        conf = _conf_str(d.get("confidence"))
        expl = d.get("explanation") or d.get("gemini_explanation") or ""
        ts = d.get("_ts")
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "—"
        rows.append({"timestamp": ts_str, "file": file_name, "prediction": label, "confidence": conf, "explanation": expl})

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # bar chart counts
    counts = df["prediction"].value_counts().reindex(["Normal", "Faulty"]).fillna(0).astype(int)
    st.header("📊 Normal vs Faulty")
    st.bar_chart(counts)


def main():
    st.title("Machine Audio Health Monitoring Dashboard")
    st.markdown("Read-only dashboard showing the latest predictions stored in Firestore.")
    st.markdown("Run with: `streamlit run dashboard.py`")

    try:
        preds = fetch_predictions(limit=200)
    except Exception as exc:
        logger.exception("Error fetching predictions")
        st.error("Could not fetch predictions from Firestore. Check backend logs and FIREBASE_CREDENTIALS.")
        return

    render_latest(preds)
    render_history_and_chart(preds)


if __name__ == "__main__":
    main()
