from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

# ensure backend package is importable when running "python backend/main.py"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend.firebase_db as firebase_db

logger = logging.getLogger(__name__)

# Absolute imports (clean and consistent)
from backend.audio_processing import extract_mfcc_features


def get_demo_model(feature_vector: Any):
    """Create a tiny demo classifier trained on perturbed copies of `feature_vector`."""
    import numpy as np

    try:
        from backend.model import train_classifier
    except Exception as e:
        raise RuntimeError("demo model creation requires backend.model (scikit-learn)") from e

    x0 = np.asarray(feature_vector).ravel()
    X = np.stack([x0 * (1.0 + 0.01 * i) for i in range(20)])
    y = np.array([0 if i < 10 else 1 for i in range(20)])
    model = train_classifier(X, y, save_path=None)
    return model

def _ts_to_datetime(ts) -> datetime:
	"""Robustly convert Firestore timestamp-like values to datetime or return None."""
	if ts is None:
		return None
	try:
		# native datetime
		if isinstance(ts, datetime):
			return ts
		# google.protobuf Timestamp
		if hasattr(ts, "ToDatetime"):
			return ts.ToDatetime()
		# some Firestore timestamp wrappers
		if hasattr(ts, "to_datetime"):
			return ts.to_datetime()
		if hasattr(ts, "to_datetime_string"):  # unlikely but defensive
			s = ts.to_datetime_string()
			return datetime.fromisoformat(s)
	except Exception:
		pass
	try:
		# epoch seconds
		return datetime.fromtimestamp(float(ts))
	except Exception:
		return None


def show_recent_predictions(limit: int = 5) -> None:
	"""Fetch recent prediction documents and print them nicely to the terminal."""
	try:
		client = firebase_db.get_firestore_client()
		if client is None:
			logger.info("Firestore client not available; skipping recent predictions display.")
			return

		# Stream all docs and sort in Python (keeps code simple and robust)
		coll = client.collection("machine_predictions")
		docs = []
		for doc_snap in coll.stream():
			try:
				data = doc_snap.to_dict() or {}
				ts = _ts_to_datetime(data.get("timestamp"))
				docs.append((ts, data))
			except Exception:
				# skip problematic document but continue
				logger.debug("Skipping invalid document while listing recent predictions.", exc_info=True)
				continue

		# sort by timestamp descending (None timestamps go last)
		docs.sort(key=lambda x: (x[0] is None, x[0]), reverse=True)
		recent = docs[:max(0, int(limit))]

		if not recent:
			print("🛈 No recent predictions found in Firestore.")
			return

		print("\n📋 Recent predictions:")
		for ts, data in recent:
			ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "—"
			metadata = data.get("metadata") or {}
			file_name = metadata.get("file_name") or metadata.get("audio_source") or "—"
			pred = data.get("prediction", "—")
			conf_val = data.get("confidence", None)
			try:
				conf_str = f"{float(conf_val) * 100:.0f}%" if conf_val is not None and float(conf_val) <= 1.0 else (f"{float(conf_val):.0f}%" if conf_val is not None else "—")
			except Exception:
				conf_str = str(conf_val or "—")
			expl = data.get("explanation", "—")

			print(f"🕓 {ts_str}  📂 {file_name}  🧾 {pred}  🎯 {conf_str}")
			print(f"   💬 {expl}\n")

	except Exception:
		logger.exception("Failed to fetch or display recent predictions; continuing.")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    # Scan data/normal and data/faulty for first available .wav or .mp3 file
    data_dir = project_root / "data"
    candidates = []
    for sub in ("normal", "faulty"):
        d = data_dir / sub
        if d.exists() and d.is_dir():
            for pattern in ("*.wav", "*.mp3"):
                candidates.extend(sorted(d.glob(pattern)))

    if not candidates:
        print(f"❌ No audio files found in {data_dir}/normal or {data_dir}/faulty")
        if data_dir.exists() and data_dir.is_dir():
            available = [p.relative_to(project_root) for p in sorted(data_dir.rglob("*")) if p.is_file()]
            if available:
                print("Available files under data/:")
                for p in available:
                    print(" -", p)
            else:
                print("No files present under data/. Please add .wav or .mp3 files.")
        else:
            print("No data/ directory found at project root. Create a data/ folder and add audio files.")
        return

    sample_audio = candidates[0]
    print("Using audio file:", sample_audio)

    try:
        features = extract_mfcc_features(str(sample_audio))
    except FileNotFoundError as e:
        print("❌", e)
        return
    except Exception as e:
        print("❌ Error extracting features:", e)
        return

    print("✅ Extracted features shape:", getattr(features, "shape", None))
    try:
        # show a small preview
        print("First 5 values:", features.ravel()[:5].tolist())
    except Exception:
        pass

    # Optional: run the rest of the pipeline if integrations are available.
    try:
        from backend.model import load_model, predict  # type: ignore
    except Exception as e:
        print("Skipping ML/Gemini/Firestore pipeline (backend.model unavailable):", e)
        return

    # Load existing model if present, otherwise create a demo model.
    model_path = project_root / "backend" / "model.joblib"
    try:
        model = None
        try:
            model = load_model(str(model_path))
        except Exception:
            model = get_demo_model(features)
    except Exception as e:
        print("❌ Failed to obtain model:", e)
        return

    # Run prediction
    try:
        import numpy as np

        X = np.asarray(features).reshape(1, -1)
        preds, probs = predict(model, X)
        pred_label = str(preds[0])
        confidence = float(probs[0].max()) if probs is not None else 1.0
    except Exception as e:
        print("❌ Prediction failed:", e)
        return

    print(f"Prediction: {pred_label}  (confidence={confidence:.3f})")

    # Ask Gemini for explanation if available
    try:
        from backend.gemini import explain_prediction  # type: ignore

        try:
            explanation = explain_prediction(pred_label, confidence)
        except Exception as e:
            explanation = f"(explanation error: {e})"
    except Exception:
        explanation = "(gemini not available)"

    print("Explanation:", explanation)

    # Save to Firestore if available
    try:
        from backend.firebase_db import save_prediction  # type: ignore

        try:
            save_prediction(prediction=pred_label, confidence=confidence, explanation=explanation, metadata={"file": str(sample_audio)})
            print("✅ Saved prediction to Firestore")
        except Exception as e:
            print("❌ Failed to save prediction to Firestore:", e)
    except Exception:
        print("Skipping Firestore save (backend.firebase_db not available)")

    # at the end, display recent predictions
    try:
        show_recent_predictions(limit=5)
    except Exception:
        # Fail silently - log already shows errors
        pass


if __name__ == "__main__":
    main()