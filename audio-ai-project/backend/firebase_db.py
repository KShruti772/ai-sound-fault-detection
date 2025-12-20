import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

try:
	# firebase_admin is an optional runtime dependency
	import firebase_admin
	from firebase_admin import credentials, firestore
except Exception:  # pragma: no cover - runtime dependency may be absent
	firebase_admin = None
	credentials = None
	firestore = None


def get_firestore_client(cred_path: Optional[str] = None):
	"""Return a Firestore client or None if unavailable.

	- Reads credential path from `cred_path` or FIREBASE_CREDENTIALS env var.
	- If path provided but file missing, raises FileNotFoundError.
	- Initializes firebase_admin only once.
	- Returns None (and logs) if firebase-admin isn't installed or init fails.
	"""
	if firebase_admin is None:
		logger.warning("firebase-admin package not installed; Firestore disabled.")
		return None

	path_str = cred_path or os.getenv("FIREBASE_CREDENTIALS")
	if path_str:
		key_path = Path(path_str)
		if not key_path.exists():
			raise FileNotFoundError(f"Firebase credential file not found: {key_path}")
		try:
			if not getattr(firebase_admin, "_apps", {}):
				cred = credentials.Certificate(str(key_path))
				firebase_admin.initialize_app(cred)
		except Exception:
			logger.exception("Failed to initialize Firebase Admin SDK; Firestore disabled.")
			return None
	else:
		# Try ADC / application default credentials
		try:
			if not getattr(firebase_admin, "_apps", {}):
				firebase_admin.initialize_app()
		except Exception:
			logger.warning("No FIREBASE_CREDENTIALS provided and ADC failed; Firestore disabled.")
			return None

	try:
		return firestore.client()
	except Exception:
		logger.exception("Failed to obtain Firestore client; Firestore disabled.")
		return None


def save_prediction(
	prediction: str,
	confidence: float,
	explanation: str,
	metadata: Optional[Dict[str, Any]] = None,
	collection: str = "machine_predictions",
	cred_path: Optional[str] = None,
) -> Optional[Tuple[Any, Any]]:
	"""Save a prediction document to Firestore, or return None if unavailable."""
	client = get_firestore_client(cred_path=cred_path)
	if client is None:
		logger.info("Skipping Firestore save: Firestore client not available.")
		return None

	doc = {
		"timestamp": firestore.SERVER_TIMESTAMP,
		"prediction": str(prediction),
		"confidence": float(confidence),
		"explanation": str(explanation),
	}
	if metadata:
		doc["metadata"] = metadata

	try:
		return client.collection(collection).add(doc)
	except Exception:
		logger.exception("Failed to save prediction to Firestore")
		return None
