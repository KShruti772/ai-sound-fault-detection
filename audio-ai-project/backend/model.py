import joblib
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_classifier(X: np.ndarray, y: np.ndarray, save_path: Optional[str] = None) -> RandomForestClassifier:
    """Train a RandomForestClassifier on the provided data.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Label vector of shape (n_samples,).
        save_path: Optional path to save the trained model with joblib.

    Returns:
        The trained RandomForestClassifier instance.
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    if save_path:
        joblib.dump(clf, save_path)
    return clf


def load_model(path: str):
    return joblib.load(path)


def predict(model, X: np.ndarray):
    preds = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
        except Exception:
            probs = None
    return preds, probs
