"""
Train a RandomForestClassifier on MFCC features.

Usage (from project root):
    python scripts/train_model.py
"""
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    # Resolve paths (script in scripts/, project root is parent)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    features_csv = project_root / "data" / "features" / "features.csv"
    model_dir = project_root / "models"
    model_path = model_dir / "audio_fault_model.pkl"

    print(f"Loading features from: {features_csv}")
    if not features_csv.exists():
        print(f"Features file not found: {features_csv}")
        sys.exit(1)

    # Load data
    df = pd.read_csv(features_csv)
    print(f"Total rows loaded: {len(df)}")

    # Determine feature columns (mfcc_1 .. mfcc_20)
    mfcc_cols = [c for c in df.columns if c.lower().startswith("mfcc_")]
    if not mfcc_cols:
        print("No MFCC feature columns found (expected mfcc_1 ... mfcc_20).")
        sys.exit(1)
    mfcc_cols = sorted(mfcc_cols, key=lambda x: int(x.split("_")[1]))  # ensure order
    print(f"Using feature columns: {mfcc_cols[:3]} ... {mfcc_cols[-3:]} (total {len(mfcc_cols)})")

    # Ensure label column exists (accept 'label' or 'machine_type' per instructions)
    if "label" not in df.columns:
        print("Column 'label' not found in features CSV.")
        sys.exit(1)

    # Encode labels: normal -> 0, faulty -> 1
    label_map = {"normal": 0, "faulty": 1}
    y = df["label"].astype(str).str.lower().map(label_map)
    if y.isnull().any():
        invalid = df["label"][y.isnull()].unique()
        print(f"Found unknown label values: {invalid}. They must be 'normal' or 'faulty'.")
        sys.exit(1)

    X = df[mfcc_cols]

    # Split train/test (80/20)
    print("Splitting data: train=80% test=20% (random_state=42)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Train RandomForestClassifier
    print("Training RandomForestClassifier (n_estimators=200)...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    print("Evaluating on test set...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["normal", "faulty"])
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)

    # Save model
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Saved trained model to: {model_path}")

if __name__ == "__main__":
    main()
