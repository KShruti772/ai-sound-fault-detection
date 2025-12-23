"""
Predict audio fault label for a single .wav using the trained RandomForest model.

Usage:
    python scripts/predict.py path\to\audio.wav
"""
import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd  # added to construct DataFrame for prediction

N_MFCC = 20

def compute_mfcc_means(wav_path: Path) -> np.ndarray:
    """Load audio and return mean of N_MFCC MFCC coefficients as 1D array."""
    y, sr = librosa.load(str(wav_path), sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    # pad if fewer rows than expected (defensive)
    if mfcc.shape[0] < N_MFCC:
        pad = np.zeros((N_MFCC - mfcc.shape[0], mfcc.shape[1]))
        mfcc = np.vstack([mfcc, pad])
    mfcc_means = np.mean(mfcc, axis=1)
    return mfcc_means

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/predict.py path\\to\\audio.wav")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    if not audio_path.exists() or not audio_path.is_file():
        print(f"Audio file not found: {audio_path}")
        sys.exit(1)

    # Resolve model path (script is in scripts/, project root is parent)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    model_path = project_root / "models" / "audio_fault_model.pkl"

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    try:
        mfcc_means = compute_mfcc_means(audio_path)
    except Exception as e:
        print(f"Failed to process audio: {e}")
        sys.exit(1)

    # Build a pandas DataFrame with exact feature column names used at training
    cols = [f"mfcc_{i+1}" for i in range(N_MFCC)]
    X_df = pd.DataFrame([mfcc_means.tolist()], columns=cols)

    try:
        pred = model.predict(X_df)
    except Exception as e:
        print(f"Model prediction failed: {e}")
        sys.exit(1)

    label = int(pred[0])
    if label == 0:
        print("NORMAL")
    else:
        print("FAULTY")

if __name__ == "__main__":
    main()
