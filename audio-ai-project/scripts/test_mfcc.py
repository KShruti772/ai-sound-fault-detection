import sys
from pathlib import Path

import librosa
import numpy as np

def extract_mfcc(wav_path):
    """
    Load audio at 16 kHz and return mean of 20 MFCC coefficients across time.
    Returns a NumPy array with shape (20,).
    """
    y, sr = librosa.load(str(wav_path), sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = np.mean(mfcc, axis=1)
    return np.asarray(mfcc_means)

if __name__ == "__main__":
    # locate temp.wav in project root (scripts/../temp.wav)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    wav_path = project_root / "temp.wav"

    if not wav_path.exists():
        print(f"File not found: {wav_path}", file=sys.stderr)
        sys.exit(1)

    feats = extract_mfcc(wav_path)
    print("MFCC shape:", feats.shape)
    print("MFCC values:")
    print(feats)
