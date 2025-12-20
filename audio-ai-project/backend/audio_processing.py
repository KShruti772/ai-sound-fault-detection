from pathlib import Path
from typing import Optional

import numpy as np

try:
    import librosa
except Exception as e:  # pragma: no cover - import-time dependency handling
    raise ImportError("librosa is required for audio processing. Install with 'pip install librosa'") from e


def extract_features(audio_path: str, n_mfcc: int = 13) -> np.ndarray:
    """
    Load an audio file and return mean MFCC features as a 1D NumPy array.

    Args:
        audio_path: Path to the audio file.
        n_mfcc: Number of MFCC coefficients to compute.

    Returns:
        1D NumPy array of shape (n_mfcc,).

    Raises:
        FileNotFoundError: If `audio_path` does not exist.
        RuntimeError: If librosa fails to load or process the audio.
    """
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    try:
        y, sr = librosa.load(str(audio_file), sr=None, mono=True)
    except Exception as e:
        # Provide a clearer hint for common mp3 loading issues
        ext = audio_file.suffix.lower()
        msg = f"Failed to load audio '{audio_file}': {e}"
        if ext == ".mp3":
            msg += (
                " — mp3 support may require an audio backend (ffmpeg/audioread). "
                "On Windows install ffmpeg and ensure it's on PATH, or convert to WAV."
            )
        raise RuntimeError(msg) from e

    if y is None or getattr(y, "size", 0) == 0:
        raise RuntimeError(f"Loaded audio is empty: {audio_file}")

    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    except Exception as e:
        raise RuntimeError(f"Failed to compute MFCCs for '{audio_file}': {e}") from e

    # mfcc shape: (n_mfcc, n_frames) -> take mean over frames -> (n_mfcc,)
    features = np.mean(mfcc, axis=1)
    return np.asarray(features, dtype=float)


def extract_mfcc_features(audio_path: str, n_mfcc: int = 13) -> np.ndarray:
    """Backward-compatible name: returns same as `extract_features`.

    Kept so callers that import `extract_mfcc_features` continue to work.
    """
    return extract_features(audio_path, n_mfcc=n_mfcc)