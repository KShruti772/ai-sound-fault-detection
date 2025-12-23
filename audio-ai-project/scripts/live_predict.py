import time
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

# Configuration
N_MFCC = 20
SAMPLE_RATE = 16000
LOOP_DELAY = 1.0  # seconds between checks

def load_model(model_path: Path):
    """Load trained model from disk."""
    return joblib.load(str(model_path))

def extract_mfcc(wav_path):
    """
    Load audio at 16 kHz and return the mean of 20 MFCC coefficients across time.
    This matches the MFCC extraction used during training: sr=16000, n_mfcc=20.
    """
    y, sr = librosa.load(str(wav_path), sr=16000)
    # Set n_fft=1024 and hop_length=512 to match the 1024-sample buffer and avoid
    # the librosa warning. hop_length=512 provides 50% overlap.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=1024, hop_length=512)
    mfcc_means = np.mean(mfcc, axis=1)
    return mfcc_means

def extract_mfcc_means(wav_path: Path):
    """Load audio and return 1D array of mean MFCCs (length N_MFCC)."""
    y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE)
    # Set n_fft=1024 and hop_length=512 to match the audio buffer length (1024 samples)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=1024, hop_length=512)
    # pad if fewer rows than expected (defensive)
    if mfcc.shape[0] < N_MFCC:
        pad = np.zeros((N_MFCC - mfcc.shape[0], mfcc.shape[1]))
        mfcc = np.vstack([mfcc, pad])
    return np.mean(mfcc, axis=1)

def realtime_audio_loop(read_audio_chunk, model, out_wav=None, sleep_sec=0.1):
    """
    Continuously read audio samples from read_audio_chunk(), save to temp.wav,
    extract MFCCs using extract_mfcc(), predict with model, and print result.
    - read_audio_chunk() should return a NumPy array of samples (integers or floats).
    - sleep_sec prevents CPU spin.
    """
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    out_wav = Path(out_wav) if out_wav else project_root / "temp.wav"

    while True:
        try:
            samples = read_audio_chunk()
            if samples is None:
                # no data available right now
                time.sleep(sleep_sec)
                continue

            audio = np.asarray(samples)

            # Normalize/convert to float32 in range [-1, 1] for saving as PCM_16.
            if np.issubdtype(audio.dtype, np.integer):
                # common case: uint16 from embedded device -> convert to [-1,1]
                if audio.dtype == np.uint16:
                    audio = (audio.astype(np.float32) - 32768.0) / 32768.0
                else:
                    # assume signed int (e.g. int16)
                    audio = audio.astype(np.float32) / 32768.0
            else:
                audio = audio.astype(np.float32)
                audio = np.clip(audio, -1.0, 1.0)

            # Compute RMS of the normalized audio buffer
            rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))

            # Save the chunk to temp.wav at the expected sample rate
            sf.write(str(out_wav), audio, SAMPLE_RATE, subtype="PCM_16")

            # Extract MFCCs (function exists above and matches training)
            mfcc_means = extract_mfcc(out_wav)  # returns shape (20,)

            # Build DataFrame with exact training column names and predict
            cols = [f"mfcc_{i+1}" for i in range(N_MFCC)]
            X_df = pd.DataFrame([mfcc_means.tolist()], columns=cols)
            pred = model.predict(X_df)
            label = int(pred[0])
            status = "NORMAL" if label == 0 else "FAULTY"
            # Print status with RMS value for easier realtime monitoring
            print(f">>> LIVE STATUS: {status} (RMS={rms:.6f})")

        except Exception as e:
            print(f"Realtime loop error: {e}")

        time.sleep(sleep_sec)

def main():
    # Resolve paths (script lives in scripts/, project root is parent)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    model_path = project_root / "models" / "audio_fault_model.pkl"
    temp_wav = project_root / "temp.wav"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Loaded model: {model_path}")
    print("Monitoring for temp.wav updates... (press Ctrl+C to stop)")

    last_mtime = None

    try:
        while True:
            try:
                if not temp_wav.exists():
                    print("temp.wav not found, waiting...")
                    time.sleep(LOOP_DELAY)
                    continue

                mtime = temp_wav.stat().st_mtime
                # Process only when file is new/updated
                if last_mtime is None or mtime != last_mtime:
                    # Extract MFCC means
                    mfcc_means = extract_mfcc_means(temp_wav)
                    # Build DataFrame with exact training column names
                    cols = [f"mfcc_{i+1}" for i in range(N_MFCC)]
                    X_df = pd.DataFrame([mfcc_means.tolist()], columns=cols)

                    # Predict
                    pred = model.predict(X_df)
                    label = int(pred[0])
                    if label == 0:
                        print(">>> LIVE STATUS: NORMAL")
                    else:
                        print(">>> LIVE STATUS: FAULTY")

                    last_mtime = mtime
                # small delay before next check
                time.sleep(LOOP_DELAY)

            except Exception as e:
                print(f"Error processing file: {e}")
                time.sleep(LOOP_DELAY)

    except KeyboardInterrupt:
        print("Live prediction stopped by user.")

if __name__ == "__main__":
    main()
