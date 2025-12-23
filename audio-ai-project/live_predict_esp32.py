"""
Read uint16 audio chunks from an ESP32 over Serial, save temp.wav,
extract 20 MFCC means, load a trained model and predict NORMAL/FAULTY.
Run with: python live_predict_esp32.py
"""
import os
import time
from pathlib import Path
from collections import deque
import csv

import numpy as np
import serial
import soundfile as sf
import librosa
import pandas as pd
import joblib

# Configuration
PORT = "COM3"              # change if needed
BAUD = 115200
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1024       # samples per chunk (uint16)
BYTES_PER_SAMPLE = 2
BYTES_NEEDED = CHUNK_SAMPLES * BYTES_PER_SAMPLE
SERIAL_TIMEOUT = 10        # serial timeout seconds
CHUNK_TIMEOUT = 5.0        # accumulated timeout to read full chunk
OUTPUT_WAV = "temp.wav"
SLEEP_SEC = 0.05          # short sleep during buffer fill
WINDOW_SEC = 2.0          # collect this many seconds before predicting
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
COOLDOWN_SEC = 1.5        # pause after each prediction
RMS_THRESHOLD = 0.01      # ignore windows with RMS below this (background noise)
SMOOTH_K = 5              # smoothing window size (majority vote)
MODEL_REL_PATH = "models/audio_fault_model.pkl"

# Replace/modify these configuration values and the main loop to use a rolling byte buffer:
REQUIRED_SECONDS = 1.0                # collect this many seconds before predicting
REQUIRED_SAMPLES = int(SAMPLE_RATE * REQUIRED_SECONDS)
REQUIRED_BYTES = REQUIRED_SAMPLES * BYTES_PER_SAMPLE
COOLDOWN_SEC = 2.5                    # cooldown after each prediction (2-3s)
RMS_THRESHOLD = 0.01                  # ignore very low RMS windows

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def read_full_buffer(ser, bytes_needed, timeout=CHUNK_TIMEOUT):
    """Read exactly bytes_needed bytes from serial, or return None on timeout."""
    data = bytearray()
    start = time.time()
    while len(data) < bytes_needed and (time.time() - start) < timeout:
        chunk = ser.read(bytes_needed - len(data))
        if chunk:
            data.extend(chunk)
        else:
            time.sleep(0.005)
    return bytes(data) if len(data) == bytes_needed else None

def normalize_audio_uint16(samples_uint16: np.ndarray) -> np.ndarray:
    """Convert uint16 -> float32, remove DC offset, normalize to [-1,1]."""
    # center around zero
    audio = samples_uint16.astype(np.float32) - 32768.0
    # remove DC offset
    audio = audio - np.mean(audio, dtype=np.float32)
    # normalize to [-1,1]
    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs
    return audio.astype(np.float32)

def extract_mfcc_means_from_wav(wav_path: Path, sr=SAMPLE_RATE) -> np.ndarray:
    """Load wav and return mean MFCCs (20,) using n_fft=1024, hop_length=512."""
    y, _ = librosa.load(str(wav_path), sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=1024, hop_length=512)
    return np.mean(mfcc, axis=1)

def main():
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / MODEL_REL_PATH
    out_wav = script_dir / OUTPUT_WAV

    # Ensure data folder and CSV exist for dashboard communication
    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "live_predictions.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as _f:
            writer = csv.writer(_f)
            writer.writerow(["time", "rms", "status", "cause", "solution"])

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    try:
        model = joblib.load(str(model_path))
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    try:
        # Use non-blocking reads: timeout=0 so read() returns immediately with available bytes.
        with serial.Serial(PORT, BAUD, timeout=0) as ser:
            time.sleep(0.2)
            ser.reset_input_buffer()
            print(f"Connected to {PORT} @ {BAUD}. Listening for audio (collecting {REQUIRED_SECONDS}s windows)...")

            byte_buffer = bytearray()
            predictions_deque = deque(maxlen=SMOOTH_K)

            while True:
                try:
                    # Read whatever bytes are currently available (non-blocking)
                    available = ser.in_waiting
                    if available:
                        chunk = ser.read(available)
                        if chunk:
                            byte_buffer.extend(chunk)
                            print(f"Read {len(chunk)} bytes, buffer size {len(byte_buffer)} bytes")
                    else:
                        # no bytes available right now
                        time.sleep(0.01)

                    # If buffer has at least the required bytes, process one window
                    if len(byte_buffer) >= REQUIRED_BYTES:
                        # take exactly REQUIRED_BYTES from the start
                        window_bytes = bytes(byte_buffer[:REQUIRED_BYTES])

                        # Clear the buffer completely (start fresh) so partial tail is ignored
                        byte_buffer = bytearray()

                        # Convert raw bytes to uint16 samples (little-endian assumed)
                        try:
                            samples = np.frombuffer(window_bytes, dtype=np.uint16)
                        except Exception as e:
                            print(f"Failed to convert bytes -> samples: {e}")
                            continue

                        if samples.size < REQUIRED_SAMPLES:
                            print(f"Insufficient samples after conversion: {samples.size}")
                            continue

                        # Normalize and compute RMS
                        audio = normalize_audio_uint16(samples[:REQUIRED_SAMPLES])
                        rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))

                        # Silence filtering
                        if rms < RMS_THRESHOLD:
                            print(f"Skipped window (low RMS={rms:.6f}), continuing to listen...")
                            continue

                        # Save window to WAV
                        try:
                            sf.write(str(out_wav), audio, SAMPLE_RATE, subtype="PCM_16")
                        except Exception as e:
                            print(f"Failed to write temp WAV: {e}")

                        # Extract MFCCs and predict (unchanged model & params)
                        try:
                            mfcc_means = extract_mfcc_means_from_wav(out_wav, sr=SAMPLE_RATE)
                            cols = [f"mfcc_{i+1}" for i in range(len(mfcc_means))]
                            X_df = pd.DataFrame([mfcc_means.tolist()], columns=cols)
                            pred = model.predict(X_df)
                            label = int(pred[0])
                        except Exception as e:
                            print(f"Prediction error: {e}")
                            # don't spam; wait a bit then continue
                            time.sleep(0.5)
                            continue

                        # smoothing: majority vote from recent predictions
                        predictions_deque.append(label)
                        faulty_votes = sum(predictions_deque)
                        majority_faulty = faulty_votes > (len(predictions_deque) // 2) if len(predictions_deque) > 0 else (label == 1)

                        # Explanation & solution logic (unchanged style)
                        if majority_faulty:
                            status_str = f"{RED}FAULTY{RESET}"
                            if rms > 0.05:
                                explanation = "High vibration intensity detected, indicating possible mechanical looseness or friction."
                                solution = "Inspect bearings/couplings, tighten loose parts, check lubrication."
                            else:
                                explanation = "Irregular frequency patterns detected, indicating abnormal machine behavior."
                                solution = "Inspect alignment and bearings; schedule detailed vibration analysis."
                        else:
                            status_str = f"{GREEN}NORMAL{RESET}"
                            explanation = "Sound pattern is stable and consistent with normal operation."
                            solution = "No immediate action required; continue monitoring."

                        # Print one clear result and details, then cooldown and continue listening
                        print(f"{status_str} (RMS={rms:.6f})")
                        print("Cause:", explanation)
                        print("Suggested action:", solution)

                        # Append prediction to CSV for dashboard consumption
                        try:
                            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f)
                                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                                 f"{rms:.6f}",
                                                 "FAULTY" if majority_faulty else "NORMAL",
                                                 explanation,
                                                 solution])
                        except Exception as e:
                            print(f"Failed to write prediction CSV: {e}")

                        # Cooldown to avoid rapid re-triggering
                        time.sleep(COOLDOWN_SEC)

                except Exception as e:
                    print(f"Main loop error: {e}")
                    # Reset buffer on unexpected errors to avoid corrupted state
                    byte_buffer = bytearray()
                    time.sleep(0.1)

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
