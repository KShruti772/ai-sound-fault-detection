import argparse
import re
import time
from collections import deque
import joblib
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Real-time RMS -> model prediction over serial")
    p.add_argument("--port", required=True, help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    p.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    p.add_argument("--model", required=True, help="Path to trained sklearn model (.pkl)")
    p.add_argument("--scaler", default=None, help="Optional scaler (.pkl) used for normalization")
    p.add_argument("--normalize", choices=("auto","minmax","zscore","none"), default="auto",
                   help="Fallback normalization when no scaler provided")
    p.add_argument("--window", type=int, default=None,
                   help="Sliding window size (overrides model expected features). If not set uses model n_features_in_.")
    return p.parse_args()

def extract_number(s):
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else None

def load_model_and_scaler(model_path, scaler_path=None):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path else None
    # try to infer expected feature count
    n_feat = None
    if hasattr(model, "n_features_in_"):
        n_feat = int(model.n_features_in_)
    elif hasattr(model, "coef_"):
        try:
            n_feat = int(model.coef_.shape[1])
        except Exception:
            n_feat = None
    return model, scaler, n_feat

def normalize_values(arr, scaler=None, method="auto"):
    X = np.array(arr, dtype=float).reshape(1, -1)
    if scaler is not None:
        return scaler.transform(X)
    if method == "none":
        return X
    if method == "minmax" or (method == "auto" and X.max() <= 1.0):
        # assume input in [0,1] or scale to [0,1] by clipping
        X = np.clip(X, 0.0, 1.0)
        return X
    # fallback z-score using the batch mean/std
    if method == "zscore" or method == "auto":
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-8
        return (X - mean) / std
    return X

def realtime_loop(port, baud, model, scaler, n_features, normalize_method):
    try:
        import serial
    except Exception as e:
        raise RuntimeError("pyserial is required. Install via `pip install pyserial`.") from e

    # determine buffer size
    buf_size = n_features if n_features is not None else 1
    buf = deque([0.0] * buf_size, maxlen=buf_size)

    ser = serial.Serial(port, baud, timeout=1)
    print(f"Connected to {port} @ {baud}. Waiting for RMS values...")

    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            val = extract_number(line)
            if val is None:
                # ignore non-numeric lines
                continue
            buf.append(float(val))

            # build feature vector from buffer (most recent last)
            x_raw = list(buf)
            X = normalize_values(x_raw, scaler=scaler, method=normalize_method)
            # ensure correct shape
            if X.ndim == 1:
                X = X.reshape(1, -1)
            # If model expects more features than buffer, pad with zeros
            if n_features is not None and X.shape[1] < n_features:
                pad = np.zeros((1, n_features - X.shape[1]))
                X = np.concatenate([X, pad], axis=1)

            try:
                pred = model.predict(X)[0]
            except Exception as e:
                print("Prediction error:", e)
                continue

            label = "normal" if int(pred) == 0 else "abnormal"
            out = f"[{time.strftime('%H:%M:%S')}] RMS={val:.6f} -> {label}"
            # show probability if available
            if hasattr(model, "predict_proba"):
                try:
                    prob = model.predict_proba(X)[0]
                    p_conf = prob[int(pred)]
                    out += f" (conf={p_conf:.3f})"
                except Exception:
                    pass
            print(out)
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        ser.close()

def main():
    args = parse_args()
    model, scaler, n_feat = load_model_and_scaler(args.model, args.scaler)
    n_features = args.window if args.window is not None else n_feat if n_feat is not None else 1
    realtime_loop(args.port, args.baud, model, scaler, n_features, args.normalize)

if __name__ == "__main__":
    main()