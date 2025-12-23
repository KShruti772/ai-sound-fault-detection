"""
Extract mean MFCC features from .wav files under data/processed/ and save to
data/features/features.csv.

Creates data/features/ if missing. Skips non-audio files and files already
processed (tracked via a small hash DB). Prints progress and a summary.

Run from project root:
    python scripts/extract_features.py
"""
import hashlib
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

# --- Configuration ---
N_MFCC = 20
FEATURES_DIR_NAME = "data/features"
FEATURES_CSV = "features.csv"
HASH_DB = ".processed_hashes.txt"  # stores one SHA256 hash per processed file
VALID_EXT = ".wav"


def compute_sha256(path: Path, chunk_size: int = 8192) -> str:
    """Return SHA256 hex digest for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_known_hashes(hash_path: Path) -> set:
    """Load known hashes from file; return empty set if missing."""
    if not hash_path.exists():
        return set()
    return {line.strip() for line in hash_path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_hash(hash_path: Path, hex_digest: str):
    """Append a single hash to the hash DB file (creates file if needed)."""
    with hash_path.open("a", encoding="utf-8") as f:
        f.write(hex_digest + "\n")


def make_feature_row(mfcc_means: np.ndarray, machine: str, label: str) -> dict:
    """Create a dictionary row mapping mfcc_1..mfcc_N and machine/label."""
    row = {f"mfcc_{i+1}": float(mfcc_means[i]) for i in range(len(mfcc_means))}
    row["machine"] = machine
    row["label"] = label
    return row


def main():
    # Resolve paths (script is in scripts/, project root is parent)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    processed_root = project_root / "data" / "processed"
    features_root = project_root / FEATURES_DIR_NAME
    features_root.mkdir(parents=True, exist_ok=True)

    csv_path = features_root / FEATURES_CSV
    hash_db_path = features_root / HASH_DB

    if not processed_root.exists():
        print(f"Processed data folder not found: {processed_root}")
        sys.exit(1)

    # Load existing processed hashes to avoid duplicate rows across runs
    known_hashes = load_known_hashes(hash_db_path)

    machines = ("fan", "pump")
    labels = ("normal", "faulty")

    rows = []
    total_seen = 0
    total_processed = 0
    total_skipped = 0

    # Walk the four expected folders
    for machine in machines:
        for label in labels:
            folder = processed_root / machine / label
            if not folder.exists():
                # skip missing folders gracefully
                print(f"Skipping missing folder: {folder}")
                continue

            # Recursively find .wav files
            for wav_path in sorted(folder.rglob(f"*{VALID_EXT}")):
                if not wav_path.is_file():
                    continue
                total_seen += 1

                try:
                    file_hash = compute_sha256(wav_path)
                except Exception as e:
                    print(f"Error hashing {wav_path}: {e} — skipping")
                    total_skipped += 1
                    continue

                if file_hash in known_hashes:
                    print(f"Skipped duplicate: {wav_path}")
                    total_skipped += 1
                    continue

                # Load audio and extract MFCCs
                try:
                    y, sr = librosa.load(str(wav_path), sr=None)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
                    if mfcc.shape[0] < N_MFCC:
                        # pad with zeros if unexpected
                        mfcc = np.vstack([mfcc, np.zeros((N_MFCC - mfcc.shape[0], mfcc.shape[1]))])
                    mfcc_means = np.mean(mfcc, axis=1)
                except Exception as e:
                    print(f"Failed to process {wav_path}: {e} — skipping")
                    total_skipped += 1
                    continue

                row = make_feature_row(mfcc_means, machine, label)
                rows.append(row)

                # Persist hash immediately to avoid reprocessing on partial runs
                append_hash(hash_db_path, file_hash)
                known_hashes.add(file_hash)

                total_processed += 1
                print(f"Processed: {wav_path}")

    # If we have new rows, append to CSV (create or append)
    if rows:
        df_new = pd.DataFrame(rows)
        # Ensure columns order mfcc_1..mfcc_N, machine, label
        mfcc_cols = [f"mfcc_{i+1}" for i in range(N_MFCC)]
        cols = mfcc_cols + ["machine", "label"]
        df_new = df_new[cols]

        if csv_path.exists():
            # Append without header
            df_new.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            # Write new CSV with header
            df_new.to_csv(csv_path, mode="w", header=True, index=False)

    # Final summary
    print("Extraction complete.")
    print(f"  Total .wav files discovered: {total_seen}")
    print(f"  New files processed: {total_processed}")
    print(f"  Files skipped (duplicates/errors): {total_skipped}")
    print(f"  Features saved to: {csv_path}")


if __name__ == "__main__":
    main()
