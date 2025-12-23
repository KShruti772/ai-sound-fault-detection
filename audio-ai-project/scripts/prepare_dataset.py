"""
Prepare dataset: traverse raw dB folders, detect machine (fan/pump),
map labels (normal -> normal, abnormal -> faulty), copy .wav files into
data/processed/<machine>/<label>/ with unique names and avoid duplicates
across runs using persisted SHA256 hashes.

Usage (from project root):
    python scripts/prepare_dataset.py
"""

import os
import re
import hashlib
import shutil
from pathlib import Path

# --- Configuration ---
# Number width for filenames, e.g. 0001
INDEX_WIDTH = 4
# metadata filename storing known SHA256 hashes (one per line)
HASH_DB_FILENAME = ".copied_hashes.txt"

def compute_sha256(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def load_known_hashes(hash_db_path: Path) -> set:
    """Load known hashes from metadata file, return set of hex strings."""
    if not hash_db_path.exists():
        return set()
    with hash_db_path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

def append_hash(hash_db_path: Path, hex_digest: str):
    """Append a new hash to the metadata file."""
    with hash_db_path.open("a", encoding="utf-8") as f:
        f.write(hex_digest + "\n")

def find_max_index(processed_root: Path) -> int:
    """Scan processed folder for existing files and return the maximum used index (0 if none)."""
    pattern = re.compile(r"_(\d{" + str(INDEX_WIDTH) + r"})\.wav$", re.IGNORECASE)
    max_idx = 0
    if not processed_root.exists():
        return 0
    for p in processed_root.rglob("*.wav"):
        m = pattern.search(p.name)
        if m:
            try:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                continue
    return max_idx

def ensure_processed_structure(processed_root: Path):
    """Ensure processed/<fan|pump>/<normal|faulty> folders exist."""
    for machine in ("fan", "pump"):
        for lbl in ("normal", "faulty"):
            (processed_root / machine / lbl).mkdir(parents=True, exist_ok=True)

def main():
    # Determine project root: script is in scripts/, so parent is project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    data_root = project_root / "data"
    processed_root = data_root / "processed"
    hash_db_path = processed_root / HASH_DB_FILENAME

    # Prepare processed folders
    ensure_processed_structure(processed_root)

    # Load known hashes to avoid re-copying the same source files across runs
    known_hashes = load_known_hashes(hash_db_path)

    # Find starting index for unique filenames (global across all categories)
    current_max = find_max_index(processed_root)
    next_index = current_max + 1

    copied = 0
    skipped = 0
    total_found = 0

    # Traverse dB folders inside data_root
    if not data_root.exists():
        print(f"Data folder not found: {data_root}")
        return

    for db_folder in sorted(p for p in data_root.iterdir() if p.is_dir()):
        # Skip processed folder itself
        if db_folder.name.lower() == "processed":
            continue

        # Inside each dB folder, find machine folders (fan/pump)
        for machine_dir in sorted(x for x in db_folder.iterdir() if x.is_dir()):
            machine_name = machine_dir.name.lower()
            # Detect machine type: prefer explicit 'fan' or 'pump'
            if "fan" in machine_name:
                machine = "fan"
            elif "pump" in machine_name:
                machine = "pump"
            else:
                # unknown machine folder, skip
                print(f"Skipping unknown machine folder: {machine_dir}")
                continue

            # Enter each id_xx folder
            for id_dir in sorted(x for x in machine_dir.iterdir() if x.is_dir()):
                # Each id_dir contains 'normal' and/or 'abnormal' folders
                for label_dir in sorted(x for x in id_dir.iterdir() if x.is_dir()):
                    label_name = label_dir.name.lower()
                    if label_name == "normal":
                        dest_label = "normal"
                    elif label_name == "abnormal":
                        dest_label = "faulty"
                    else:
                        # skip unrelated folders
                        continue

                    dest_folder = processed_root / machine / dest_label
                    dest_folder.mkdir(parents=True, exist_ok=True)

                    # Process .wav files
                    for wav in sorted(label_dir.glob("*.wav")):
                        total_found += 1
                        try:
                            file_hash = compute_sha256(wav)
                        except Exception as e:
                            print(f"Error hashing {wav}: {e}")
                            skipped += 1
                            continue

                        if file_hash in known_hashes:
                            print(f"Skipped duplicate (already copied): {wav}")
                            skipped += 1
                            continue

                        # Generate unique destination filename
                        while True:
                            name = f"{machine}_{dest_label}_{next_index:0{INDEX_WIDTH}d}.wav"
                            dest_path = dest_folder / name
                            if not dest_path.exists():
                                break
                            # If exists (unexpected), bump index
                            next_index += 1

                        # Copy file
                        try:
                            shutil.copy2(wav, dest_path)
                            append_hash(hash_db_path, file_hash)
                            known_hashes.add(file_hash)
                            print(f"Copied: {wav} -> {dest_path}")
                            copied += 1
                            next_index += 1
                        except Exception as e:
                            print(f"Failed to copy {wav} -> {dest_path}: {e}")
                            skipped += 1

    # Final summary
    print("Summary:")
    print(f"  Total source .wav files found: {total_found}")
    print(f"  Copied: {copied}")
    print(f"  Skipped (duplicates/errors): {skipped}")

if __name__ == "__main__":
    main()
