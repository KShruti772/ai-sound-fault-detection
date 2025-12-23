import time
import sys
from pathlib import Path

import numpy as np
import serial
import soundfile as sf

# Configuration
PORT = "COM3"               # change if needed
BAUD = 115200
SAMPLE_RATE = 16000         # Hz
BUFFER_SIZE = 1024          # samples per buffer (uint16)
BYTES_PER_SAMPLE = 2
TIMEOUT_SECONDS = 5.0       # total read timeout
OUTPUT_WAV = "temp.wav"

def read_full_buffer(ser, bytes_needed, timeout=TIMEOUT_SECONDS):
    """Read exactly bytes_needed bytes from serial, or return None on timeout."""
    data = bytearray()
    start = time.time()
    while len(data) < bytes_needed and (time.time() - start) < timeout:
        chunk = ser.read(bytes_needed - len(data))
        if chunk:
            data.extend(chunk)
        else:
            time.sleep(0.01)
    return bytes(data) if len(data) == bytes_needed else None

def main():
    bytes_needed = BUFFER_SIZE * BYTES_PER_SAMPLE
    print(f"Opening serial port {PORT} at {BAUD} baud...")
    try:
        with serial.Serial(PORT, BAUD, timeout=0.5) as ser:
            # Small delay to let the USB-serial stabilize
            time.sleep(0.5)
            ser.reset_input_buffer()
            print(f"Waiting for {BUFFER_SIZE} uint16 samples ({bytes_needed} bytes)...")

            data = read_full_buffer(ser, bytes_needed, timeout=TIMEOUT_SECONDS)
            if data is None:
                print(f"Error: timed out after {TIMEOUT_SECONDS}s while waiting for {bytes_needed} bytes.")
                sys.exit(1)

            # Convert raw bytes to numpy array of uint16 (assumes little-endian from ESP32)
            samples = np.frombuffer(data, dtype=np.uint16)

            if samples.size != BUFFER_SIZE:
                print(f"Warning: expected {BUFFER_SIZE} samples but got {samples.size} samples.")

            # Normalize uint16 -> float32 in range [-1.0, 1.0]
            # uint16 ranges 0..65535, convert to centered signed range
            audio = (samples.astype(np.float32) - 32768.0) / 32768.0

            # Save to WAV
            out_path = Path(OUTPUT_WAV)
            sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
            print(f"Saved {samples.size} samples to {out_path} (sr={SAMPLE_RATE})")

    except serial.SerialException as e:
        print(f"Serial error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
