# WhisperNet

AI‑Based Machine Fault Detection System

WhisperNet is a hackathon-ready prototype that performs real-time machine fault detection using audio captured from an ESP32. It extracts MFCC features, classifies machine health with a trained Random Forest, and visualizes results on a Streamlit dashboard for intuitive, actionable insights.

---

## Problem Statement

Mechanical faults often manifest as subtle changes in sound long before catastrophic failure. Manual inspection is time-consuming and reactive. Teams need an automated, low-cost, real‑time method to detect developing faults and surface clear corrective actions during operation.

---

## Solution Overview

WhisperNet listens to machine audio via an ESP32-mounted microphone, extracts robust acoustic features (20 MFCCs averaged over a time window), and uses a pre-trained Random Forest classifier to label operation as NORMAL or FAULTY. Results — including RMS, a short explanation of the likely cause, and a suggested corrective action — are written to a simple record file and displayed on a live Streamlit dashboard for operators and judges.

---

## System Architecture

Simple text diagram:

ESP32 (MAX4466 mic) --USB Serial--> live_predict_esp32.py (read & buffer audio)  
 --> feature extraction (MFCC, mean) --> RandomForest model --> data/live_predictions.csv  
 --> Streamlit dashboard (WhisperNet) reads CSV and displays live status & history

---

## Components

### Hardware

- ESP32 microcontroller
- MAX4466 (or similar) electret microphone amplifier
- USB connection for serial audio streaming to the host machine

### Feature Extraction

- Mel-frequency cepstral coefficients (MFCC)
- n_mfcc = 20
- Mean of each coefficient across a fixed time window (1s or configurable)
- n_fft/hop_length chosen to match buffer size for reliable extraction

### Machine Learning Model

- Algorithm: Random Forest Classifier
- Labels: `NORMAL` (0) and `FAULTY` (1)
- Trained offline on MFCC-processed examples from normal and faulty machine states

### Dashboard

- Streamlit application: "WhisperNet"
- Live view: current status (big, colored), latest RMS, cause & solution (for FAULTY), and recent prediction history (last 20)
- Uses a simple CSV file (data/live_predictions.csv) for process-safe communication between the predictor and UI

---

## Dataset

- Collection of raw .wav audio files from machines under normal and faulty conditions
- Preprocessing: channel conversion, consistent sample rate (16 kHz), windowing, MFCC extraction (20 coefficients), mean aggregation
- Split into training/validation/test for model development

---

## Installation & Setup

Run the following commands:

```bash
git clone https://github.com/KShruti772/ai-sound-fault-detection
cd ai-sound-fault-detection
python -m venv venv
pip install -r requirements.txt
```

---

## How to Run

1. Connect the ESP32 (with microphone) to the host via USB and ensure the correct serial port (e.g., COM5 on Windows).
2. Start the live predictor (reads serial audio, extracts features, runs model, appends CSV):
   ```bash
   python live_predict_esp32.py
   ```

   - Edit PORT variable in the script if needed.
3. Open the dashboard in a browser:
   ```bash
   streamlit run dashboard.py
   ```
4. The dashboard auto-refreshes and will show the latest prediction, RMS, and suggested action.

---

## Live Demonstration

- Run the ESP32 streaming firmware and the host predictor as above.
- Present dashboard in full-screen mode for judges.
- Demo points:
  - Show a baseline NORMAL reading (low RMS, green badge).
  - Introduce a mechanical fault or simulated noise; observe FAULTY detection and explanation.
  - Highlight the prediction history and RMS trends.

---

## Use Cases

- Predictive maintenance for rotating machinery (motors, fans, pumps)
- Early detection of bearing wear, misalignment, looseness, or friction
- Low-cost machine health monitoring in small industrial setups
- Demonstration/prototyping platform for audio-based anomaly detection

---

## Hackathon Value

- End-to-end demoable: hardware → model → live dashboard
- Strong storytelling potential: show clear before/after behavior with RMS and explanation
- Lightweight and reproducible: runs on commodity hardware (ESP32 + laptop)
- Judges can interact in real time and observe model decisions and suggested corrective actions

---

## Future Improvements

- Improve robustness: add calibration and automatic gain control (AGC) on the capture side
- Add localization: multi-mic array to find fault source
- Model enhancements: time-series models or ensemble of models for higher sensitivity and lower false positives
- Edge inference: run a lightweight model on the ESP32 (or Raspberry Pi) to reduce host dependency
- Enhanced UI: trend charts, alert thresholds, and email/SMS notifications for critical faults
- Dataset expansion: more machine types, loads, and fault modes to generalize across equipment

---

Thank you for exploring WhisperNet — a compact, practical demonstration of AI + Audio + IoT for real-world predictive maintenance.

## 🧪 Model Training (Initial MVP)
The initial model experimentation and training were done using a Google Colab notebook.
This helped in:
- Dataset exploration
- MFCC feature extraction experiments
- Model selection and validation

📓 Colab Notebook (for reference only):
https://colab.research.google.com/github/KShruti772/ai-sound-fault-detection/blob/main/model_code/audio_fault_detection_mvp.ipynb.ipynb


## 🔄 Current Implementation
The final system has been migrated from Colab to a full Python project
with real-time audio input, trained model loading, and a deployed dashboard.


## 🚀 Live MVP
🔗 Streamlit App: https://ai-sound-fault-detection-xxxxx.streamlit.app  
🔗 GitHub Repo: https://github.com/KShruti772/ai-sound-fault-detection
