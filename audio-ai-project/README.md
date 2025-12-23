# 🔊 WhisperNet  
### AI‑Based Machine Fault Detection System

WhisperNet is an end‑to‑end machine fault detection system that uses **audio signal analysis and machine learning** to detect abnormal machine behavior in real time.  
The system listens to machine sounds, extracts MFCC features, classifies machine health, and visualizes results on a live dashboard.

---

## 🚀 Problem Statement
Mechanical faults often produce abnormal sounds before complete failure.  
Manual monitoring is inefficient and error‑prone.

**WhisperNet solves this by providing automated, real‑time fault detection using sound.**

---

## 🧠 Solution Overview
WhisperNet continuously listens to machine sounds, processes them using the same feature pipeline used during training, and predicts whether the machine is operating normally or faultily.

Key highlights:
- Real‑time audio capture using ESP32
- MFCC‑based feature extraction
- Machine learning classification
- Live dashboard visualization
- AI‑generated fault explanation and corrective suggestions

---

## 🏗️ System Architecture


---

## 🧩 Components

### 1️⃣ Hardware
- ESP32
- MAX4466 microphone
- USB serial communication

### 2️⃣ Feature Extraction
- MFCC (Mel Frequency Cepstral Coefficients)
- 20 MFCC features
- Mean aggregation over time window

### 3️⃣ Machine Learning Model
- Algorithm: **Random Forest Classifier**
- Classes: `NORMAL`, `FAULTY`
- Accuracy: ~94% on test data

### 4️⃣ Dashboard
- Built using Streamlit
- Displays:
  - Live machine status
  - RMS sound level
  - Fault explanation
  - Recommended corrective action
  - Prediction history

---

## 📊 Dataset
- Machine audio recordings
- Normal operation sounds
- Faulty operation sounds
- Audio preprocessed and converted to MFCC features

---

## ⚙️ Installation & Setup

### 🔹 Clone Repository
```bash
git clone https://github.com/KShruti772/ai-sound-fault-detection
cd whispernet


# audio-ai-project

Project scaffold for audio AI MVP.

## Setup (Windows)

1. Create a virtual environment and install dependencies:

```powershell
.\scripts\setup_venv.ps1
```

2. Activate the environment:

PowerShell:

.\venv\Scripts\Activate.ps1

Command Prompt:

venv\Scripts\activate.bat

## Service credentials

1. Create a `.env` file at the project root based on `.env.example` and fill in values for:
   - `FIREBASE_CREDENTIALS` — path to your Firebase service account JSON (e.g. `firebase-key.json`).
   - `GEMINI_API_KEY` — your Google Gemini API key.

2. Place your Firebase service account JSON in the project root (or another path) and set `FIREBASE_CREDENTIALS` accordingly. Example:

```
FIREBASE_CREDENTIALS=firebase-key.json
GEMINI_API_KEY=your_gemini_key_here
```

3. The repository `.gitignore` already ignores `.env` and `firebase-key.json` so credentials won't be committed accidentally.

If Firestore credentials are missing, the app will continue running but Firestore saves will be skipped with a clear message.
