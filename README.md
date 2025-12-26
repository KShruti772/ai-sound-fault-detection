⚠ This project is an MVP built for hackathon demonstration purposes.

# AI-based Machine Fault Detection using Sound

## 📌 Problem Statement
Machines often show early signs of failure through abnormal sounds.
Existing monitoring systems are expensive and inaccessible for small labs and industries.

## 💡 Our Solution
We propose a low-cost AI system that listens to machine sounds,
classifies them as Normal or Faulty, and uses Gemini to explain
the possible fault and corrective action.

## ⚙️ System Architecture
Machine Sound → Microphone → ESP32 → Audio Processing →
ML Classifier → Gemini API → Result Storage (Firebase)

## 🛠️ Technologies Used
- ESP32 + Microphone (Sound capture)
- Python + Librosa (Audio feature extraction)
- ML Classifier (Sound classification)
- Google Gemini API (Explanation)
- Firebase Firestore (Result storage)

## 🚀 MVP Scope
- Audio-based binary classification (Normal / Faulty)
- Cloud-based AI processing
- Proof-of-concept (not industrial scale)

## 📊 Sample Output
- Prediction: Faulty
- Confidence: 87%
- Gemini Explanation: Possible bearing misalignment detected

## 👥 Team Members
- Member 1: ESP32 + Hardware
- Member 2: Data & Audio Processing
- Member 3: ML Model
- Member 4: Gemini + Firebase

## 🔮 Future Scope
- Edge AI deployment
- Multi-machine monitoring
- Real-time alerts

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