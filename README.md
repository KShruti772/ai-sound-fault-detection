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
