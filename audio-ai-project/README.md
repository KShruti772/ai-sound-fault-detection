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
