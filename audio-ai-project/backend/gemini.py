"""
backend/gemini.py

Simple wrapper around the google-generativeai SDK to produce a short,
human-readable explanation for a prediction.

Behavior:
- Reads GEMINI_API_KEY from the environment or .env
- If the SDK or API key is missing/invalid, returns a safe fallback string
  and never raises.
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Try import; don't crash if missing
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # type: ignore
    logger.debug("google.generativeai not available; Gemini disabled.")


# Configure only if genai is present and key exists
if genai is not None and GEMINI_API_KEY:
    try:
        if hasattr(genai, "configure"):
            genai.configure(api_key=GEMINI_API_KEY)
        logger.debug("Gemini configured.")
    except Exception:
        logger.exception("Failed to configure Gemini; disabling Gemini usage.")
        genai = None


def _label_name(label) -> str:
    s = str(label).strip().lower()
    if s in ("0", "normal", "ok", "healthy"):
        return "normal"
    return "faulty"


def _format_confidence(confidence) -> str:
    try:
        c = float(confidence)
        if 0.0 <= c <= 1.0:
            return f"{c * 100:.0f}%"
        return f"{c:.0f}%" if c >= 1.0 else f"{c:.2f}"
    except Exception:
        return str(confidence)


def _local_fallback(label: str, confidence: float) -> str:
    name = _label_name(label)
    pct = _format_confidence(confidence)
    if name == "normal":
        return (
            f"The model classified the audio as normal with confidence {pct}. "
            "The frequency patterns and MFCC features resemble healthy machine operation, showing stable harmonics and no abnormal noise spikes."
        )
    else:
        return (
            f"The model classified the audio as faulty with confidence {pct}. "
            "This often reflects shifted frequency peaks, atypical MFCC patterns, or increased broadband noise that match known fault signatures."
        )


def _extract_text(resp) -> Optional[str]:
    if resp is None:
        return None
    # 1) direct text attribute
    try:
        text = getattr(resp, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:
        pass
    # 2) attributes like 'content', 'output', 'response'
    try:
        for attr in ("content", "output", "response", "result"):
            v = getattr(resp, attr, None)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, dict):
                for key in ("content", "text"):
                    vv = v.get(key)
                    if isinstance(vv, str) and vv.strip():
                        return vv.strip()
    except Exception:
        pass
    # 3) dict-like shapes with candidates/outputs/choices
    try:
        if isinstance(resp, dict):
            for key in ("candidates", "outputs", "choices"):
                items = resp.get(key) or []
                if items and isinstance(items, (list, tuple)):
                    first = items[0] or {}
                    if isinstance(first, dict):
                        for sub in ("content", "text", "message"):
                            val = first.get(sub)
                            if isinstance(val, str) and val.strip():
                                return val.strip()
            for key in ("content", "text"):
                v = resp.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    except Exception:
        pass
    # 4) object-like candidates/outputs
    try:
        for attr in ("candidates", "outputs", "choices"):
            items = getattr(resp, attr, None)
            if items:
                first = items[0] if isinstance(items, (list, tuple)) else None
                if first:
                    if isinstance(first, dict):
                        for sub in ("content", "text", "message"):
                            v = first.get(sub)
                            if isinstance(v, str) and v.strip():
                                return v.strip()
                    else:
                        v = getattr(first, "content", None) or getattr(first, "text", None)
                        if isinstance(v, str) and v.strip():
                            return v.strip()
    except Exception:
        pass
    # 5) fallback to str()
    try:
        s = str(resp)
        return s.strip() if s else None
    except Exception:
        return None


def _first_sentences(text: str, max_sentences: int = 2) -> str:
    if not text:
        return text
    # Normalize and split into sentences conservatively
    s = " ".join(p.strip() for p in text.replace("\r", "\n").splitlines() if p.strip())
    sentences = []
    cur = ""
    for ch in s:
        cur += ch
        if ch in ".!?":
            sentences.append(cur.strip())
            cur = ""
            if len(sentences) >= max_sentences:
                break
    if sentences:
        return " ".join(sentences)
    # as fallback, truncate reasonably
    return s if len(s) <= 300 else s[:300].rsplit(" ", 1)[0] + "..."


def explain_prediction(label: str, confidence: float) -> str:
    """
    Return a concise, user-friendly explanation for the audio prediction.
    Uses Gemini when available; otherwise returns a domain-aware fallback.
    Never raises; always returns a string.
    """
    try:
        fallback = _local_fallback(label, confidence)

        if genai is None or not GEMINI_API_KEY:
            logger.debug("Gemini unavailable or API key missing; using local fallback.")
            return fallback

        name = _label_name(label).upper()
        pct = _format_confidence(confidence)
        prompt = (
            f"An audio classification model labeled this recording as {name} with confidence {pct}. "
            "The model uses MFCC-based features and examines frequency patterns, harmonic stability, and noise characteristics. "
            "In 2 short, non-technical sentences for a machine operator, explain what this label means and why the confidence might be high or low. "
            "Mention concrete audio cues such as shifted frequency peaks, irregular harmonics, periodic impulses, or elevated broadband noise, and keep the language practical."
        )

        # Prefer GenerativeModel("gemini-pro").generate_content(prompt)
        try:
            GM = getattr(genai, "GenerativeModel", None)
            if GM:
                try:
                    gm = GM("gemini-pro")
                except TypeError:
                    try:
                        gm = GM(model="gemini-pro")
                    except Exception:
                        gm = GM("gemini-pro")
                # Try several call shapes defensively
                for kw in ({"prompt": prompt}, {"input": prompt}, {"messages": [{"role": "user", "content": prompt}]}, {"content": prompt}):
                    try:
                        resp = gm.generate_content(**kw)
                        text = _extract_text(resp)
                        if text:
                            return _first_sentences(text, max_sentences=2)
                    except TypeError:
                        continue
                    except Exception:
                        logger.debug("GenerativeModel.generate_content attempt failed; trying other patterns.")
                        continue
        except Exception:
            logger.debug("GenerativeModel pattern not supported or failed.")

        # Generic genai fallbacks
        try:
            if hasattr(genai, "generate"):
                try:
                    resp = genai.generate(model="gemini-pro", input=prompt)
                except TypeError:
                    resp = genai.generate(model="gemini-pro", prompt=prompt)
                text = _extract_text(resp)
                if text:
                    return _first_sentences(text, max_sentences=2)
            if hasattr(genai, "generate_text"):
                try:
                    resp = genai.generate_text(model="gemini-pro", prompt=prompt)
                    text = _extract_text(resp)
                    if text:
                        return _first_sentences(text, max_sentences=2)
                except Exception:
                    pass
            if hasattr(genai, "chat"):
                try:
                    resp = genai.chat(model="gemini-pro", messages=[{"role": "user", "content": prompt}])
                    text = _extract_text(resp)
                    if text:
                        return _first_sentences(text, max_sentences=2)
                except Exception:
                    pass
        except Exception:
            logger.debug("Generic genai calls failed.")

        # No usable result -> fallback
        logger.debug("Gemini returned no usable text; using local fallback.")
        return fallback

    except Exception:
        logger.exception("Unexpected error in explain_prediction; returning fallback.")
        return _local_fallback(label, confidence)