"""Pure helpers mirrored from agent.py so unit tests do not import livekit."""
from __future__ import annotations

import time
from collections import defaultdict

_call_timestamps: dict = defaultdict(list)
RATE_LIMIT_CALLS = 5
RATE_LIMIT_WINDOW = 3600


def reset_rate_limits() -> None:
    _call_timestamps.clear()


def is_rate_limited(phone: str) -> bool:
    if phone in ("unknown", "demo"):
        return False
    now = time.time()
    _call_timestamps[phone] = [t for t in _call_timestamps[phone] if now - t < RATE_LIMIT_WINDOW]
    if len(_call_timestamps[phone]) >= RATE_LIMIT_CALLS:
        return True
    _call_timestamps[phone].append(now)
    return False


LANG_PRESETS = {
    "multilingual": "Detect the caller's language and reply in it.",
    "en": "Reply only in English.",
    "hi": "Reply only in Hindi.",
    "ta": "Reply only in Tamil.",
}


def get_language_instruction(lang_preset: str) -> str:
    return LANG_PRESETS.get(lang_preset, LANG_PRESETS["multilingual"])


def estimate_cost(duration_s: int, tts_chars: int, llm_rate_per_min: float = 0.06,
                  tts_rate_per_1k: float = 0.02) -> float:
    """Rough USD cost model used in the shutdown hook (mirrors agent.py)."""
    minutes = max(duration_s, 0) / 60.0
    return round(minutes * llm_rate_per_min + (tts_chars / 1000.0) * tts_rate_per_1k, 4)
