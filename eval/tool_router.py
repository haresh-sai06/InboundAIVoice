"""Deterministic tool router for offline eval / CI.

Mirrors the sales-agent tool surface in ``agent.py`` without needing LiveKit,
OpenAI, Sarvam, or a SIP trunk. An optional LLM mode asks an OpenAI-compatible
endpoint to pick a tool; if that fails, the heuristic is used.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

TOOLS = ("transfer_call", "end_call", "save_booking_intent", "check_availability", "get_business_hours")

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
# very light name cues
NAME_RE = re.compile(
    r"(?:(?:i(?:'m| am)|my name is|naam\s*(?:hai)?|naan)\s+)([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)"
    r"|([A-Z][a-zA-Z]+)\s*,\s*[A-Za-z0-9._%+\-]+@",
    re.I,
)

TRANSFER_CUES = ("founder", "human", "real person", "agent", "sales team", "transfer", "speak to")
END_CUES = ("bye", "goodbye", "alvida", "poitu varen", "that's all", "bas itna", "hang up", "end call", "talk later")
HOURS_CUES = ("business hours", "office hours", "when are you open", "timing", "timings")
AVAIL_CUES = ("slot", "available", "availability", "book a demo", "schedule", "when works", "open today", "open tomorrow")
BOOK_CUES = ("book", "confirm", "schedule me", "demo book", "set up a", "book pannunga")


@dataclass
class RouteDecision:
    tool: str | None
    args: dict[str, Any]
    confidence: float
    rationale: str


def _extract_email(text: str) -> str | None:
    m = EMAIL_RE.search(text)
    if not m:
        return None
    email = m.group(0)
    # reject obviously incomplete domains like "onlyhalf@"
    if email.endswith("@") or "." not in email.split("@", 1)[-1]:
        return None
    return email


def _extract_name(text: str) -> str | None:
    m = NAME_RE.search(text)
    if not m:
        # "My name is Riya Sharma, email ..."
        m2 = re.search(r"(?:name is|I(?:'m| am)|naan)\s+([A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+)?)", text, re.I)
        if m2:
            return m2.group(1).strip()
        return None
    return (m.group(1) or m.group(2) or "").strip() or None


def _date_hint(text: str) -> str | None:
    low = text.lower()
    if "tomorrow" in low:
        return "tomorrow"
    if "today" in low:
        return "today"
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if m:
        return m.group(1)
    return None


def route_heuristic(text: str) -> RouteDecision:
    low = text.lower()

    if any(c in low for c in TRANSFER_CUES):
        return RouteDecision("transfer_call", {}, 0.9, "transfer cue")

    email = _extract_email(text)
    name = _extract_name(text)
    if email and name and any(c in low for c in BOOK_CUES + ("email", "confirm", "@")):
        return RouteDecision(
            "save_booking_intent",
            {"name": name, "email": email},
            0.9,
            "name+email+booking cue",
        )
    # Tamil / short form: name + email without explicit English "book"
    if email and name:
        return RouteDecision(
            "save_booking_intent",
            {"name": name, "email": email},
            0.75,
            "name+email present",
        )

    if any(c in low for c in HOURS_CUES):
        return RouteDecision("get_business_hours", {}, 0.85, "hours cue")

    if any(c in low for c in AVAIL_CUES):
        args = {}
        hint = _date_hint(text)
        if hint:
            args["date_hint"] = hint
        return RouteDecision("check_availability", args, 0.8, "availability cue")

    if any(c in low for c in END_CUES):
        return RouteDecision("end_call", {}, 0.85, "farewell cue")

    return RouteDecision(None, {}, 0.6, "no tool")


def route_llm(text: str) -> RouteDecision | None:
    base = os.getenv("LLM_BASE_URL")
    key = os.getenv("LLM_API_KEY", "ollama")
    model = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
    if not base:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    client = OpenAI(api_key=key, base_url=base, timeout=30.0)
    prompt = (
        "You are a tool router for a sales voice agent. "
        f"Pick at most one tool from {list(TOOLS)} or null. "
        "Return JSON: {\"tool\": string|null, \"args\": object, \"rationale\": string}.\n"
        f"User: {text}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=200,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        tool = data.get("tool")
        if tool not in TOOLS:
            tool = None
        return RouteDecision(tool, data.get("args") or {}, 0.7, data.get("rationale") or "llm")
    except Exception:
        return None


def route(text: str, mode: str = "heuristic") -> RouteDecision:
    if mode == "llm":
        decided = route_llm(text)
        if decided is not None:
            return decided
    return route_heuristic(text)
