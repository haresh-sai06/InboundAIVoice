"""Rate-limit helper tests — import the function without loading LiveKit plugins."""
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_rate_limit():
    """Load just the rate-limit helpers by exec'ing a stubbed slice of agent.py is fragile;
    re-implement the contract here against a tiny extracted module.
    """
    sys.path.insert(0, str(ROOT))
    # Prefer the extracted module if present
    from eval import agent_pure
    return agent_pure


def test_demo_and_unknown_never_rate_limited():
    m = _load_rate_limit()
    m.reset_rate_limits()
    assert m.is_rate_limited("demo") is False
    assert m.is_rate_limited("unknown") is False


def test_rate_limit_trips_after_window_budget():
    m = _load_rate_limit()
    m.reset_rate_limits()
    phone = "+919999000001"
    for _ in range(m.RATE_LIMIT_CALLS):
        assert m.is_rate_limited(phone) is False
    assert m.is_rate_limited(phone) is True
