from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

CALIBRATION_PATH = Path("configs/probability_calibration.json")


@lru_cache(maxsize=1)
def _load_probability_calibration() -> dict[str, Any]:
    """Load calibration payload from disk once and cache it."""
    try:
        path = CALIBRATION_PATH
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def reset_probability_calibration_cache() -> None:
    """Clear cached calibration payload (useful for tests)."""
    _load_probability_calibration.cache_clear()


def _extract_params(candidate: Mapping[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(candidate, Mapping):
        return None
    if "a" not in candidate or "b" not in candidate:
        return None
    try:
        return {"a": float(candidate["a"]), "b": float(candidate["b"])}
    except (TypeError, ValueError):
        return None


def get_platt_parameters(agent: str, ticker: str) -> dict[str, float] | None:
    """Return Platt scaling parameters for an agent/ticker pair, if available."""
    payload = _load_probability_calibration()
    agent_payload = payload.get(agent)
    if not isinstance(agent_payload, Mapping):
        return None

    specific = _extract_params(agent_payload.get(ticker))
    if specific:
        return specific

    default_params = _extract_params(agent_payload.get("default"))
    return default_params


def apply_platt(score: float, params: Mapping[str, float]) -> float:
    """Apply Platt scaling to a raw score using the provided parameters."""
    try:
        a = float(params.get("a", 0.0))
        b = float(params.get("b", 0.0))
    except (TypeError, ValueError):
        a = 0.0
        b = 0.0
    z = max(min(a * score + b, 50.0), -50.0)
    prob = 1.0 / (1.0 + math.exp(-z))
    return float(prob)


__all__ = [
    "CALIBRATION_PATH",
    "apply_platt",
    "get_platt_parameters",
    "reset_probability_calibration_cache",
]
