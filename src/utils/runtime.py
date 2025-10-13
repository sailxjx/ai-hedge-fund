from __future__ import annotations

import os


def _normalize_flag(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().lower()


def async_personas_enabled(default: bool = True) -> bool:
    """Return True when async persona execution is enabled via ASYNC_PERSONAS env var (default on)."""
    value = _normalize_flag(os.getenv("ASYNC_PERSONAS"))
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def resolve_int_env(name: str, default: int) -> int:
    """Parse a positive integer environment variable with a fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    return parsed if parsed > 0 else default
