from __future__ import annotations

import asyncio
from typing import Any
from weakref import WeakKeyDictionary

from src.graph.state import AgentState

_LOOP_LOCKS: "WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]]" = WeakKeyDictionary()


def _lock_for(slot: str) -> asyncio.Lock:
    """Return an asyncio.Lock scoped to the active event loop and logical slot."""
    loop = asyncio.get_running_loop()
    bucket = _LOOP_LOCKS.get(loop)
    if bucket is None:
        bucket = {}
        _LOOP_LOCKS[loop] = bucket
    lock = bucket.get(slot)
    if lock is None:
        lock = asyncio.Lock()
        bucket[slot] = lock
    return lock


async def update_analyst_signals_async(state: AgentState, agent_id: str, payload: Any) -> None:
    """Safely mutate the shared analyst_signals dictionary."""
    async with _lock_for("analyst_signals"):
        signals = state["data"].setdefault("analyst_signals", {})
        signals[agent_id] = payload


async def update_risk_state_async(state: AgentState, payload: Any) -> None:
    """Safely persist risk manager state for downstream nodes."""
    async with _lock_for("risk_state"):
        state["data"]["risk_manager_state"] = payload
