import json
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.agents.persona_utils import (
    async_persona_from_observations,
    persona_from_observations,
)
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices, get_prices_async, prices_to_df, prices_to_df_async
from src.utils.api_key import get_api_key_from_state
from src.utils.async_state import update_analyst_signals_async
from src.utils.progress import progress


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, (float, int)):
            return float(value)
        return float(value)
    except (TypeError, ValueError):
        return default


def _calculate_atr(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = prices_df["high"].astype(float)
    low = prices_df["low"].astype(float)
    close = prices_df["close"].astype(float)
    prev_close = close.shift(1)

    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr.bfill()


PERSONA_NAME = "Aegis"
PERSONA_ROLE = "a stop-loss sentinel guarding the book against runaway squeezes"
PERSONA_BACKSTORY = "Aegis managed short risk through volatile markets and now interprets stop metrics before ordering covers."
PERSONA_INSTRUCTIONS = (
    "Scrutinize loss percentages, ATR bands, and position context before deciding.",
    "Force covers only when losses clearly breach thresholds; otherwise justify trims.",
    "Explain the call in first person, highlighting the decisive metrics.",
)
ALLOWED_SIGNALS = ["force_exit", "trim_position", "within_tolerance", "data_unavailable", "no_short_position"]


##### Stop-Loss Guardian Analyst #####
def stop_loss_guardian_agent(state: AgentState, agent_id: str = "stop_loss_guardian_agent"):
    """Monitors active positions and recommends forced covers when losses breach thresholds."""

    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    portfolio = data["portfolio"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    position_monitor: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Evaluating stop-loss bands")

        position = (portfolio.get("positions") or {}).get(ticker, {})
        short_shares = int(position.get("short", 0) or 0)
        avg_entry = _safe_float(position.get("short_cost_basis", 0.0), 0.0)

        if short_shares <= 0 or avg_entry <= 0:
            position_monitor[ticker] = {
                "signal": "no_short_position",
                "confidence": 0,
                "reasoning": "No active short to monitor",
                "constraints": {},
            }
            continue

        prices = get_prices(ticker=ticker, start_date=start_date, end_date=end_date, api_key=api_key)
        if not prices:
            position_monitor[ticker] = {
                "signal": "data_unavailable",
                "confidence": 50,
                "reasoning": "Missing price data; maintain manual vigilance",
                "constraints": {"block_new_shorts": True},
            }
            continue

        prices_df = prices_to_df(prices)
        if prices_df.empty:
            position_monitor[ticker] = {
                "signal": "data_unavailable",
                "confidence": 50,
                "reasoning": "Missing price data; maintain manual vigilance",
                "constraints": {"block_new_shorts": True},
            }
            continue

        close = prices_df["close"].dropna()
        current_price = _safe_float(close.iloc[-1], avg_entry)
        atr_series = _calculate_atr(prices_df)
        atr_latest = _safe_float(atr_series.iloc[-1], 0.0)
        atr_pct = atr_latest / current_price if current_price > 0 else 0.0

        loss_pct = (current_price - avg_entry) / avg_entry if avg_entry > 0 else 0.0
        loss_pct = float(loss_pct)

        base_threshold = max(0.03, atr_pct * 1.5)
        escalated_threshold = base_threshold * 1.5

        suggested_actions = {
            "full_cover_qty": short_shares if loss_pct >= escalated_threshold else 0,
            "partial_cover_qty": max(1, int(np.ceil(short_shares * 0.5))) if loss_pct >= base_threshold else 0,
            "thresholds": {
                "base": base_threshold,
                "escalated": escalated_threshold,
            },
        }

        observations = {
            "ticker": ticker,
            "short_shares": short_shares,
            "average_entry": avg_entry,
            "current_price": current_price,
            "loss_pct": loss_pct,
            "atr_pct": atr_pct,
            "base_threshold": base_threshold,
            "escalated_threshold": escalated_threshold,
            "suggested_actions": suggested_actions,
        }

        decision = persona_from_observations(
            state=state,
            agent_id=agent_id,
            persona_name=PERSONA_NAME,
            persona_role=PERSONA_ROLE,
            persona_backstory=PERSONA_BACKSTORY,
            allowed_signals=ALLOWED_SIGNALS,
            observations=observations,
            persona_instructions=PERSONA_INSTRUCTIONS,
            default_signal="within_tolerance",
            default_confidence=55.0,
            default_reasoning="Defaulted to within_tolerance after missing persona judgement.",
        )

        position_monitor[ticker] = {
            "signal": decision.signal,
            "confidence": int(np.clip(decision.confidence, 0, 100)),
            "reasoning": decision.reasoning,
            "constraints": decision.constraints or {},
            "metrics": {
                "current_price": current_price,
                "avg_entry": avg_entry,
                "short_shares": short_shares,
                "loss_pct": loss_pct,
                "base_threshold": base_threshold,
                "escalated_threshold": escalated_threshold,
                "atr_pct": atr_pct,
            },
            "meta": {"observations": observations},
        }

    message = HumanMessage(content=json.dumps(position_monitor), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(position_monitor, "Stop-Loss Guardian")

    state["data"].setdefault("analyst_signals", {})[agent_id] = position_monitor
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


async def stop_loss_guardian_agent_async(state: AgentState, agent_id: str = "stop_loss_guardian_agent"):
    """Async variant of the stop-loss guardian analyst."""

    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    portfolio = data["portfolio"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    position_monitor: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Evaluating stop-loss bands")

        position = (portfolio.get("positions") or {}).get(ticker, {})
        short_shares = int(position.get("short", 0) or 0)
        avg_entry = _safe_float(position.get("short_cost_basis", 0.0), 0.0)

        if short_shares <= 0 or avg_entry <= 0:
            position_monitor[ticker] = {
                "signal": "no_short_position",
                "confidence": 0,
                "reasoning": "No active short to monitor",
                "constraints": {},
            }
            continue

        prices = await get_prices_async(ticker=ticker, start_date=start_date, end_date=end_date, api_key=api_key)
        if not prices:
            position_monitor[ticker] = {
                "signal": "data_unavailable",
                "confidence": 50,
                "reasoning": "Missing price data; maintain manual vigilance",
                "constraints": {"block_new_shorts": True},
            }
            continue

        prices_df = await prices_to_df_async(prices)
        if prices_df.empty:
            position_monitor[ticker] = {
                "signal": "data_unavailable",
                "confidence": 50,
                "reasoning": "Missing price data; maintain manual vigilance",
                "constraints": {"block_new_shorts": True},
            }
            continue

        close = prices_df["close"].dropna()
        current_price = _safe_float(close.iloc[-1], avg_entry)
        atr_series = _calculate_atr(prices_df)
        atr_latest = _safe_float(atr_series.iloc[-1], 0.0)
        atr_pct = atr_latest / current_price if current_price > 0 else 0.0

        loss_pct = (current_price - avg_entry) / avg_entry if avg_entry > 0 else 0.0
        loss_pct = float(loss_pct)

        base_threshold = max(0.03, atr_pct * 1.5)
        escalated_threshold = base_threshold * 1.5

        suggested_actions = {
            "full_cover_qty": short_shares if loss_pct >= escalated_threshold else 0,
            "partial_cover_qty": max(1, int(np.ceil(short_shares * 0.5))) if loss_pct >= base_threshold else 0,
            "thresholds": {
                "base": base_threshold,
                "escalated": escalated_threshold,
            },
        }

        observations = {
            "ticker": ticker,
            "short_shares": short_shares,
            "average_entry": avg_entry,
            "current_price": current_price,
            "loss_pct": loss_pct,
            "atr_pct": atr_pct,
            "base_threshold": base_threshold,
            "escalated_threshold": escalated_threshold,
            "suggested_actions": suggested_actions,
        }

        decision = await async_persona_from_observations(
            state=state,
            agent_id=agent_id,
            persona_name=PERSONA_NAME,
            persona_role=PERSONA_ROLE,
            persona_backstory=PERSONA_BACKSTORY,
            allowed_signals=ALLOWED_SIGNALS,
            observations=observations,
            persona_instructions=PERSONA_INSTRUCTIONS,
            default_signal="within_tolerance",
            default_confidence=55.0,
            default_reasoning="Defaulted to within_tolerance after missing persona judgement.",
        )

        position_monitor[ticker] = {
            "signal": decision.signal,
            "confidence": int(np.clip(decision.confidence, 0, 100)),
            "reasoning": decision.reasoning,
            "constraints": decision.constraints or {},
            "metrics": {
                "current_price": current_price,
                "avg_entry": avg_entry,
                "short_shares": short_shares,
                "loss_pct": loss_pct,
                "base_threshold": base_threshold,
                "escalated_threshold": escalated_threshold,
                "atr_pct": atr_pct,
            },
            "meta": {"observations": observations},
        }

    message = HumanMessage(content=json.dumps(position_monitor), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(position_monitor, "Stop-Loss Guardian")

    await update_analyst_signals_async(state, agent_id, position_monitor)
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }
