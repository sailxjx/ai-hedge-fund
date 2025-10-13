"""Guardian agent that suppresses short exposure during squeeze conditions."""

from __future__ import annotations

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

EPSILON = 1e-9
PERSONA_NAME = "Nyx"
PERSONA_ROLE = "an anthropomorphic squeeze guardian who shields the book from violent upside reversals"
PERSONA_BACKSTORY = "Nyx once ran a discretionary short book and earned her scars from brutal squeezes; now she blends tape-reading intuition with quantitative cues to guard the fund."
PERSONA_INSTRUCTIONS = (
    "Emphasize velocity, breadth of upside momentum, and volume surges when determining squeeze risk.",
    "Force full covers only when momentum and positioning metrics shout danger; otherwise recommend trims or guidance.",
    "Explain the judgement in first person, citing the most persuasive metrics.",
)
ALLOWED_SIGNALS = [
    "no_data",
    "insufficient_history",
    "no_features",
    "calm",
    "elevated_risk",
    "squeeze_warning",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compute_features(df: pd.DataFrame) -> dict[str, float]:
    close = df["close"].astype(float).dropna()
    if close.empty:
        return {}

    returns = close.pct_change().dropna()
    if returns.empty:
        return {}

    recent_close = float(close.iloc[-1])
    ret_5 = float(close.pct_change(5).iloc[-1]) if len(close) > 5 else 0.0
    ret_10 = float(close.pct_change(10).iloc[-1]) if len(close) > 10 else 0.0
    ret_20 = float(close.pct_change(20).iloc[-1]) if len(close) > 20 else 0.0

    std_5 = float(returns.rolling(window=5).std().iloc[-1]) if len(returns) >= 5 else 0.0
    std_20 = float(returns.rolling(window=20).std().iloc[-1]) if len(returns) >= 20 else 0.0

    velocity_z = ret_5 / max(std_5 * np.sqrt(5), EPSILON)
    acceleration = ret_5 - ret_20

    rolling_high_20 = float(df["high"].rolling(window=20).max().iloc[-1]) if len(df) >= 20 else recent_close
    gap_vs_high = recent_close / max(rolling_high_20, EPSILON) - 1.0

    rolling_ma_20 = float(close.rolling(window=20).mean().iloc[-1]) if len(close) >= 20 else recent_close
    gap_vs_ma = recent_close / max(rolling_ma_20, EPSILON) - 1.0

    volume = df["volume"].astype(float).dropna()
    volume_ratio = 0.0
    if not volume.empty and len(volume) >= 20:
        vol_ma_20 = float(volume.rolling(window=20).mean().iloc[-1])
        latest_volume = float(volume.iloc[-1])
        if vol_ma_20 > 0:
            volume_ratio = latest_volume / vol_ma_20 - 1.0

    range_series = (df["high"] - df["low"]).astype(float)
    range_ratio = 0.0
    if len(range_series.dropna()) >= 5:
        avg_true_range = float(range_series.rolling(window=5).mean().iloc[-1])
        if recent_close > 0:
            range_ratio = avg_true_range / recent_close

    return {
        "close": recent_close,
        "ret_5": ret_5,
        "ret_10": ret_10,
        "ret_20": ret_20,
        "velocity_z": velocity_z,
        "acceleration": acceleration,
        "gap_vs_high": gap_vs_high,
        "gap_vs_ma": gap_vs_ma,
        "volume_ratio": volume_ratio,
        "range_ratio": range_ratio,
        "std_5": std_5,
        "std_20": std_20,
    }


def _prepare_observation(
    *,
    df: pd.DataFrame | None,
    ticker: str,
    current_short: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    notes: list[str] = []
    features: dict[str, float] = {}
    data_status = "missing_prices"

    if df is None:
        notes.append("No price history retrieved for squeeze diagnostics.")
    else:
        if df.empty or len(df) < 15:
            data_status = "insufficient_history"
            notes.append("Need at least 15 observations to evaluate squeeze risk.")
        else:
            features = _compute_features(df)
            if not features:
                data_status = "feature_extraction_failed"
                notes.append("Unable to derive squeeze metrics from the recent window.")
            else:
                data_status = "data_ready"
                notes.append(
                    "Squeeze diagnostics: "
                    f"ret_5 {features['ret_5']:.1%}, "
                    f"velocity_z {features['velocity_z']:.2f}, "
                    f"volume_ratio {features['volume_ratio']:+.2f}, "
                    f"gap_vs_high {features['gap_vs_high']:.1%}."
                )

    observations = {
        "ticker": ticker,
        "current_short_shares": current_short,
        "data_status": data_status,
        "squeeze_features": {key: float(val) for key, val in features.items()} if features else {},
        "diagnostic_notes": notes,
    }

    return observations, features


def short_squeeze_guardian_agent(state: AgentState, agent_id: str = "short_squeeze_guardian_agent"):
    """Detect aggressive upside squeezes and enforce defensive short constraints."""

    data = state["data"]
    tickers = data["tickers"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    portfolio = data.get("portfolio", {})
    positions = portfolio.get("positions") or {}
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    guardian_view: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Evaluating squeeze risk")
        prices = get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
        )

        current_short = int(_safe_float((positions.get(ticker, {}) or {}).get("short", 0), 0.0))
        df = prices_to_df(prices) if prices else None
        observations, features = _prepare_observation(df=df, ticker=ticker, current_short=current_short)
        progress.update_status(agent_id, ticker, f"Observation ready ({observations['data_status']})")

        decision = persona_from_observations(
            state=state,
            agent_id=agent_id,
            persona_name=PERSONA_NAME,
            persona_role=PERSONA_ROLE,
            persona_backstory=PERSONA_BACKSTORY,
            allowed_signals=ALLOWED_SIGNALS,
            observations=observations,
            persona_instructions=PERSONA_INSTRUCTIONS,
            default_signal="calm",
            default_confidence=55.0,
            default_reasoning="Defaulted to calm after observation-only fallback.",
        )

        metrics = dict(observations["squeeze_features"])
        metrics["current_short_shares"] = current_short

        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": int(max(0, min(round(decision.confidence), 100))),
            "reasoning": decision.reasoning,
            "metrics": metrics,
            "constraints": decision.constraints or {},
            "meta": {"observations": observations},
        }

        progress.update_status(agent_id, ticker, f"{payload['signal'].upper()} @ {payload['confidence']}/100")

        guardian_view[ticker] = payload

    message = HumanMessage(content=json.dumps(guardian_view), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(guardian_view, "Short Squeeze Guardian")

    state["data"].setdefault("analyst_signals", {})[agent_id] = guardian_view
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }

async def short_squeeze_guardian_agent_async(state: AgentState, agent_id: str = "short_squeeze_guardian_agent"):
    """Async short squeeze guardian leveraging persona observations."""

    data = state["data"]
    tickers = data["tickers"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    portfolio = data.get("portfolio", {})
    positions = portfolio.get("positions") or {}
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    guardian_view: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Evaluating squeeze risk")
        prices = await get_prices_async(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
        )

        current_short = int(_safe_float((positions.get(ticker, {}) or {}).get("short", 0), 0.0))
        df = await prices_to_df_async(prices) if prices else None
        observations, features = _prepare_observation(df=df, ticker=ticker, current_short=current_short)
        progress.update_status(agent_id, ticker, f"Observation ready ({observations['data_status']})")

        decision = await async_persona_from_observations(
            state=state,
            agent_id=agent_id,
            persona_name=PERSONA_NAME,
            persona_role=PERSONA_ROLE,
            persona_backstory=PERSONA_BACKSTORY,
            allowed_signals=ALLOWED_SIGNALS,
            observations=observations,
            persona_instructions=PERSONA_INSTRUCTIONS,
            default_signal="calm",
            default_confidence=55.0,
            default_reasoning="Defaulted to calm after observation-only fallback.",
        )

        metrics = dict(observations["squeeze_features"])
        metrics["current_short_shares"] = current_short

        confidence = int(max(0, min(round(decision.confidence), 100)))
        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": confidence,
            "reasoning": decision.reasoning,
            "metrics": metrics,
            "constraints": decision.constraints or {},
            "meta": {"observations": observations},
        }

        progress.update_status(agent_id, ticker, f"{payload['signal'].upper()} @ {confidence}/100")

        guardian_view[ticker] = payload

    message = HumanMessage(content=json.dumps(guardian_view), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(guardian_view, "Short Squeeze Guardian")

    await update_analyst_signals_async(state, agent_id, guardian_view)
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }

