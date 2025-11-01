"""Sentinel agent that forces stubborn shorts to cover during upside breakouts."""

from __future__ import annotations

import json
from typing import Any

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
PERSONA_NAME = "Vega"
PERSONA_ROLE = "an anthropomorphic breakout sentinel safeguarding the portfolio from runaway rallies"
PERSONA_BACKSTORY = "A former short seller, Vega now scouts upside breakouts to decide when entrenched shorts must yield to momentum."
PERSONA_INSTRUCTIONS = (
    "Balance the quantitative breakout score with practical portfolio context before issuing the signal.",
    "Demand forced covers only when upside pressure is overwhelming or short inventory is stubborn.",
    "Explain decisions in first person, highlighting the key metrics that drove the call.",
)
ALLOWED_SIGNALS = [
    "no_data",
    "insufficient_history",
    "no_features",
    "balanced",
    "uptrend_defensive",
    "breakout_cover",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compute_features(df: pd.DataFrame) -> dict[str, float]:
    close = df["close"].astype(float).dropna()
    if close.empty or len(close) < 25:
        return {}

    volume = df["volume"].astype(float).dropna()
    high = df["high"].astype(float).dropna()
    low = df["low"].astype(float).dropna()

    latest_close = float(close.iloc[-1])
    ret_3 = float(close.pct_change(3).iloc[-1]) if len(close) > 3 else 0.0
    ret_5 = float(close.pct_change(5).iloc[-1]) if len(close) > 5 else 0.0
    ret_10 = float(close.pct_change(10).iloc[-1]) if len(close) > 10 else 0.0

    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    ema55 = float(close.ewm(span=55, adjust=False).mean().iloc[-1])
    ema21_gap = latest_close / max(ema21, EPSILON) - 1.0 if ema21 else 0.0
    ema_slope = ema21 / max(ema55, EPSILON) - 1.0 if ema55 else 0.0

    rolling_high_20 = float(high.rolling(window=20).max().iloc[-1]) if len(high) >= 20 else latest_close
    breakout_gap = latest_close / max(rolling_high_20, EPSILON) - 1.0

    volume_ratio = 0.0
    if not volume.empty and len(volume) >= 20:
        vol_ma_20 = float(volume.rolling(window=20).mean().iloc[-1])
        latest_volume = float(volume.iloc[-1])
        if vol_ma_20 > 0:
            volume_ratio = latest_volume / vol_ma_20 - 1.0

    tr = (high - low).abs().dropna()
    atr_10 = float(tr.rolling(window=10).mean().iloc[-1]) if len(tr) >= 10 else 0.0
    atr_ratio = atr_10 / max(latest_close, EPSILON)

    return {
        "close": latest_close,
        "ret_3": ret_3,
        "ret_5": ret_5,
        "ret_10": ret_10,
        "ema21_gap": ema21_gap,
        "ema_slope": ema_slope,
        "breakout_gap": breakout_gap,
        "volume_ratio": volume_ratio,
        "atr_ratio": atr_ratio,
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
        notes.append("No price history retrieved for breakout diagnostics.")
    else:
        if df.empty or len(df) < 25:
            data_status = "insufficient_history"
            notes.append("Need at least 25 observations to assess breakout persistence.")
        else:
            features = _compute_features(df)
            if not features:
                data_status = "feature_extraction_failed"
                notes.append("Unable to derive breakout metrics from the recent window.")
            else:
                data_status = "data_ready"
                notes.append(
                    "Breakout diagnostics: "
                    f"5d return {features['ret_5']:.1%}, "
                    f"10d return {features['ret_10']:.1%}, "
                    f"gap vs. 20d high {features['breakout_gap']:.1%}, "
                    f"volume vs. 20d avg {(features['volume_ratio'] + 1):.2f}x."
                )

    observations = {
        "ticker": ticker,
        "current_short_shares": current_short,
        "data_status": data_status,
        "breakout_features": {key: float(val) for key, val in features.items()} if features else {},
        "diagnostic_notes": notes,
    }

    return observations, features


def breakout_cover_sentinel_agent(state: AgentState, agent_id: str = "breakout_cover_sentinel_agent"):
    """Aggressively unwind shorts when price action signals durable upside breakouts."""

    data = state["data"]
    tickers = data["tickers"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    portfolio = data.get("portfolio", {})
    positions = portfolio.get("positions") or {}
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    sentinel_view: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Scanning breakout pressure")
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
            default_signal="balanced",
            default_confidence=55.0,
            default_reasoning="Defaulted to balanced after observation-only fallback.",
        )

        confidence = int(max(0, min(round(decision.confidence), 100)))
        metrics = dict(observations["breakout_features"])
        metrics["current_short_shares"] = current_short

        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": confidence,
            "reasoning": decision.reasoning,
            "metrics": metrics,
            "constraints": decision.constraints or {},
            "meta": {"observations": observations},
        }

        progress.update_status(agent_id, ticker, f"{payload['signal'].upper()} @ {confidence}/100")

        sentinel_view[ticker] = payload

    message = HumanMessage(content=json.dumps(sentinel_view), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(sentinel_view, "Breakout Cover Sentinel")

    state["data"].setdefault("analyst_signals", {})[agent_id] = sentinel_view
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


async def breakout_cover_sentinel_agent_async(state: AgentState, agent_id: str = "breakout_cover_sentinel_agent"):
    """Async breakout sentinel orchestrating persona guidance."""

    data = state["data"]
    tickers = data["tickers"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    portfolio = data.get("portfolio", {})
    positions = portfolio.get("positions") or {}
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    sentinel_view: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Scanning breakout pressure")
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
            default_signal="balanced",
            default_confidence=55.0,
            default_reasoning="Defaulted to balanced after observation-only fallback.",
        )

        confidence = int(max(0, min(round(decision.confidence), 100)))
        metrics = dict(observations["breakout_features"])
        metrics["current_short_shares"] = current_short

        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": confidence,
            "reasoning": decision.reasoning,
            "metrics": metrics,
            "constraints": decision.constraints or {},
            "meta": {"observations": observations},
        }

        progress.update_status(agent_id, ticker, f"{payload['signal'].upper()} @ {confidence}/100")

        sentinel_view[ticker] = payload

    message = HumanMessage(content=json.dumps(sentinel_view), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(sentinel_view, "Breakout Cover Sentinel")

    await update_analyst_signals_async(state, agent_id, sentinel_view)
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }
