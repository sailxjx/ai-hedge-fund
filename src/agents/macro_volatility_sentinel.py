"""Sentinel agent monitoring volatility expansion to curb long exposure overrides."""

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
PERSONA_NAME = "Storm"
PERSONA_ROLE = "a volatility warden who reins in long exposure when macro turbulence spikes"
PERSONA_BACKSTORY = "Storm cut their teeth running volatility overlay strategies and now advises the fund on when swelling turbulence should throttle long adds."
PERSONA_INSTRUCTIONS = (
    "Focus on short-term vs medium-term volatility ratios, ATR acceleration, and recent drawdowns to judge whether to cap longs.",
    "Reserve crash alerts for decisive volatility explosions paired with downside damage; otherwise offer calibrated guidance.",
    "Speak in first person and call out the key metrics behind the recommendation.",
)
ALLOWED_SIGNALS = [
    "no_data",
    "insufficient_history",
    "no_features",
    "calm",
    "vol_watch",
    "crash_alert",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compute_features(df: pd.DataFrame) -> dict[str, float]:
    close = df["close"].astype(float).dropna()
    if close.empty or len(close) < 40:
        return {}

    returns = close.pct_change().dropna()
    if returns.empty:
        return {}

    vol_5 = returns.rolling(window=5).std(ddof=0).iloc[-1] * np.sqrt(252.0)
    vol_20 = returns.rolling(window=20).std(ddof=0).iloc[-1] * np.sqrt(252.0)
    vol_60 = returns.rolling(window=60).std(ddof=0).iloc[-1] * np.sqrt(252.0)

    ret_1 = close.pct_change().iloc[-1]
    ret_5 = close.pct_change(5).iloc[-1]
    ret_10 = close.pct_change(10).iloc[-1]

    rolling_high_20 = close.rolling(window=20).max()
    drawdown_20 = float(close.iloc[-1] / max(rolling_high_20.iloc[-1], EPSILON) - 1.0)

    true_range = (df["high"].astype(float) - df["low"].astype(float)).fillna(0.0)
    atr_5 = true_range.rolling(window=5).mean().iloc[-1]
    atr_20 = true_range.rolling(window=20).mean().iloc[-1]
    atr_ratio = atr_5 / max(atr_20, EPSILON) - 1.0 if atr_20 > 0 else 0.0

    downside_trailing = returns.tail(20)
    downside_tail = float(np.percentile(downside_trailing, 10)) if len(downside_trailing) >= 5 else 0.0

    vol_ratio = vol_5 / max(vol_20, EPSILON) - 1.0 if np.isfinite(vol_5) and np.isfinite(vol_20) else 0.0
    vol_trend = vol_5 - vol_20
    vol_accel = vol_5 - vol_60 if np.isfinite(vol_60) else vol_trend

    return {
        "close": float(close.iloc[-1]),
        "ret_1": float(ret_1) if np.isfinite(ret_1) else 0.0,
        "ret_5": float(ret_5) if np.isfinite(ret_5) else 0.0,
        "ret_10": float(ret_10) if np.isfinite(ret_10) else 0.0,
        "vol_5": float(vol_5) if np.isfinite(vol_5) else 0.0,
        "vol_20": float(vol_20) if np.isfinite(vol_20) else 0.0,
        "vol_60": float(vol_60) if np.isfinite(vol_60) else 0.0,
        "vol_ratio": float(vol_ratio) if np.isfinite(vol_ratio) else 0.0,
        "vol_trend": float(vol_trend) if np.isfinite(vol_trend) else 0.0,
        "vol_accel": float(vol_accel) if np.isfinite(vol_accel) else 0.0,
        "drawdown_20": drawdown_20,
        "atr_ratio": float(atr_ratio) if np.isfinite(atr_ratio) else 0.0,
        "downside_tail": downside_tail,
    }


def _prepare_observation(
    *,
    df: pd.DataFrame | None,
    ticker: str,
    existing_long: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    notes: list[str] = []
    features: dict[str, float] = {}
    data_status = "missing_prices"

    if df is None:
        notes.append("No price history retrieved for the requested window.")
    else:
        if df.empty or len(df) < 40:
            data_status = "insufficient_history"
            notes.append("Need at least 40 observations to evaluate volatility changes.")
        else:
            features = _compute_features(df)
            if not features:
                data_status = "feature_extraction_failed"
                notes.append("Unable to derive volatility metrics from the recent window.")
            else:
                data_status = "data_ready"
                notes.append(
                    "Key diagnostics: "
                    f"vol_ratio {features['vol_ratio']:+.2f}, "
                    f"5d return {features['ret_5']:.1%}, "
                    f"20d drawdown {features['drawdown_20']:.1%}, "
                    f"ATR acceleration {features['atr_ratio']:.3f}."
                )

    observations = {
        "ticker": ticker,
        "existing_long_shares": existing_long,
        "data_status": data_status,
        "volatility_features": {key: float(val) for key, val in features.items()} if features else {},
        "diagnostic_notes": notes,
    }

    return observations, features


def macro_volatility_sentinel_agent(state: AgentState, agent_id: str = "macro_volatility_sentinel_agent"):
    """Flag volatility expansion regimes so risk manager can avoid pro-cyclical long adds."""

    data = state["data"]
    tickers = data["tickers"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    portfolio = data.get("portfolio", {})
    positions = portfolio.get("positions") or {}
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    sentinel_view: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Scanning volatility regime")
        prices = get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
        )

        existing_position = positions.get(ticker, {}) or {}
        existing_long = int(_safe_float(existing_position.get("long", 0), 0.0))

        df = prices_to_df(prices) if prices else None
        observations, features = _prepare_observation(df=df, ticker=ticker, existing_long=existing_long)
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

        confidence = int(max(0, min(round(decision.confidence), 100)))
        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": confidence,
            "reasoning": decision.reasoning,
            "metrics": observations["volatility_features"],
            "constraints": decision.constraints or {},
            "meta": {"observations": observations},
        }

        progress.update_status(agent_id, ticker, f"{payload['signal'].upper()} @ {confidence}/100")

        sentinel_view[ticker] = payload

    message = HumanMessage(content=json.dumps(sentinel_view), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(sentinel_view, "Macro Volatility Sentinel")

    state["data"].setdefault("analyst_signals", {})[agent_id] = sentinel_view
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


async def macro_volatility_sentinel_agent_async(state: AgentState, agent_id: str = "macro_volatility_sentinel_agent"):
    """Async volatility sentinel leveraging persona observations."""

    data = state["data"]
    tickers = data["tickers"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    portfolio = data.get("portfolio", {})
    positions = portfolio.get("positions") or {}
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    sentinel_view: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Scanning volatility regime")
        prices = await get_prices_async(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
        )

        existing_position = positions.get(ticker, {}) or {}
        existing_long = int(_safe_float(existing_position.get("long", 0), 0.0))

        df = await prices_to_df_async(prices) if prices else None
        observations, features = _prepare_observation(df=df, ticker=ticker, existing_long=existing_long)
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

        confidence = int(max(0, min(round(decision.confidence), 100)))
        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": confidence,
            "reasoning": decision.reasoning,
            "metrics": observations["volatility_features"],
            "constraints": decision.constraints or {},
            "meta": {"observations": observations},
        }

        progress.update_status(agent_id, ticker, f"{payload['signal'].upper()} @ {confidence}/100")

        sentinel_view[ticker] = payload

    message = HumanMessage(content=json.dumps(sentinel_view), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(sentinel_view, "Macro Volatility Sentinel")

    await update_analyst_signals_async(state, agent_id, sentinel_view)
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }
