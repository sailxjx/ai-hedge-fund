"""Sentinel detecting persistent downside flows to enforce defensive positioning."""

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
PERSONA_NAME = "Vale"
PERSONA_ROLE = "a downside-flow sentinel who leans short when crash pressure builds"
PERSONA_BACKSTORY = "Vale is a risk sentry forged during 2008, now combining flow statistics with trader instinct to decide when the book must lean short."
PERSONA_INSTRUCTIONS = (
    "Weigh drawdowns, downside hit-rates, volatility spikes, and trend slopes before forcing risk-off moves.",
    "Reserve the harshest constraints for genuine crash-flow signatures; otherwise prescribe calibrated trims.",
    "Explain decisions in first person, pointing to the metrics that mattered most.",
)
ALLOWED_SIGNALS = [
    "no_data",
    "insufficient_history",
    "stable",
    "downside_trend",
    "crash_flow",
]


def _safe_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed


def _compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    closes = df["close"].astype(float).dropna()
    if closes.empty or len(closes) < 40:
        return {}

    returns = closes.pct_change().dropna()
    if returns.empty:
        return {}

    recent = returns.tail(10)
    tail_20 = returns.tail(20)

    ret_1 = closes.pct_change().iloc[-1]
    ret_3 = closes.pct_change(3).iloc[-1]
    ret_5 = closes.pct_change(5).iloc[-1]
    ret_10 = closes.pct_change(10).iloc[-1]
    ret_20 = closes.pct_change(20).iloc[-1]

    rolling_high_20 = closes.rolling(window=20).max()
    rolling_high_40 = closes.rolling(window=40).max()

    drawdown_20 = float(closes.iloc[-1] / max(rolling_high_20.iloc[-1], EPSILON) - 1.0)
    drawdown_40 = float(closes.iloc[-1] / max(rolling_high_40.iloc[-1], EPSILON) - 1.0)

    vol_5 = returns.rolling(window=5).std(ddof=0).iloc[-1]
    vol_20 = returns.rolling(window=20).std(ddof=0).iloc[-1]
    vol_ratio = float(vol_5 / max(vol_20, EPSILON) - 1.0) if vol_20 > 0 else 0.0

    downside_share10 = float((recent < 0).mean()) if len(recent) >= 3 else 0.0
    tail_loss20 = float(np.percentile(tail_20, 25)) if len(tail_20) >= 5 else 0.0

    closes_tail = closes.tail(10)
    if len(closes_tail) >= 4:
        idx = np.arange(len(closes_tail))
        slope, _ = np.polyfit(idx, closes_tail, 1)
        trend_slope10 = float(slope / max(abs(closes_tail.mean()), EPSILON))
    else:
        trend_slope10 = 0.0

    return {
        "ret_1": float(ret_1) if np.isfinite(ret_1) else 0.0,
        "ret_3": float(ret_3) if np.isfinite(ret_3) else 0.0,
        "ret_5": float(ret_5) if np.isfinite(ret_5) else 0.0,
        "ret_10": float(ret_10) if np.isfinite(ret_10) else 0.0,
        "ret_20": float(ret_20) if np.isfinite(ret_20) else 0.0,
        "drawdown_20": drawdown_20,
        "drawdown_40": drawdown_40,
        "vol_ratio": vol_ratio,
        "downside_share10": downside_share10,
        "tail_loss20": float(tail_loss20),
        "trend_slope10": trend_slope10,
    }


def _prepare_observation(
    *,
    df: pd.DataFrame | None,
    ticker: str,
    existing_long: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    notes: list[str] = []
    metrics: dict[str, float] = {}
    data_status = "missing_prices"

    if df is None:
        notes.append("No price history retrieved for the analysis window.")
    else:
        if df.empty or len(df) < 40:
            data_status = "insufficient_history"
            notes.append("Need at least 40 observations to analyse flow regimes.")
        else:
            metrics = _compute_metrics(df)
            if not metrics:
                data_status = "feature_extraction_failed"
                notes.append("Unable to compute downside metrics from the recent window.")
            else:
                data_status = "data_ready"
                notes.append(
                    "Downside diagnostics: "
                    f"5d return {metrics['ret_5']:.1%}, "
                    f"20d drawdown {metrics['drawdown_20']:.1%}, "
                    f"vol ratio {metrics['vol_ratio']:.2f}, "
                    f"downside-hit rate10 {metrics['downside_share10']:.2f}."
                )

    observations = {
        "ticker": ticker,
        "existing_long_shares": existing_long,
        "data_status": data_status,
        "downside_metrics": {key: float(val) for key, val in metrics.items()} if metrics else {},
        "diagnostic_notes": notes,
    }

    return observations, metrics


##### Downside Flow Sentinel #####
def downside_flow_sentinel_agent(state: AgentState, agent_id: str = "downside_flow_sentinel_agent"):
    """Detect drawdown/volatility clusters that warrant a short bias."""

    data = state.get("data", {})
    tickers = data.get("tickers", [])
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    portfolio = data.get("portfolio", {})
    positions = portfolio.get("positions", {}) or {}
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    sentinel_signals: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Scouting downside flows")

        prices = get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
        )

        position = positions.get(ticker, {}) or {}
        existing_long = _safe_int(position.get("long"))

        df = prices_to_df(prices) if prices else None
        observations, metrics = _prepare_observation(df=df, ticker=ticker, existing_long=existing_long)
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
            default_signal="stable",
            default_confidence=55.0,
            default_reasoning="Defaulted to stable after observation-only fallback.",
        )

        metrics_payload = dict(observations["downside_metrics"])
        metrics_payload["existing_long_shares"] = existing_long

        confidence = int(max(0, min(round(decision.confidence), 100)))
        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": confidence,
            "reasoning": decision.reasoning,
            "metrics": metrics_payload,
            "constraints": decision.constraints or {},
            "meta": {"observations": observations},
        }

        progress.update_status(agent_id, ticker, f"{payload['signal'].upper()} @ {confidence}/100")

        sentinel_signals[ticker] = payload

    message = HumanMessage(content=json.dumps(sentinel_signals), name=agent_id)

    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(sentinel_signals, "Downside Flow Sentinel")

    state.setdefault("data", {}).setdefault("analyst_signals", {})[agent_id] = sentinel_signals
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state.get("messages", []) + [message],
        "data": state["data"],
    }


async def downside_flow_sentinel_agent_async(state: AgentState, agent_id: str = "downside_flow_sentinel_agent"):
    """Async downside flow sentinel to support persona parallelism."""

    data = state.get("data", {})
    tickers = data.get("tickers", [])
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    portfolio = data.get("portfolio", {})
    positions = portfolio.get("positions", {}) or {}
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    sentinel_signals: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Scouting downside flows")

        prices = await get_prices_async(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
        )

        position = positions.get(ticker, {}) or {}
        existing_long = _safe_int(position.get("long"))

        df = await prices_to_df_async(prices) if prices else None
        observations, metrics = _prepare_observation(df=df, ticker=ticker, existing_long=existing_long)
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
            default_signal="stable",
            default_confidence=55.0,
            default_reasoning="Defaulted to stable after observation-only fallback.",
        )

        metrics_payload = dict(observations["downside_metrics"])
        metrics_payload["existing_long_shares"] = existing_long

        confidence = int(max(0, min(round(decision.confidence), 100)))
        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": confidence,
            "reasoning": decision.reasoning,
            "metrics": metrics_payload,
            "constraints": decision.constraints or {},
            "meta": {"observations": observations},
        }

        progress.update_status(agent_id, ticker, f"{payload['signal'].upper()} @ {confidence}/100")

        sentinel_signals[ticker] = payload

    message = HumanMessage(content=json.dumps(sentinel_signals), name=agent_id)

    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(sentinel_signals, "Downside Flow Sentinel")

    await update_analyst_signals_async(state, agent_id, sentinel_signals)
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state.get("messages", []) + [message],
        "data": state["data"],
    }


__all__ = ["downside_flow_sentinel_agent", "downside_flow_sentinel_agent_async"]
