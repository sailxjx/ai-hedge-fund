"""Sentinel agent monitoring volatility expansion to curb long exposure overrides."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.agents.persona_utils import persona_from_observations
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices, prices_to_df
from src.utils.api_key import get_api_key_from_state
from src.utils.progress import progress


EPSILON = 1e-9
PERSONA_NAME = "Storm"
PERSONA_ROLE = "a volatility warden who reins in long exposure when macro turbulence spikes"
PERSONA_BACKSTORY = (
    "Storm cut their teeth running volatility overlay strategies and now advises the fund on when swelling turbulence should throttle long adds."
)
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


def _bounded_component(value: float, scale: float, cap: float = 1.0) -> float:
    if scale <= 0 or not np.isfinite(value):
        return 0.0
    normalised = value / scale
    return float(np.clip(normalised, 0.0, cap))


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


def _score_risk(features: dict[str, float]) -> tuple[str, float]:
    vol_component = _bounded_component(max(0.0, features.get("vol_ratio", 0.0)), 0.45, cap=1.15)
    atr_component = _bounded_component(max(0.0, features.get("atr_ratio", 0.0)), 0.35, cap=1.0)
    downside_component = _bounded_component(max(0.0, -features.get("ret_5", 0.0)), 0.12, cap=1.0)
    drawdown_component = _bounded_component(max(0.0, -features.get("drawdown_20", 0.0)), 0.18, cap=1.0)
    gap_component = _bounded_component(max(0.0, -features.get("ret_1", 0.0)), 0.06, cap=1.0)

    risk_score = float(
        np.clip(
            0.32 * vol_component
            + 0.20 * atr_component
            + 0.20 * downside_component
            + 0.18 * drawdown_component
            + 0.10 * gap_component,
            0.0,
            1.35,
        )
    )

    vol_ratio = features.get("vol_ratio", 0.0)
    ret_5 = features.get("ret_5", 0.0)
    drawdown_20 = features.get("drawdown_20", 0.0)

    if risk_score >= 0.75 or (vol_ratio > 0.55 and ret_5 <= -0.04) or (drawdown_20 <= -0.08 and vol_ratio > 0.45):
        return "crash_alert", risk_score
    if risk_score >= 0.45 or (vol_ratio > 0.30 and ret_5 <= -0.02):
        return "vol_watch", risk_score
    return "calm", risk_score


def macro_volatility_sentinel_agent(state: AgentState, agent_id: str = "macro_volatility_sentinel_agent"):
    """Flag volatility expansion regimes so risk manager can avoid pro-cyclical long adds."""

    data = state["data"]
    tickers = data["tickers"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    portfolio = data.get("portfolio", {})
    positions = (portfolio.get("positions") or {})
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

        features: dict[str, float] = {}
        risk_score = 0.0
        existing_position = positions.get(ticker, {}) or {}
        existing_long = int(_safe_float(existing_position.get("long", 0), 0.0))

        auto_signal = "no_data"
        auto_confidence = 0
        auto_reasoning = "Missing price history prevents volatility diagnostics."
        auto_constraints: dict[str, Any] = {}

        if not prices:
            progress.update_status(agent_id, ticker, "Failed: Missing price data")
        else:
            df = prices_to_df(prices)
            if df.empty or len(df) < 40:
                auto_signal = "insufficient_history"
                auto_confidence = 20
                auto_reasoning = "Need at least 40 observations to gauge volatility expansion."
                progress.update_status(agent_id, ticker, "Failed: Insufficient history")
            else:
                features = _compute_features(df)
                if not features:
                    auto_signal = "no_features"
                    auto_confidence = 15
                    auto_reasoning = "Unable to derive volatility metrics from recent history."
                    progress.update_status(agent_id, ticker, "Failed: Feature extraction")
                else:
                    auto_signal, risk_score = _score_risk(features)
                    auto_confidence = int(round(min(1.0, risk_score) * 100))

                    auto_constraints = {}
                    reasoning_parts = [
                        f"vol5/20 ratio {features['vol_ratio']:+.2f}",
                        f"5d return {features['ret_5']:.1%}",
                        f"drawdown20 {features['drawdown_20']:.1%}",
                    ]
                    auto_reasoning = "; ".join(reasoning_parts)

                    if auto_signal == "crash_alert":
                        auto_constraints.update(
                            {
                                "preferred_direction": "short",
                                "max_long_exposure_pct": 0.05,
                                "max_additional_long_shares": 0,
                                "max_long_shares": max(0, int(np.floor(existing_long * 0.6))),
                                "reduce_position_change": True,
                            }
                        )
                    elif auto_signal == "vol_watch":
                        additional_long_cap = max(0, int(np.floor(existing_long * 0.25)))
                        auto_constraints.update(
                            {
                                "max_long_exposure_pct": 0.10,
                                "max_additional_long_shares": additional_long_cap,
                                "reduce_position_change": True,
                            }
                        )
                    else:
                        auto_constraints["max_long_exposure_pct"] = 0.18

        observations = {
            "ticker": ticker,
            "existing_long_shares": existing_long,
            "volatility_features": {key: round(val, 6) if isinstance(val, float) else val for key, val in features.items()},
            "risk_score": round(risk_score, 3),
            "suggested_constraints": dict(auto_constraints),
            "auto_signal_hint": auto_signal,
            "auto_confidence_hint": auto_confidence,
            "auto_reasoning": auto_reasoning,
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
            default_signal="vol_monitor",
            default_confidence=55.0,
            default_reasoning="Defaulted to vol_monitor after observation-only fallback.",
        )

        metrics = dict(observations["volatility_features"])
        metrics.update(
            {
                "risk_score": round(risk_score, 3),
                "existing_long_shares": existing_long,
            }
        )

        confidence = int(max(0, min(round(decision.confidence), 100)))
        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": confidence,
            "score": round(risk_score, 3),
            "reasoning": decision.reasoning,
            "metrics": metrics,
            "constraints": decision.constraints or {},
            "meta": {"observations": observations},
        }

        progress.update_status(
            agent_id,
            ticker,
            f"{payload['signal'].upper()} @ {confidence}/100 | score {payload['score']:.2f}",
        )

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
