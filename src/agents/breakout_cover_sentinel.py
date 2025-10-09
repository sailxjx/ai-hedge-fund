"""Sentinel agent that forces stubborn shorts to cover during upside breakouts."""

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
PERSONA_NAME = "Vega"
PERSONA_ROLE = "an anthropomorphic breakout sentinel safeguarding the portfolio from runaway rallies"
PERSONA_BACKSTORY = (
    "A former short seller, Vega now scouts upside breakouts to decide when entrenched shorts must yield to momentum."
)
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


def _bounded_component(value: float, scale: float, cap: float = 1.0) -> float:
    if scale <= 0 or not np.isfinite(value):
        return 0.0
    normalised = value / scale
    return float(np.clip(normalised, 0.0, cap))


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


def _score_breakout(features: dict[str, float]) -> tuple[str, float]:
    ret3_component = _bounded_component(max(0.0, features.get("ret_3", 0.0)), 0.04, cap=1.0)
    ret5_component = _bounded_component(max(0.0, features.get("ret_5", 0.0)), 0.06, cap=1.1)
    ret10_component = _bounded_component(max(0.0, features.get("ret_10", 0.0)), 0.12, cap=1.0)
    breakout_component = _bounded_component(max(0.0, features.get("breakout_gap", 0.0)), 0.035, cap=1.1)
    ema_component = _bounded_component(max(0.0, features.get("ema_slope", 0.0)), 0.05, cap=1.0)
    volume_component = _bounded_component(max(0.0, features.get("volume_ratio", 0.0)), 0.9, cap=1.0)

    score = float(
        np.clip(
            0.18 * ret3_component
            + 0.25 * ret5_component
            + 0.20 * ret10_component
            + 0.17 * breakout_component
            + 0.10 * ema_component
            + 0.10 * volume_component,
            0.0,
            1.35,
        )
    )

    ret5 = features.get("ret_5", 0.0)
    breakout_gap = features.get("breakout_gap", 0.0)

    if score >= 0.75 or (ret5 >= 0.055 and breakout_gap >= 0.018):
        return "breakout_cover", score
    if score >= 0.45 or (ret5 >= 0.04 and breakout_gap >= 0.012):
        return "uptrend_defensive", score
    return "balanced", score



def breakout_cover_sentinel_agent(state: AgentState, agent_id: str = "breakout_cover_sentinel_agent"):
    """Aggressively unwind shorts when price action signals durable upside breakouts."""

    data = state["data"]
    tickers = data["tickers"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    portfolio = data.get("portfolio", {})
    positions = (portfolio.get("positions") or {})
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
        features: dict[str, float] = {}
        score = 0.0
        auto_signal = "no_data"
        auto_confidence = 0
        auto_reasoning = "I do not have sufficient market data to assess breakout pressure."
        auto_constraints: dict[str, Any] = {}

        if not prices:
            progress.update_status(agent_id, ticker, "Failed: Missing price data")
        else:
            df = prices_to_df(prices)
            if df.empty or len(df) < 25:
                auto_signal = "insufficient_history"
                auto_confidence = 20
                auto_reasoning = "Need at least 25 sessions to evaluate breakout persistence."
                progress.update_status(agent_id, ticker, "Failed: Insufficient history")
            else:
                features = _compute_features(df)
                if not features:
                    auto_signal = "no_features"
                    auto_confidence = 15
                    auto_reasoning = "Unable to derive breakout metrics from recent history."
                    progress.update_status(agent_id, ticker, "Failed: Feature extraction")
                else:
                    auto_signal, score = _score_breakout(features)
                    auto_confidence = int(round(min(1.0, score) * 100))
                    reasoning_parts = [
                        f"5d {features['ret_5']:.1%}",
                        f"10d {features['ret_10']:.1%}",
                        f"gap_high {features['breakout_gap']:.1%}",
                        f"vol_ratio {features['volume_ratio'] + 1:.2f}x",
                    ]
                    auto_reasoning = "; ".join(reasoning_parts)

                    auto_constraints = {}
                    if auto_signal == "breakout_cover":
                        auto_constraints.update(
                            {
                                "preferred_direction": "long",
                                "block_new_shorts": True,
                                "allow_short": False,
                                "max_short_exposure_pct": 0.0,
                                "target_short_shares": 0,
                            }
                        )
                        if current_short > 0:
                            auto_constraints["force_cover_qty"] = current_short
                            auto_constraints["force_cover_reason"] = "Breakout cover sentinel forcing full unwind"
                    elif auto_signal == "uptrend_defensive":
                        trimmed_short = max(0, int(np.floor(current_short * 0.3)))
                        auto_constraints.update(
                            {
                                "preferred_direction": "long",
                                "block_new_shorts": True,
                                "allow_short": True,
                                "max_short_exposure_pct": 0.02,
                                "target_short_shares": trimmed_short,
                            }
                        )
                        if current_short > trimmed_short:
                            auto_constraints["force_cover_qty"] = current_short - trimmed_short
                            auto_constraints["force_cover_reason"] = "Breakout cover sentinel trimming shorts"
                    else:
                        auto_constraints.update(
                            {
                                "allow_short": True,
                                "max_short_exposure_pct": 0.06,
                            }
                        )

        observations = {
            "ticker": ticker,
            "current_short_shares": current_short,
            "breakout_features": {key: round(val, 6) for key, val in features.items()},
            "breakout_score": round(score, 3),
            "suggested_constraints": dict(auto_constraints),
            "auto_signal_hint": auto_signal,
            "auto_reasoning": auto_reasoning,
            "auto_confidence": auto_confidence,
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
            default_signal="balanced",
            default_confidence=55.0,
            default_reasoning="Defaulted to balanced after missing persona call.",
        )

        metrics = dict(observations["breakout_features"])
        metrics.update(
            {
                "score": round(score, 3),
                "current_short_shares": current_short,
            }
        )

        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": int(max(0, min(round(decision.confidence), 100))),
            "score": round(score, 3),
            "reasoning": decision.reasoning,
            "metrics": metrics,
            "constraints": decision.constraints or {},
            "meta": {"observations": observations},
        }

        progress.update_status(
            agent_id,
            ticker,
            f"{payload['signal'].upper()} @ {payload['confidence']}/100 | breakout {payload['score']:.2f}",
        )

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
