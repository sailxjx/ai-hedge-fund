"""Guardian agent that suppresses short exposure during squeeze conditions."""

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
PERSONA_NAME = "Nyx"
PERSONA_ROLE = "an anthropomorphic squeeze guardian who shields the book from violent upside reversals"
PERSONA_BACKSTORY = (
    "Nyx once ran a discretionary short book and earned her scars from brutal squeezes; now she blends tape-reading intuition with quantitative cues to guard the fund."
)
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


def _bounded_score(value: float, scale: float, cap: float = 1.0) -> float:
    if not np.isfinite(value):
        return 0.0
    if scale <= 0:
        return 0.0
    normalised = value / scale
    return float(np.clip(normalised, 0.0, cap))


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


def short_squeeze_guardian_agent(state: AgentState, agent_id: str = "short_squeeze_guardian_agent"):
    """Detect aggressive upside squeezes and enforce defensive short constraints."""

    data = state["data"]
    tickers = data["tickers"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    portfolio = data.get("portfolio", {})
    positions = (portfolio.get("positions") or {})
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

        features: dict[str, float] = {}
        squeeze_score = 0.0
        current_short = int(_safe_float((positions.get(ticker, {}) or {}).get("short", 0), 0.0))

        auto_signal = "no_data"
        auto_confidence = 0
        auto_reasoning = "No price history available to evaluate squeeze risk."
        auto_constraints: dict[str, Any] = {}
        component_breakdown: dict[str, float] = {}

        if not prices:
            progress.update_status(agent_id, ticker, "Failed: Missing price data")
        else:
            df = prices_to_df(prices)
            if df.empty or len(df) < 15:
                auto_signal = "insufficient_history"
                auto_confidence = 20
                auto_reasoning = "Need at least 15 sessions to size squeeze risk."
                progress.update_status(agent_id, ticker, "Failed: Insufficient history")
            else:
                features = _compute_features(df)
                if not features:
                    auto_signal = "no_features"
                    auto_confidence = 15
                    auto_reasoning = "Unable to derive squeeze metrics from the recent history."
                    progress.update_status(agent_id, ticker, "Failed: Feature extraction")
                else:
                    velocity_component = _bounded_score(max(0.0, features["velocity_z"]), 2.5)
                    momentum_component = _bounded_score(max(0.0, features["ret_10"]), 0.12)
                    gap_high_component = _bounded_score(max(0.0, features["gap_vs_high"]), 0.03)
                    gap_ma_component = _bounded_score(max(0.0, features["gap_vs_ma"]), 0.08)
                    volume_component = _bounded_score(max(0.0, features["volume_ratio"]), 1.0)
                    acceleration_component = _bounded_score(max(0.0, features["acceleration"]), 0.08)
                    range_component = _bounded_score(max(0.0, features["range_ratio"]), 0.02)

                    squeeze_score = (
                        0.30 * velocity_component
                        + 0.20 * momentum_component
                        + 0.15 * gap_high_component
                        + 0.10 * gap_ma_component
                        + 0.10 * volume_component
                        + 0.10 * acceleration_component
                        + 0.05 * range_component
                    )
                    squeeze_score = float(np.clip(squeeze_score, 0.0, 1.25))

                    if squeeze_score >= 0.70:
                        auto_signal = "squeeze_warning"
                    elif squeeze_score >= 0.45:
                        auto_signal = "elevated_risk"
                    else:
                        auto_signal = "calm"

                    auto_confidence = int(round(min(1.0, squeeze_score) * 100))

                    auto_constraints = {}
                    reasoning_parts = [
                        f"5d {features['ret_5']:.1%}",
                        f"10d {features['ret_10']:.1%}",
                        f"vel_z {features['velocity_z']:.2f}",
                        f"vol/avg {features['volume_ratio'] + 1:.2f}x",
                    ]
                    auto_reasoning = "; ".join(reasoning_parts)

                    if auto_signal == "squeeze_warning":
                        auto_constraints.update(
                            {
                                "allow_short": False,
                                "block_new_shorts": True,
                                "max_short_exposure_pct": 0.0,
                                "target_short_shares": 0,
                            }
                        )
                        if current_short > 0:
                            auto_constraints["force_cover_qty"] = current_short
                            auto_constraints["force_cover_reason"] = "Short squeeze guardian triggered"
                    elif auto_signal == "elevated_risk":
                        auto_constraints.update(
                            {
                                "allow_short": True,
                                "max_short_exposure_pct": 0.03,
                                "block_new_shorts": True,
                            }
                        )
                        if current_short > 0:
                            trimmed_target = max(0, int(np.floor(current_short * 0.4)))
                            auto_constraints["target_short_shares"] = trimmed_target
                            auto_constraints["force_cover_reason"] = "Short squeeze risk trimming"
                    else:
                        auto_constraints.update(
                            {
                                "allow_short": True,
                                "max_short_exposure_pct": 0.08,
                            }
                        )

                    component_breakdown = {
                        "velocity_component": round(velocity_component, 3),
                        "momentum_component": round(momentum_component, 3),
                        "gap_high_component": round(gap_high_component, 3),
                        "gap_ma_component": round(gap_ma_component, 3),
                        "volume_component": round(volume_component, 3),
                        "acceleration_component": round(acceleration_component, 3),
                        "range_component": round(range_component, 3),
                    }

        observations = {
            "ticker": ticker,
            "current_short_shares": current_short,
            "squeeze_features": {key: round(val, 6) for key, val in features.items()},
            "component_breakdown": component_breakdown,
            "squeeze_score": round(squeeze_score, 3),
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
            default_signal="balanced",
            default_confidence=55.0,
            default_reasoning="Defaulted to balanced guidance after missing persona output.",
        )

        metrics = {
            **observations["squeeze_features"],
            **component_breakdown,
            "squeeze_score": round(squeeze_score, 3),
            "current_short_shares": current_short,
        }

        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": int(max(0, min(round(decision.confidence), 100))),
            "score": round(squeeze_score, 3),
            "reasoning": decision.reasoning,
            "metrics": metrics,
            "constraints": decision.constraints or {},
            "meta": {"observations": observations},
        }

        progress.update_status(
            agent_id,
            ticker,
            f"{payload['signal'].upper()} @ {payload['confidence']}/100 | squeeze {payload['score']:.2f}",
        )

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
