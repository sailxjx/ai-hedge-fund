"""Sentinel detecting range recoveries to unwind stale shorts."""

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
PERSONA_NAME = "Lyra"
PERSONA_ROLE = "a range-recovery sentinel who protects the book from upside drift inside compressions"
PERSONA_BACKSTORY = (
    "Lyra made her reputation unwinding crowded shorts the moment ranges broke higher; she now balances quantitative drift diagnostics with trader intuition."
)
PERSONA_INSTRUCTIONS = (
    "Lean on the range position, rebound metrics, and volatility contraction to judge the risk of stubborn shorts.",
    "Reserve full covers for decisive breaks; allow softer guidance when evidence is mixed.",
    "Explain the call in first person, noting which metrics mattered most.",
)
ALLOWED_SIGNALS = [
    "no_data",
    "insufficient_history",
    "monitor",
    "range_monitor",
    "drift_recovery",
    "compression_break",
]


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _clip_component(value: float, scale: float, *, cap: float = 1.0) -> float:
    if scale <= 0 or not np.isfinite(value):
        return 0.0
    normalised = value / scale
    return float(np.clip(normalised, 0.0, cap))


def _normalised_slope(series: pd.Series, window: int) -> float:
    if len(series) < window:
        return 0.0
    tail = series.tail(window)
    if tail.nunique() <= 1:
        return 0.0
    idx = np.arange(len(tail), dtype=float)
    try:
        slope, _ = np.polyfit(idx, tail.to_numpy(dtype=float), 1)
    except np.linalg.LinAlgError:
        return 0.0
    mean_val = float(np.nanmean(np.abs(tail)))
    if not np.isfinite(mean_val) or mean_val <= EPSILON:
        return 0.0
    return float(slope / mean_val)


def _compute_features(df: pd.DataFrame) -> dict[str, float]:
    df = df.sort_index()
    if df.empty or len(df) < 30:
        return {}

    close = df["close"].astype(float).dropna()
    high = df.get("high", df["close"]).astype(float).dropna()
    low = df.get("low", df["close"]).astype(float).dropna()

    if close.empty or len(close) < 30:
        return {}

    returns = close.pct_change().dropna()
    if returns.empty:
        return {}

    ret_1 = float(close.pct_change().iloc[-1])
    ret_3 = float(close.pct_change(3).iloc[-1]) if len(close) > 3 else 0.0
    ret_5 = float(close.pct_change(5).iloc[-1]) if len(close) > 5 else 0.0
    ret_10 = float(close.pct_change(10).iloc[-1]) if len(close) > 10 else 0.0

    rolling_high_20 = high.rolling(window=20).max()
    rolling_low_20 = low.rolling(window=20).min()
    rolling_high_40 = high.rolling(window=40).max()

    latest_close = float(close.iloc[-1])
    high_20 = float(rolling_high_20.iloc[-1]) if len(rolling_high_20.dropna()) else latest_close
    low_20 = float(rolling_low_20.iloc[-1]) if len(rolling_low_20.dropna()) else latest_close
    high_40 = float(rolling_high_40.iloc[-1]) if len(rolling_high_40.dropna()) else latest_close

    range_width = max(high_20 - low_20, 0.0)
    range_position = 0.0
    if range_width > EPSILON:
        range_position = float(np.clip((latest_close - low_20) / range_width, 0.0, 1.0))

    range_width_pct = range_width / max(latest_close, EPSILON)

    drawdown_20 = latest_close / max(high_20, EPSILON) - 1.0
    drawdown_40 = latest_close / max(high_40, EPSILON) - 1.0
    rebound_from_low = latest_close / max(low_20, EPSILON) - 1.0 if low_20 > 0 else 0.0

    vol_5 = returns.rolling(window=5).std(ddof=0)
    vol_20 = returns.rolling(window=20).std(ddof=0)
    vol_ratio = 0.0
    if len(vol_20.dropna()):
        latest_vol_20 = float(vol_20.iloc[-1])
        if latest_vol_20 > 0:
            latest_vol_5 = float(vol_5.iloc[-1]) if len(vol_5.dropna()) else latest_vol_20
            vol_ratio = latest_vol_5 / latest_vol_20 - 1.0

    high_low = (high - low).abs()
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            high_low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr_14 = float(true_range.rolling(window=14).mean().iloc[-1]) if len(true_range.dropna()) >= 14 else 0.0
    atr_ratio = atr_14 / max(latest_close, EPSILON)

    trend_slope_8 = _normalised_slope(close, 8)
    trend_slope_12 = _normalised_slope(close, 12)

    return {
        "close": latest_close,
        "ret_1": ret_1,
        "ret_3": ret_3,
        "ret_5": ret_5,
        "ret_10": ret_10,
        "drawdown_20": drawdown_20,
        "drawdown_40": drawdown_40,
        "rebound_from_low": rebound_from_low,
        "range_width_pct": range_width_pct,
        "range_position": range_position,
        "vol_ratio": vol_ratio,
        "atr_ratio": atr_ratio,
        "trend_slope_8": trend_slope_8,
        "trend_slope_12": trend_slope_12,
    }


def _score_recovery(features: dict[str, float]) -> tuple[str, float]:
    drift_component = _clip_component(max(0.0, features.get("ret_5", 0.0)), 0.05, cap=1.2)
    slope_component = _clip_component(max(0.0, features.get("trend_slope_8", 0.0)), 0.007, cap=1.1)
    position_component = float(
        np.clip(features.get("range_position", 0.0) - 0.45, 0.0, 0.55) / 0.55
    )
    contraction_component = _clip_component(max(0.0, -features.get("vol_ratio", 0.0)), 0.6, cap=1.0)
    rebound_component = _clip_component(max(0.0, features.get("rebound_from_low", 0.0)), 0.12, cap=1.0)
    width_component = _clip_component(max(0.0, 0.1 - features.get("range_width_pct", 0.1)), 0.08, cap=1.0)

    score = float(
        np.clip(
            0.24 * drift_component
            + 0.20 * slope_component
            + 0.18 * position_component
            + 0.18 * contraction_component
            + 0.12 * rebound_component
            + 0.08 * width_component,
            0.0,
            1.25,
        )
    )

    ret_5 = features.get("ret_5", 0.0)
    trend_slope = features.get("trend_slope_8", 0.0)
    range_position = features.get("range_position", 0.0)
    range_width = features.get("range_width_pct", 1.0)
    vol_ratio = features.get("vol_ratio", 0.0)

    if ret_5 >= 0.045 or (range_position >= 0.8 and trend_slope >= 0.004 and range_width <= 0.07):
        return "compression_break", score
    if score >= 0.55 or (ret_5 >= 0.02 and trend_slope >= 0.0025 and vol_ratio <= 0.15):
        return "drift_recovery", score
    if score >= 0.38 or (range_position >= 0.65 and vol_ratio <= 0.2):
        return "range_monitor", score
    return "monitor", score


def _confidence(stage: str, score: float) -> int:
    if stage == "compression_break":
        base, scale, ceiling = 70, 28, 98
    elif stage == "drift_recovery":
        base, scale, ceiling = 60, 24, 94
    elif stage == "range_monitor":
        base, scale, ceiling = 52, 20, 90
    else:
        base, scale, ceiling = 44, 18, 80
    conf = base + score * scale
    return int(np.clip(conf, 35, ceiling))


##### Range Recovery Sentinel #####
def range_recovery_sentinel_agent(
    state: AgentState, agent_id: str = "range_recovery_sentinel_agent"
):
    """Trim stubborn shorts when price drifts higher inside compressed ranges."""

    data = state.get("data", {})
    tickers = data.get("tickers", [])
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    portfolio = data.get("portfolio", {}) or {}
    positions = portfolio.get("positions", {}) or {}
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    sentinel_signals: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Evaluating range recovery")
        prices = get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            api_key=api_key,
        )

        features: dict[str, float] = {}
        score = 0.0
        current_position = positions.get(ticker, {}) or {}
        existing_short = _safe_int(current_position.get("short"))

        auto_signal = "no_data"
        auto_confidence = 35
        auto_reasoning = "Missing price history prevents range recovery diagnostics."
        auto_constraints: dict[str, Any] = {}

        if not prices:
            progress.update_status(agent_id, ticker, "Failed: Missing price data")
        else:
            df = prices_to_df(prices)
            if df.empty:
                auto_signal = "no_data"
                auto_confidence = 35
                auto_reasoning = "Empty price history prevents range recovery diagnostics."
                progress.update_status(agent_id, ticker, "Failed: Empty price data")
            else:
                features = _compute_features(df)
                if not features:
                    auto_signal = "insufficient_history"
                    auto_confidence = 38
                    auto_reasoning = "Need at least 30 observations with variance to evaluate drift."
                    progress.update_status(agent_id, ticker, "Failed: Insufficient history")
                else:
                    auto_signal, score = _score_recovery(features)
                    auto_confidence = _confidence(auto_signal, score)
                    reasoning_parts = [
                        f"5d {features['ret_5']:.1%}",
                        f"10d {features['ret_10']:.1%}",
                        f"range pos {features['range_position']:.2f}",
                        f"vol ratio {features['vol_ratio']:+.2f}",
                    ]
                    auto_reasoning = "; ".join(reasoning_parts)

                    auto_constraints = {}
                    if auto_signal == "compression_break":
                        auto_constraints.update(
                            {
                                "preferred_direction": "long",
                                "block_new_shorts": True,
                                "allow_short": False,
                                "max_short_exposure_pct": 0.0,
                                "max_additional_short_shares": 0,
                                "target_short_shares": 0,
                            }
                        )
                        if existing_short > 0:
                            auto_constraints["force_cover_qty"] = existing_short
                            auto_constraints["force_cover_reason"] = (
                                "Range recovery sentinel covering shorts after upside break."
                            )
                    elif auto_signal == "drift_recovery":
                        trimmed_target = max(0, int(np.floor(existing_short * 0.35)))
                        force_cover_qty = max(0, existing_short - trimmed_target)
                        auto_constraints.update(
                            {
                                "preferred_direction": "neutral",
                                "block_new_shorts": True,
                                "allow_short": False,
                                "max_short_exposure_pct": 0.03,
                                "max_additional_short_shares": 0,
                                "target_short_shares": trimmed_target,
                            }
                        )
                        if force_cover_qty > 0:
                            auto_constraints["force_cover_qty"] = force_cover_qty
                            auto_constraints["force_cover_reason"] = (
                                "Range recovery sentinel trimming shorts into upside drift."
                            )
                    elif auto_signal == "range_monitor":
                        if existing_short > 0:
                            softened_target = max(0, int(np.ceil(existing_short * 0.6)))
                            auto_constraints.update(
                                {
                                    "preferred_direction": "neutral",
                                    "block_new_shorts": True,
                                    "allow_short": False,
                                    "max_short_exposure_pct": 0.05,
                                    "max_additional_short_shares": 0,
                                    "target_short_shares": softened_target,
                                }
                            )
                        else:
                            auto_constraints.update(
                                {
                                    "preferred_direction": "neutral",
                                    "block_new_shorts": True,
                                    "allow_short": False,
                                    "max_short_exposure_pct": 0.04,
                                    "max_additional_short_shares": 0,
                                }
                            )
                    else:
                        auto_constraints = {}

        observations = {
            "ticker": ticker,
            "existing_short_shares": existing_short,
            "range_features": {key: round(val, 6) for key, val in features.items()},
            "range_score": round(score, 3),
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
            default_signal="monitor",
            default_confidence=55.0,
            default_reasoning="Defaulted to monitor after observation-only fallback.",
        )

        metrics = dict(observations["range_features"])
        metrics.update(
            {
                "score": round(score, 3),
                "existing_short_shares": existing_short,
            }
        )

        payload: dict[str, Any] = {
            "signal": decision.signal,
            "score": round(score, 3),
            "confidence": int(max(0, min(round(decision.confidence), 100))),
            "reasoning": decision.reasoning,
            "metrics": metrics,
            "constraints": decision.constraints or {},
            "meta": {"observations": observations},
        }

        sentinel_signals[ticker] = payload
        progress.update_status(
            agent_id,
            ticker,
            f"{payload['signal'].upper()} @ {payload['confidence']}/100 | score {score:.2f}",
        )

    message = HumanMessage(content=json.dumps(sentinel_signals), name=agent_id)

    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(sentinel_signals, "Range Recovery Sentinel")

    state.setdefault("data", {}).setdefault("analyst_signals", {})[agent_id] = sentinel_signals
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state.get("messages", []) + [message],
        "data": state["data"],
    }


__all__ = ["range_recovery_sentinel_agent"]
