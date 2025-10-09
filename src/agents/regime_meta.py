"""Meta-level regime classifier nudging risk overrides toward prevailing trends."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.agents.persona_utils import persona_from_observations
from src.tools.api import get_prices, prices_to_df
from src.utils.api_key import get_api_key_from_state
from src.utils.progress import progress


@dataclass(slots=True)
class FeatureSnapshot:
    close: float
    vol_30: float
    vol_60: float
    volatility_slope: float
    drawdown_depth: float
    return5_skew20: float
    momentum_20: float
    momentum_60: float


@dataclass(slots=True)
class AnalystSignals:
    prob_up: float | None
    base_rate: float
    delta_prob_up: float
    prob_revert_up: float | None
    revert_delta: float


@dataclass(slots=True)
class CalibrationWeights:
    classes: tuple[str, ...]
    feature_order: tuple[str, ...]
    means: np.ndarray
    stds: np.ndarray
    weights: np.ndarray
    bias: np.ndarray


CALIBRATION_PATH = Path("configs/regime_meta_weights.json")

PERSONA_NAME = "Morgan"
PERSONA_ROLE = "a regime cartographer who blends quantitative classes with on-desk intuition"
PERSONA_BACKSTORY = (
    "Morgan interpreted macro regimes for multi-asset desks and now decides whether the book should lean rally, crash, or consolidation."
)
PERSONA_INSTRUCTIONS = (
    "Use the probabilities and feature snapshot to judge the dominant regime.",
    "You may override the model recommendation when qualitative context or conflicting signals warrant it.",
    "Speak in first person and cite the key probabilities/metrics that shaped your call.",
)
ALLOWED_SIGNALS = ["rally", "crash", "consolidation", "neutral"]

EPSILON = 1e-6
DOWNSIDE_FLOW_BOOST = 0.08
DOWNSIDE_VOL_SLOPE_THRESHOLD = 0.015
DOWNSIDE_DRAWDOWN_THRESHOLD = -0.05
HISTORICAL_LOOKBACK_DAYS = 120


@lru_cache(maxsize=1)
def _downside_flow_boost_scale() -> float:
    value = os.getenv("REGIME_META_DOWNSIDE_BOOST")
    if value in (None, ""):
        return DOWNSIDE_FLOW_BOOST
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return DOWNSIDE_FLOW_BOOST
    if not np.isfinite(parsed):
        return DOWNSIDE_FLOW_BOOST
    return max(0.0, parsed)


def _extend_start_date(start_date: str | None, lookback: int) -> str | None:
    if not start_date:
        return None
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        return start_date
    return (start_dt - timedelta(days=lookback)).strftime("%Y-%m-%d")


def _sigmoid(value: float) -> float:
    clipped = float(np.clip(value, -8.0, 8.0))
    return 1.0 / (1.0 + float(np.exp(-clipped)))


def _prepare_snapshot(df: pd.DataFrame) -> FeatureSnapshot | None:
    if df.empty or len(df) < 70:
        return None

    closes = df["close"].astype(float)
    returns = closes.pct_change().fillna(0.0)

    vol_30 = returns.rolling(window=30).std(ddof=0) * np.sqrt(252.0)
    vol_60 = returns.rolling(window=60).std(ddof=0) * np.sqrt(252.0)

    if vol_30.isna().iloc[-1] or vol_60.isna().iloc[-1]:
        return None

    rolling_high_60 = closes.rolling(window=60).max()
    if rolling_high_60.isna().iloc[-1]:
        return None

    drawdown_depth = float(closes.iloc[-1] / rolling_high_60.iloc[-1] - 1.0)

    returns_5 = closes.pct_change(periods=5)
    return5_skew20 = returns_5.rolling(window=20).skew().iloc[-1]
    if pd.isna(return5_skew20):
        return5_skew20 = 0.0

    momentum_20 = closes.pct_change(20).iloc[-1]
    momentum_60 = closes.pct_change(60).iloc[-1]
    if pd.isna(momentum_20):
        momentum_20 = 0.0
    if pd.isna(momentum_60):
        momentum_60 = 0.0

    snapshot = FeatureSnapshot(
        close=float(closes.iloc[-1]),
        vol_30=float(vol_30.iloc[-1]),
        vol_60=float(vol_60.iloc[-1]),
        volatility_slope=float(vol_30.iloc[-1] - vol_60.iloc[-1]),
        drawdown_depth=drawdown_depth,
        return5_skew20=float(return5_skew20),
        momentum_20=float(momentum_20),
        momentum_60=float(momentum_60),
    )
    return snapshot


def _softmax_vector(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    denominator = np.sum(exp)
    if denominator <= 0:
        return np.full_like(exp, 1.0 / len(exp))
    return exp / denominator


@lru_cache(maxsize=1)
def _load_calibration() -> CalibrationWeights | None:
    try:
        payload = json.loads(CALIBRATION_PATH.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None

    classes = payload.get("classes") or []
    feature_order = payload.get("feature_order") or []
    if not classes or not feature_order:
        return None

    weights_blob = payload.get("weights") or {}
    bias_blob = payload.get("bias") or {}
    if any(cls not in weights_blob or cls not in bias_blob for cls in classes):
        return None

    means_blob = payload.get("feature_means") or {}
    stds_blob = payload.get("feature_stds") or {}

    try:
        means = np.array([float(means_blob[name]) for name in feature_order], dtype=float)
        stds = np.array([max(float(stds_blob[name]), EPSILON) for name in feature_order], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None

    try:
        weights = np.array([weights_blob[cls] for cls in classes], dtype=float)
        bias = np.array([bias_blob[cls] for cls in classes], dtype=float)
    except (TypeError, ValueError):
        return None

    if weights.shape[1] != len(feature_order):
        return None

    ordered_classes = tuple(str(cls) for cls in classes)
    ordered_features = tuple(str(name) for name in feature_order)

    return CalibrationWeights(
        classes=ordered_classes,
        feature_order=ordered_features,
        means=means,
        stds=stds,
        weights=weights,
        bias=bias,
    )


def _snapshot_feature_map(snapshot: FeatureSnapshot) -> dict[str, float]:
    vol_ratio = snapshot.vol_30 / max(snapshot.vol_60, EPSILON) - 1.0
    return {
        "volatility_slope": snapshot.volatility_slope,
        "drawdown_depth": snapshot.drawdown_depth,
        "return5_skew20": snapshot.return5_skew20,
        "momentum_20": snapshot.momentum_20,
        "momentum_60": snapshot.momentum_60,
        "vol_30": snapshot.vol_30,
        "vol_60": snapshot.vol_60,
        "vol_ratio": vol_ratio,
    }


def _collect_analyst_signals(
    growth_payload: Mapping[str, Any],
    mean_rev_payload: Mapping[str, Any],
) -> AnalystSignals:
    growth_indicators = growth_payload.get("indicators") or {}
    prob_up_raw = growth_indicators.get("prob_up")
    base_rate_raw = growth_indicators.get("base_rate", 0.5)

    try:
        base_rate = float(base_rate_raw)
    except (TypeError, ValueError):
        base_rate = 0.5

    prob_up: float | None
    try:
        prob_up = float(prob_up_raw) if prob_up_raw is not None else None
    except (TypeError, ValueError):
        prob_up = None

    delta_prob_up = float(prob_up - base_rate) if prob_up is not None else 0.0

    mean_rev_indicators = mean_rev_payload.get("indicators") or {}
    prob_revert_up_raw = mean_rev_indicators.get("prob_revert_up")
    try:
        prob_revert_up = (
            float(prob_revert_up_raw) if prob_revert_up_raw is not None else None
        )
    except (TypeError, ValueError):
        prob_revert_up = None

    revert_delta = float(prob_revert_up - 0.5) if prob_revert_up is not None else 0.0

    return AnalystSignals(
        prob_up=prob_up,
        base_rate=base_rate,
        delta_prob_up=delta_prob_up,
        prob_revert_up=prob_revert_up,
        revert_delta=revert_delta,
    )


def _downside_stage_and_score(
    payload: Mapping[str, Any] | None,
) -> tuple[str, float]:
    if not isinstance(payload, Mapping):
        return "", 0.0

    stage = str(payload.get("signal") or payload.get("stage") or "").lower()
    score_raw = payload.get("score")
    if score_raw is None:
        metrics = payload.get("metrics")
        if isinstance(metrics, Mapping):
            score_raw = metrics.get("score")

    try:
        severity = float(score_raw)
    except (TypeError, ValueError):
        severity = 0.0

    if not np.isfinite(severity):
        severity = 0.0

    return stage, float(np.clip(severity, 0.0, 1.0))


def _probabilities_data_driven(snapshot: FeatureSnapshot) -> dict[str, float] | None:
    calibration = _load_calibration()
    if calibration is None:
        return None

    feature_map = _snapshot_feature_map(snapshot)
    try:
        vector = np.array(
            [float(feature_map[name]) for name in calibration.feature_order],
            dtype=float,
        )
    except KeyError:
        return None
    if not np.all(np.isfinite(vector)):
        return None

    normalised = (vector - calibration.means) / calibration.stds
    logits = calibration.weights @ normalised + calibration.bias
    probabilities = _softmax_vector(logits)
    return {cls: float(prob) for cls, prob in zip(calibration.classes, probabilities)}


def _apply_external_adjustments(
    base_probs: dict[str, float],
    signals: AnalystSignals,
    snapshot: FeatureSnapshot,
    downside_payload: Mapping[str, Any] | None,
) -> dict[str, float]:
    adjusted = {
        "rally": float(base_probs.get("rally", 0.0)),
        "crash": float(base_probs.get("crash", 0.0)),
        "consolidation": float(base_probs.get("consolidation", base_probs.get("neutral", 0.0))),
    }

    adjusted["rally"] += 0.08 * max(0.0, signals.delta_prob_up)
    adjusted["crash"] += 0.08 * max(0.0, -signals.delta_prob_up)

    adjusted["rally"] += 0.06 * max(0.0, signals.revert_delta)
    adjusted["crash"] += 0.06 * max(0.0, -signals.revert_delta)

    floor = 1e-6

    stage, severity = _downside_stage_and_score(downside_payload)
    boost_scale = _downside_flow_boost_scale()
    volatility_alignment = snapshot.volatility_slope >= DOWNSIDE_VOL_SLOPE_THRESHOLD
    drawdown_alignment = snapshot.drawdown_depth <= DOWNSIDE_DRAWDOWN_THRESHOLD
    sentinel_alignment = stage in {"downside_trend", "crash_flow"}
    if (
        sentinel_alignment
        and volatility_alignment
        and drawdown_alignment
        and severity > 0
        and boost_scale > 0
    ):
        boost = boost_scale * severity
        adjusted["crash"] += boost
        bleed = boost * 0.5
        adjusted["rally"] = max(floor, adjusted["rally"] - bleed)
        adjusted["consolidation"] = max(floor, adjusted["consolidation"] - bleed)

    for key in adjusted:
        adjusted[key] = max(floor, adjusted[key])

    return adjusted


def _probabilities_manual(snapshot: FeatureSnapshot, signals: AnalystSignals) -> dict[str, float]:
    volatility_ratio = snapshot.vol_30 / max(snapshot.vol_60, EPSILON) - 1.0
    drawdown_penalty = max(0.0, -snapshot.drawdown_depth - 0.05)
    skew_penalty = max(0.0, snapshot.return5_skew20)

    rally_input = (
        5.2 * snapshot.momentum_20
        + 3.6 * snapshot.momentum_60
        + 2.3 * max(0.0, -snapshot.volatility_slope)
        + 2.0 * max(0.0, -volatility_ratio)
        + 1.8 * max(0.0, -snapshot.drawdown_depth)
        + 2.1 * max(0.0, signals.delta_prob_up)
        + 1.4 * max(0.0, signals.revert_delta)
        - 1.2 * skew_penalty
    )

    crash_input = (
        4.8 * max(0.0, -snapshot.momentum_20)
        + 3.4 * max(0.0, -snapshot.momentum_60)
        + 2.7 * max(0.0, snapshot.volatility_slope)
        + 2.2 * max(0.0, volatility_ratio)
        + 2.5 * drawdown_penalty
        + 1.3 * max(0.0, -signals.delta_prob_up)
        + 1.1 * max(0.0, -signals.revert_delta)
        + 1.0 * skew_penalty
    )

    rally_prob = _sigmoid(rally_input)
    crash_prob = _sigmoid(crash_input)

    neutral_input = 2.0 - abs(rally_input - crash_input)
    neutral_prob = _sigmoid(neutral_input) * 0.5

    total = rally_prob + crash_prob + neutral_prob
    if total <= 0:
        return {"rally": 1.0 / 3.0, "crash": 1.0 / 3.0, "consolidation": 1.0 / 3.0}

    return {
        "rally": float(rally_prob / total),
        "crash": float(crash_prob / total),
        "consolidation": float(neutral_prob / total),
    }


def _blend_probabilities(
    data_driven: dict[str, float],
    manual: dict[str, float],
) -> dict[str, float]:
    blended: dict[str, float] = {}
    for regime in ("rally", "crash", "consolidation"):
        blended[regime] = 0.6 * data_driven.get(regime, 0.0) + 0.4 * manual.get(regime, 0.0)

    top_manual = sorted(manual.items(), key=lambda kv: kv[1], reverse=True)
    if len(top_manual) >= 2:
        primary_label, primary_score = top_manual[0]
        secondary_score = top_manual[1][1]
        manual_edge = primary_score - secondary_score
        blended_leader = max(blended, key=blended.get)
        if primary_label != blended_leader and manual_edge >= 0.08:
            boost = min(0.25, manual_edge * 0.6)
            blended[primary_label] = blended.get(primary_label, 0.0) + boost
            for label in blended:
                if label != primary_label:
                    blended[label] = max(0.0, blended[label] - boost / 2.0)

        consolidation_score = manual.get("consolidation", 0.0)
        directional_gap = primary_score - consolidation_score
        if directional_gap >= 0.15 and blended.get("consolidation", 0.0) > 0:
            reduction = min(blended.get("consolidation", 0.0), directional_gap * 0.5)
            blended[primary_label] = blended.get(primary_label, 0.0) + reduction
            blended["consolidation"] = max(0.0, blended.get("consolidation", 0.0) - reduction)

    total = sum(blended.values())
    if total <= 0 or not np.isfinite(total):
        return manual
    return {key: value / total for key, value in blended.items()}


def _probabilities(
    snapshot: FeatureSnapshot,
    growth_payload: dict[str, Any],
    mean_rev_payload: dict[str, Any],
    downside_payload: Mapping[str, Any] | None,
) -> dict[str, float]:
    signals = _collect_analyst_signals(growth_payload, mean_rev_payload)
    manual = _probabilities_manual(snapshot, signals)

    data_driven = _probabilities_data_driven(snapshot)
    if data_driven is None:
        return manual

    adjusted = _apply_external_adjustments(data_driven, signals, snapshot, downside_payload)
    return _blend_probabilities(adjusted, manual)


def _build_constraints(
    signal: str,
    probabilities: dict[str, float],
    snapshot: FeatureSnapshot,
    portfolio: dict[str, Any],
    ticker: str,
) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    portfolio_cash = float(portfolio.get("cash", 0.0) or 0.0)
    positions = portfolio.get("positions", {}) or {}
    ticker_position = positions.get(ticker, {}) or {}
    current_long = int(ticker_position.get("long", 0) or 0)

    rally_strength = probabilities.get("rally", 0.0) - probabilities.get("crash", 0.0)

    if signal == "rally":
        constraints["preferred_direction"] = "long"
        constraints["max_short_exposure_pct"] = 0.05

        if snapshot.close > 0 and portfolio_cash > 0:
            target_pct = min(0.4, max(0.1, rally_strength * 1.5))
            target_cash = portfolio_cash * target_pct
            add_shares = int(max(0.0, target_cash // snapshot.close))
            target_total = current_long + add_shares
            if target_total > current_long:
                constraints["target_long_shares"] = target_total

    elif signal == "crash":
        constraints["preferred_direction"] = "short"
        constraints["max_short_exposure_pct"] = min(0.3, 0.15 + probabilities.get("crash", 0.0) * 0.3)
    else:
        constraints["max_short_exposure_pct"] = 0.1

    return constraints


##### Regime Meta-Model Analyst #####
def regime_meta_agent(state: AgentState, agent_id: str = "regime_meta_agent"):
    """Lightweight regime classifier to steer risk overrides via deterministic signals."""

    data = state.get("data", {})
    tickers = data.get("tickers", [])
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    portfolio = data.get("portfolio", {})
    analyst_signals = data.setdefault("analyst_signals", {})

    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    signals: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching price history")

        fetch_start = _extend_start_date(start_date, HISTORICAL_LOOKBACK_DAYS)

        prices = get_prices(
            ticker=ticker,
            start_date=fetch_start or start_date,
            end_date=end_date,
            api_key=api_key,
        )

        if not prices:
            signals[ticker] = {
                "signal": "neutral",
                "confidence": 45,
                "reasoning": "Missing price history for regime inference.",
            }
            continue

        df = prices_to_df(prices)
        snapshot = _prepare_snapshot(df)
        if snapshot is None:
            signals[ticker] = {
                "signal": "neutral",
                "confidence": 50,
                "reasoning": "Insufficient observations for volatility regime features.",
            }
            continue

        growth_payload = (analyst_signals.get("growth_momentum_agent") or {}).get(ticker, {})
        mean_rev_payload = (analyst_signals.get("stat_mean_reversion_agent") or {}).get(ticker, {})
        downside_payload_raw = (
            (analyst_signals.get("downside_flow_sentinel_agent") or {}).get(ticker)
        )
        downside_payload = downside_payload_raw if isinstance(downside_payload_raw, Mapping) else None

        probs = _probabilities(snapshot, growth_payload, mean_rev_payload, downside_payload)

        signal = max(probs, key=probs.get)
        confidence = int(round(100 * probs[signal]))

        constraints = _build_constraints(signal, probs, snapshot, portfolio, ticker)

        reasoning = f"Rally {probs['rally']:.0%} vs crash {probs['crash']:.0%}; " f"mom20 {snapshot.momentum_20:.1%}, drawdown {snapshot.drawdown_depth:.1%}, " f"vol slope {snapshot.volatility_slope:.3f}."

        indicators = {
            "close": snapshot.close,
            "vol_30": snapshot.vol_30,
            "vol_60": snapshot.vol_60,
            "volatility_slope": snapshot.volatility_slope,
            "drawdown_depth": snapshot.drawdown_depth,
            "return5_skew20": snapshot.return5_skew20,
            "momentum_20": snapshot.momentum_20,
            "momentum_60": snapshot.momentum_60,
            "probabilities": probs,
        }

        auto_confidence = max(0, min(confidence, 100))
        auto_constraints = dict(constraints) if constraints else {}
        payload: dict[str, Any] = {
            "signal": signal,
            "confidence": auto_confidence,
            "reasoning": reasoning,
            "indicators": indicators,
            "auto_reasoning": reasoning,
            "constraints": auto_constraints,
        }

        observations = {
            "ticker": ticker,
            "probabilities": probs,
            "snapshot": {
                "close": snapshot.close,
                "vol_30": snapshot.vol_30,
                "vol_60": snapshot.vol_60,
                "volatility_slope": snapshot.volatility_slope,
                "drawdown_depth": snapshot.drawdown_depth,
                "return5_skew20": snapshot.return5_skew20,
                "momentum_20": snapshot.momentum_20,
                "momentum_60": snapshot.momentum_60,
            },
            "suggested_constraints": auto_constraints,
            "auto_signal_hint": signal,
            "auto_confidence_hint": auto_confidence,
            "auto_reasoning": reasoning,
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

        payload.update(
            {
                "signal": decision.signal,
                "confidence": int(max(0, min(round(decision.confidence), 100))),
                "reasoning": decision.reasoning,
                "constraints": decision.constraints or {},
            }
        )
        payload.setdefault("meta", {})["observations"] = observations

        progress.update_status(agent_id, ticker, f"{decision.signal.upper()} @ {payload['confidence']}/100 | rally edge {probs['rally'] - probs['crash']:.2f}")

        signals[ticker] = payload

    message = HumanMessage(content=json.dumps(signals), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(signals, "Regime Meta-Model")

    state["data"].setdefault("analyst_signals", {})[agent_id] = signals
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }
