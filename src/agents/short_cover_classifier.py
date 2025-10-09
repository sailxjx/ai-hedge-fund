"""Classifier-based sentinel that unwinds shorts when upside squeeze risk spikes."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.agents.growth_momentum import _logistic_regression as _fit_logistic
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices, prices_to_df
from src.utils.api_key import get_api_key_from_state
from src.utils.calibration import apply_platt, get_platt_parameters
from src.utils.llm import call_llm
from src.utils.progress import progress

BUFFER_DAYS = 220
MIN_OBSERVATIONS = 120
MIN_TRAINING_SAMPLES = 60
LABEL_QUANTILE = 0.80
TRIGGER_QUANTILE = 0.70
TRIGGER_LEEWAY = 0.01
IMPROVEMENT_SIGMA = 0.05
MIN_IMPROVEMENT = 0.0015
NEAR_THRESHOLD_BUFFER = 0.025  # 2.5 percentage points gap
NEAR_THRESHOLD_IMPROVEMENT_SCALE = 0.25
NEAR_THRESHOLD_TRIM_RATIO = 0.35
BOOST_ESCALATION_MARGIN = 0.018  # 1.8 percentage point tolerance for raw probability
BOOST_CAUTIOUS_TRIM_RATIO = 0.45
BIAS_LOCK_MARGIN = 0.035  # Require a 3.5pp edge above base rate to hard block
SOFT_BIAS_MARGIN = 0.015  # Treat smaller edges as guidance only
PRECISION_WINDOW_LEEWAY = 0.012
PRECISION_MIN_SAMPLES = 12
PRECISION_MIN_PRECISION = 0.6
PRECISION_THRESHOLD_SHIFT_SCALE = 0.18
PRECISION_THRESHOLD_SHIFT_MAX = 0.02
PRECISION_MARGIN_SCALE = 0.12
PRECISION_MARGIN_MAX = 0.018
PRECISION_STREAK_WINDOW = 5
PRECISION_STREAK_TRIGGER = 3
PRECISION_STREAK_MARGIN_DECAY = 0.9
PRECISION_STREAK_MARGIN_FLOOR = 0.004
PRECISION_STREAK_MARGIN_MIN_RATIO = 0.6
PRECISION_STREAK_SHIFT_SCALE = 0.5
PRECISION_STREAK_SHIFT_STEP = 0.003
PRECISION_STREAK_SHIFT_RECOVERY = 0.0025
PRECISION_IMPROVEMENT_DISCOUNT_SCALE = 1.6
PRECISION_IMPROVEMENT_DISCOUNT_MAX = 0.45

PRECISION_IMPROVEMENT_DISCOUNT_MIN_FACTOR = 0.35
PRECISION_IMPROVEMENT_DISCOUNT_TREND_GAP_WEIGHT = 6.0
PRECISION_IMPROVEMENT_DISCOUNT_TREND_SLOPE_WEIGHT = 9.0
PRECISION_IMPROVEMENT_DISCOUNT_TREND_BREAKOUT_WEIGHT = 7.0
PRECISION_IMPROVEMENT_DISCOUNT_TREND_CAP = 0.65

PRECISION_STREAK_IMPROVEMENT_DISCOUNT_SCALE = 0.15
PRECISION_STREAK_IMPROVEMENT_DISCOUNT_MAX = 0.25

PRECISION_STREAK_TRIM_SCALE = 0.65
PRECISION_STREAK_TRIM_CAP = 0.98

LONG_BIAS_NEUTRALIZE_MARGIN = 0.012

BOOST_ONLY_NEW_SHORT_UNBLOCK_MARGIN = 0.015
NEAR_THRESHOLD_RAW_GAP_LIMIT = 0.028
NEAR_THRESHOLD_CLAMP_FACTOR_LIMIT = 0.85
NEUTRAL_GUARDRAIL_STREAK_RELAX = 3

BOOST_CLAMP_MIN_FACTOR = 0.35
BOOST_CLAMP_MAX_PENALTY = 0.65
BOOST_VOLATILITY_CLAMP_START = 0.10
BOOST_VOLATILITY_CLAMP_WEIGHT = 1.2
BOOST_VOLUME_CLAMP_START = 0.4
BOOST_VOLUME_CLAMP_WEIGHT = 0.25
BOOST_ATR_CLAMP_START = 0.018
BOOST_ATR_CLAMP_WEIGHT = 3.5
BOOST_THRESHOLD_GAP_CLAMP_START = 0.015
BOOST_THRESHOLD_GAP_CLAMP_WEIGHT = 8.0

BOOST_RECLASSIFY_RAW_GAP = 0.035
BOOST_RECLASSIFY_MIN_IMPROVEMENT_MARGIN = 0.0015
BOOST_RECLASSIFY_MIN_IMPROVEMENT_FACTOR = 1.7


PRECISION_TICKER_OVERRIDES: dict[str, dict[str, float]] = {
    "NVDA": {
        "min_precision": 0.26,
        "threshold_shift_scale": 0.22,
        "threshold_shift_cap": 0.028,
        "margin_scale": 0.15,
        "margin_cap": 0.022,
        "streak_shift_scale": 0.6,
        "streak_shift_step": 0.004,
        "streak_shift_recovery": 0.003,
        "streak_margin_decay": 0.85,
        "streak_margin_min_ratio": 0.55,
        "streak_margin_floor": 0.005,
        "improvement_discount_scale": 1.8,
        "improvement_discount_cap": 0.5,
        "improvement_discount_min_factor": 0.3,
        "improvement_discount_trend_cap": 0.55,
        "streak_improvement_discount_scale": 0.2,
        "streak_improvement_discount_cap": 0.3,
        "streak_trim_scale": 0.75,
        "streak_trim_cap": 0.995,
    },
}


_PRECISION_CONSTANT_BASELINE = {
    "PRECISION_THRESHOLD_SHIFT_SCALE": PRECISION_THRESHOLD_SHIFT_SCALE,
    "PRECISION_THRESHOLD_SHIFT_MAX": PRECISION_THRESHOLD_SHIFT_MAX,
    "PRECISION_MARGIN_SCALE": PRECISION_MARGIN_SCALE,
    "PRECISION_MARGIN_MAX": PRECISION_MARGIN_MAX,
    "PRECISION_STREAK_SHIFT_SCALE": PRECISION_STREAK_SHIFT_SCALE,
    "PRECISION_STREAK_SHIFT_STEP": PRECISION_STREAK_SHIFT_STEP,
    "PRECISION_STREAK_SHIFT_RECOVERY": PRECISION_STREAK_SHIFT_RECOVERY,
    "PRECISION_STREAK_MARGIN_DECAY": PRECISION_STREAK_MARGIN_DECAY,
    "PRECISION_STREAK_MARGIN_MIN_RATIO": PRECISION_STREAK_MARGIN_MIN_RATIO,
    "PRECISION_STREAK_MARGIN_FLOOR": PRECISION_STREAK_MARGIN_FLOOR,
    "PRECISION_IMPROVEMENT_DISCOUNT_SCALE": PRECISION_IMPROVEMENT_DISCOUNT_SCALE,
    "PRECISION_IMPROVEMENT_DISCOUNT_MAX": PRECISION_IMPROVEMENT_DISCOUNT_MAX,
    "PRECISION_IMPROVEMENT_DISCOUNT_MIN_FACTOR": PRECISION_IMPROVEMENT_DISCOUNT_MIN_FACTOR,
    "PRECISION_IMPROVEMENT_DISCOUNT_TREND_CAP": PRECISION_IMPROVEMENT_DISCOUNT_TREND_CAP,
    "PRECISION_STREAK_IMPROVEMENT_DISCOUNT_SCALE": PRECISION_STREAK_IMPROVEMENT_DISCOUNT_SCALE,
    "PRECISION_STREAK_IMPROVEMENT_DISCOUNT_MAX": PRECISION_STREAK_IMPROVEMENT_DISCOUNT_MAX,
    "PRECISION_STREAK_TRIM_SCALE": PRECISION_STREAK_TRIM_SCALE,
    "PRECISION_STREAK_TRIM_CAP": PRECISION_STREAK_TRIM_CAP,
}


class ShortCoverDecision(BaseModel):
    signal: Literal["neutral", "squeeze_risk", "squeeze_cover", "trim_short", "bias_long"]
    confidence: float
    reasoning: str
    constraints: Optional[dict[str, Any]] = None


def _to_builtin(value: Any) -> Any:
    """Convert numpy/pandas scalars into built-in Python types for JSON serialization."""

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]

    return value


def _as_real(value: Any) -> float | None:
    """Best-effort conversion to a float, excluding booleans."""

    if isinstance(value, bool):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_pct(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.1%}"


def _scale_trim_ratio_for_precision(
    trim_ratio_base: float,
    precision_override_active: bool,
    precision_boost_trim_streak: int,
    precision_streak_trigger: int,
    precision_streak_trim_scale: float,
    precision_streak_trim_cap: float,
) -> tuple[float, dict[str, Any]]:
    """Adjust the trim ratio when precision streak overrides are active."""

    metrics = {
        "precision_streak_trim_active": False,
        "precision_streak_trim_factor": 0,
        "precision_streak_trim_multiplier": 1.0,
        "precision_streak_trim_cap": precision_streak_trim_cap,
    }

    trim_ratio = max(trim_ratio_base, NEAR_THRESHOLD_TRIM_RATIO)

    if precision_override_active and precision_boost_trim_streak >= precision_streak_trigger:
        streak_factor = precision_boost_trim_streak - precision_streak_trigger + 1
        multiplier = 1.0 + precision_streak_trim_scale * streak_factor
        trim_ratio = min(
            precision_streak_trim_cap,
            max(trim_ratio_base * multiplier, NEAR_THRESHOLD_TRIM_RATIO),
        )
        metrics["precision_streak_trim_active"] = True
        metrics["precision_streak_trim_factor"] = streak_factor
        if trim_ratio_base > 0:
            metrics["precision_streak_trim_multiplier"] = trim_ratio / trim_ratio_base
        else:
            metrics["precision_streak_trim_multiplier"] = multiplier

    return trim_ratio, metrics


def _assess_neutral_guardrail(metrics: dict[str, Any], existing_short: int) -> tuple[bool, list[str]]:
    """Determine whether the persona should default to neutral and record reasons."""

    probability = _as_real(metrics.get("probability"))
    raw_probability = _as_real(metrics.get("raw_probability"))
    threshold = _as_real(metrics.get("threshold"))
    improvement = _as_real(metrics.get("improvement"))
    improvement_threshold_effective = _as_real(metrics.get("improvement_threshold_effective"))
    boost_only = bool(metrics.get("boost_driven_squeeze") or metrics.get("boost_driven_soft") or metrics.get("caution_trigger") == "boost_only")
    precision_boost_trim_streak = int(metrics.get("precision_boost_trim_streak") or 0)

    prob_gap = None
    raw_gap = None
    if probability is not None and threshold is not None:
        prob_gap = probability - threshold
    if raw_probability is not None and threshold is not None:
        raw_gap = raw_probability - threshold

    improvement_gap = None
    if improvement is not None and improvement_threshold_effective is not None:
        improvement_gap = improvement - improvement_threshold_effective

    guardrail = False
    reasons: list[str] = []
    relaxed = False
    relaxed_reasons: list[str] = []

    if existing_short > 0:
        relaxed = True
        relaxed_reasons.append("existing_short_inventory")
    elif boost_only and precision_boost_trim_streak >= NEUTRAL_GUARDRAIL_STREAK_RELAX:
        relaxed = True
        relaxed_reasons.append("precision_boost_streak")

    if existing_short <= 0 and not relaxed:
        if boost_only:
            guardrail = True
            reasons.append("boost-only probabilities")
        if raw_gap is not None and raw_gap < 0:
            guardrail = True
            reasons.append("raw probability below trigger")
        if prob_gap is not None and abs(prob_gap) <= NEAR_THRESHOLD_BUFFER:
            guardrail = True
            reasons.append("marginal probability gap")
        if improvement_gap is not None and improvement_gap < 0:
            guardrail = True
            reasons.append("calibration-improved edge")

    metrics["neutral_guardrail"] = guardrail
    if reasons:
        metrics["neutral_guardrail_reasons"] = reasons
    else:
        metrics.pop("neutral_guardrail_reasons", None)

    metrics["neutral_guardrail_relaxed"] = relaxed
    if relaxed_reasons:
        metrics["neutral_guardrail_relaxed_reasons"] = relaxed_reasons
    else:
        metrics.pop("neutral_guardrail_relaxed_reasons", None)

    return guardrail, reasons


def _build_persona_notes(metrics: dict[str, Any], existing_short: int) -> list[str]:
    """Provide human-readable cues that help the persona weigh soft cautions."""

    notes: list[str] = []

    caution_trigger = metrics.get("caution_trigger")
    raw_probability = _as_real(metrics.get("raw_probability"))
    probability = _as_real(metrics.get("probability"))
    threshold = _as_real(metrics.get("threshold"))
    soft_threshold = _as_real(metrics.get("soft_threshold"))
    probability_boost = _as_real(metrics.get("probability_boost"))
    improvement = _as_real(metrics.get("improvement"))
    improvement_threshold = _as_real(metrics.get("improvement_threshold"))
    improvement_threshold_effective = _as_real(metrics.get("improvement_threshold_effective"))

    boost_only = bool(metrics.get("boost_driven_squeeze") or metrics.get("boost_driven_soft") or caution_trigger == "boost_only")
    neutral_guardrail_relaxed = bool(metrics.get("neutral_guardrail_relaxed"))
    neutral_guardrail_relaxed_reasons = metrics.get("neutral_guardrail_relaxed_reasons", [])

    def _format_relaxed_reasons() -> str:
        reason_map = {
            "existing_short_inventory": "short inventory still on the book",
            "precision_boost_streak": "repeated boost-only threshold beats",
        }
        formatted = [reason_map.get(reason, reason.replace("_", " ")) for reason in neutral_guardrail_relaxed_reasons]
        return ", ".join(formatted)

    if boost_only:
        raw_gap = None
        if raw_probability is not None and threshold is not None:
            raw_gap = raw_probability - threshold
        gap_text = f"gap {raw_gap:+.1%}" if raw_gap is not None else "below trigger"
        boost_text = f"boost applied {probability_boost:.1%}" if probability_boost is not None else "boost-only signal"
        if existing_short > 0:
            action_text = "Use trim_short to bleed exposure; avoid squeeze_risk or full covers unless catalysts confirm a true squeeze."
        elif neutral_guardrail_relaxed:
            relaxed_text = _format_relaxed_reasons() or "guardrail relaxed"
            action_text = f"Guardrail relaxed ({relaxed_text}); surface squeeze_risk or bias_long when catalysts support it and document the justification."
        else:
            action_text = "Remain neutral and do not emit squeeze_risk—there is no short on, so bias_long requires explicit catalysts."
        notes.append(f"Boost-only caution: raw probability {_format_pct(raw_probability)} vs trigger {_format_pct(threshold)} ({gap_text}); {boost_text}. {action_text}")

    if caution_trigger == "near_threshold":
        soft_gap = None
        if probability is not None and soft_threshold is not None:
            soft_gap = probability - soft_threshold
        soft_gap_text = f"soft gap {soft_gap:+.1%}" if soft_gap is not None else "soft threshold not engaged"
        if existing_short > 0:
            action_text = "Prefer trim_short to lighten the position; escalate to squeeze_cover only when catalysts prove decisive."
        else:
            action_text = "Stay neutral and skip squeeze_risk—there is no short to defend, so long bias still needs external support."
        notes.append(f"Near-threshold caution: squeeze odds are marginal (prob {_format_pct(probability)} vs trigger {_format_pct(threshold)}; {soft_gap_text}). {action_text}")

    if improvement_threshold is not None and improvement_threshold_effective is not None and improvement is not None and improvement < improvement_threshold and improvement >= improvement_threshold_effective:
        notes.append("Improvement discount engaged: raw expected edge is below the base hurdle; soft override provided " "by calibration.")

    if metrics.get("precision_override_active"):
        notes.append("Precision override is active; focus on whether recent covers succeeded before enforcing bias.")

    if existing_short > 0:
        notes.append("Short exposure is still on the book; lead with trim_short to reduce risk and only escalate to squeeze_cover when metrics are overwhelming.")
    else:
        notes.append("No short exposure is open; stay neutral unless you can cite independent catalysts that justify leaning long. Squeeze_risk and squeeze_cover do not apply when the book is flat.")

    neutral_guardrail, guardrail_reasons = _assess_neutral_guardrail(metrics, existing_short)
    if neutral_guardrail:
        reason_text = ", ".join(guardrail_reasons) if guardrail_reasons else "soft signals"
        notes.append(f"NEUTRAL_GUARDRAIL: No short exposure is open and conditions hinge on {reason_text}. Hold neutral unless explicit catalysts in context justify a long bias; do not rely on boosted probabilities alone, and avoid squeeze_risk while the guardrail holds.")
    elif neutral_guardrail_relaxed:
        relaxed_text = _format_relaxed_reasons() or "guardrail conditions eased"
        notes.append(f"NEUTRAL_GUARDRAIL RELAXED: {relaxed_text}. Escalation is allowed when you can cite catalysts or analyst evidence supporting a squeeze or long bias—explain the basis if you deviate from neutral.")

    return notes


def _prepare_llm_context(
    ticker: str,
    existing_short: int,
    metrics: dict[str, Any],
    recommendation: dict[str, Any],
) -> dict[str, Any]:
    """Build a structured context payload for the short-cover persona."""

    persona_notes = _build_persona_notes(metrics, existing_short)
    sanitized_metrics = {k: _to_builtin(v) for k, v in metrics.items()}
    if persona_notes:
        metrics["persona_notes"] = persona_notes
        sanitized_metrics["persona_notes"] = persona_notes
    suggestion = {
        "signal_hint": recommendation["signal"],
        "confidence_hint": recommendation["confidence"],
        "reasoning_hint": recommendation["reasoning"],
        "constraints_hint": _to_builtin(recommendation.get("constraints") or {}),
    }

    guardrail_summary = {
        "neutral_guardrail_active": bool(metrics.get("neutral_guardrail")),
        "neutral_guardrail_reasons": metrics.get("neutral_guardrail_reasons", []),
        "neutral_guardrail_relaxed": bool(metrics.get("neutral_guardrail_relaxed")),
        "neutral_guardrail_relaxed_reasons": metrics.get("neutral_guardrail_relaxed_reasons", []),
    }

    return {
        "ticker": ticker,
        "existing_short_position": existing_short,
        "suggestion": suggestion,
        "metrics": sanitized_metrics,
        "guardrails": guardrail_summary,
    }


def _build_persona_prompt(context: dict[str, Any]):
    """Create the prompt instructing the anthropomorphic short-cover persona."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Selene, the Anthropomorphic Short-Squeeze Sentinel for our hedge fund.\n"
                "You specialize in assessing short squeeze risk, balancing downside catalysts with squeeze likelihood.\n"
                "Use the quantitative metrics as evidence, but make the final decision yourself.\n"
                "Signals you may return: neutral, bias_long, trim_short, squeeze_risk, squeeze_cover.\n"
                "If squeeze pressure is overwhelming and the book still holds shorts, prefer squeeze_cover.\n"
                "If pressure is rising but not decisive, begin with trim_short to bleed risk; escalate beyond neutral only when catalysts justify it.\n"
                "Consult metrics.persona_notes for quick context. Treat boost-only or near-threshold cautions as guardrails, not optional hints.\n"
                "When persona_notes reference boost-only or near-threshold cautions, follow their path: with a short on, lean trim_short; with no short, stay neutral unless guardrails.neutral_guardrail_relaxed is true.\n"
                "When raw probabilities remain below thresholds or improvements depend on calibration discounts, stay neutral unless catalysts warrant a bullish stance.\n"
                "Neutral is an active choice when evidence conflicts; do not force a bias when the context is inconclusive.\n"
                "If metrics.neutral_guardrail is true and existing_short_position == 0, you must return neutral. There are no exceptions—explain why the guardrail holds.\n"
                "If metrics.neutral_guardrail is true and existing_short_position > 0, you may only respond with trim_short or neutral unless the context quotes specific catalysts that justify escalation.\n"
                "Persona notes may list NEUTRAL_GUARDRAIL. When present, treat it as binding: default to neutral unless the context explicitly provides catalysts or analyst signals that justify breaking the guardrail.\n"
                "Breaking a neutral guardrail requires quoting the catalysts or analyst evidence verbatim from the context; if you cannot, respond neutral and document the absence of support.\n"
                "If guardrails.neutral_guardrail_relaxed is true, acknowledge the reason it was lifted and support any non-neutral stance with the cited catalysts or analyst evidence.\n"
                "Do not invent catalysts or analyst voices that are not present in the provided context—if they are absent, say so and remain neutral.\n"
                "If you lean long while probabilities lag the trigger, call out the gap and explain the catalyst that overrides it. Without that narrative, favor neutrality.\n"
                "When existing_short_position is zero, do not emit squeeze_risk or squeeze_cover; there is nothing to defend.\n"
                "Bias_long with no existing short demands concrete catalysts or corroborating analyst views beyond the squeeze probability, and it is invalid while metrics.neutral_guardrail is true.\n"
                "Always explain the rationale from Selene's perspective, referencing key metrics and catalysts.\n"
                "When recommending constraints, be explicit about whether new shorts are blocked, trims required, or force covers triggered.\n"
                "A suggestion block is provided as a quantitative hint; treat it as a reference only—you are not required to follow it.\n"
                "Return thoughtful judgement; you may disagree with the model recommendation when the narrative warrants it.\n""",
            ),
            (
                "human",
                """Context:\n{context}\n\n"
                "Respond with strict JSON:\n"
                "{{\n"
                "  \"signal\": \"neutral|bias_long|trim_short|squeeze_risk|squeeze_cover\",\n"
                "  \"confidence\": float (0-100),\n"
                "  \"reasoning\": \"string\",\n"
                "  \"constraints\": {{\"preferred_direction\": str, ...}} or null\n"
                "}}\n""",
            ),
        ]
    )

    return template.invoke({"context": json.dumps(context, indent=2, sort_keys=True)})


def _request_llm_decision(
    state: AgentState,
    agent_id: str,
    context: dict[str, Any],
) -> ShortCoverDecision:
    """Call the LLM persona; fall back to the model recommendation if it fails."""

    prompt = _build_persona_prompt(context)
    suggestion = context["suggestion"]

    def _default_decision() -> ShortCoverDecision:
        return ShortCoverDecision(
            signal=suggestion["signal_hint"],
            confidence=float(suggestion["confidence_hint"]),
            reasoning=suggestion["reasoning_hint"],
            constraints=suggestion.get("constraints_hint") or {},
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=ShortCoverDecision,
        agent_name=agent_id,
        state=state,
        default_factory=_default_decision,
    )


def _extend_start(start_date: str | None, buffer_days: int = BUFFER_DAYS) -> str | None:
    if not start_date:
        return None
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        return start_date
    return (start_dt - timedelta(days=buffer_days)).strftime("%Y-%m-%d")


def _prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    closes = df["close"].astype(float)
    volumes = df.get("volume", pd.Series(dtype=float)).astype(float)
    highs = df.get("high", pd.Series(dtype=float)).astype(float)
    lows = df.get("low", pd.Series(dtype=float)).astype(float)

    returns = closes.pct_change()
    ret_3 = closes.pct_change(3)
    ret_5 = closes.pct_change(5)
    ret_10 = closes.pct_change(10)

    ema21 = closes.ewm(span=21, adjust=False).mean()
    ema55 = closes.ewm(span=55, adjust=False).mean()
    ema_gap = closes / ema21.replace(0.0, np.nan) - 1.0
    ema_slope = ema21 / ema55.replace(0.0, np.nan) - 1.0

    rolling_high_20 = highs.rolling(window=20).max()
    rolling_low_20 = lows.rolling(window=20).min()
    breakout_gap = closes / rolling_high_20.replace(0.0, np.nan) - 1.0
    retrace_gap = closes / rolling_low_20.replace(0.0, np.nan) - 1.0

    vol_5 = returns.rolling(window=5).std(ddof=0)
    vol_20 = returns.rolling(window=20).std(ddof=0)
    vol_slope = vol_5 - vol_20
    vol_ratio = (vol_5 / vol_20.replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)

    true_range = (highs - lows).abs()
    atr_14 = true_range.rolling(window=14).mean()
    atr_ratio = (atr_14 / closes.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)

    if volumes.empty:
        volume_ratio = pd.Series(0.0, index=closes.index)
        volume_zscore = pd.Series(0.0, index=closes.index)
    else:
        volume_ma_10 = volumes.rolling(window=10).mean()
        volume_ratio = volumes / volume_ma_10.replace(0.0, np.nan) - 1.0

        volume_mean_20 = volumes.rolling(window=20).mean()
        volume_std_20 = volumes.rolling(window=20).std(ddof=0).replace(0.0, np.nan)
        volume_zscore = (volumes - volume_mean_20) / volume_std_20
        volume_zscore = volume_zscore.replace([np.inf, -np.inf], np.nan)

    future_ret_5 = closes.shift(-5) / closes - 1.0

    features = pd.DataFrame(
        {
            "ret_1": returns,
            "ret_3": ret_3,
            "ret_5": ret_5,
            "ret_10": ret_10,
            "ema_gap": ema_gap,
            "ema_slope": ema_slope,
            "breakout_gap": breakout_gap,
            "retrace_gap": retrace_gap,
            "volume_ratio": volume_ratio,
            "vol_slope": vol_slope,
            "vol_ratio": vol_ratio,
            "atr_ratio": atr_ratio,
            "volume_zscore": volume_zscore,
        }
    )

    dataset = features.join(future_ret_5.rename("future_ret_5")).dropna()

    if dataset.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    threshold = float(dataset["future_ret_5"].quantile(LABEL_QUANTILE))
    labels = (dataset["future_ret_5"] >= threshold).astype(float)

    if labels.nunique() < 2:
        alt_threshold = float(dataset["future_ret_5"].mean())
        labels = (dataset["future_ret_5"] >= alt_threshold).astype(float)

    if labels.nunique() < 2:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    return dataset.drop(columns=["future_ret_5"]), labels, dataset["future_ret_5"]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def _build_reasoning(
    prob: float,
    base_rate: float,
    expected: float,
    base_expected: float,
    threshold: float,
    existing_short: int,
    improvement: float,
    improvement_threshold: float,
) -> str:
    direction = "force covering" if existing_short else "blocking new shorts"
    caution_clauses: list[str] = []
    if existing_short == 0 and prob < threshold:
        caution_clauses.append("probability below trigger")
    if improvement < improvement_threshold:
        caution_clauses.append("calibration discount active")
    caution_suffix = ""
    if caution_clauses:
        caution_suffix = " Soft caution: " + ", ".join(caution_clauses) + "."
    return f"Upside squeeze probability {prob:.1%} (base {base_rate:.1%}, trigger {threshold:.1%}); " f"expected 5d return {expected:.2%} vs base {base_expected:.2%} (Δ {improvement:.2%}, " f"trigger {improvement_threshold:.2%}) → {direction}.{caution_suffix}"


def _compute_probability_boost(
    row: pd.Series,
    improvement_raw: float,
    improvement_threshold: float,
) -> tuple[float, dict[str, float]]:
    volatility_ratio = float(row.get("vol_ratio", 0.0))
    volume_zscore = float(row.get("volume_zscore", 0.0))
    atr_ratio = float(row.get("atr_ratio", 0.0))
    breakout_gap = float(row.get("breakout_gap", 0.0))

    probability_boost = 0.0
    if volatility_ratio > 0.03:
        probability_boost += min(0.06, (volatility_ratio - 0.03) * 0.5)
    if volume_zscore > 0.5:
        probability_boost += min(0.04, (volume_zscore - 0.5) * 0.015)
    if atr_ratio > 0.015:
        probability_boost += min(0.04, (atr_ratio - 0.015) * 1.5)
    if breakout_gap > 0.0:
        probability_boost += min(0.03, breakout_gap * 4.0)
    if improvement_raw >= improvement_threshold * 1.15 and improvement_raw > 0:
        probability_boost += min(0.03, (improvement_raw / max(improvement_threshold, 1e-9) - 1.15) * 0.02)

    probability_boost = float(np.clip(probability_boost, 0.0, 0.12))
    return probability_boost, {
        "volatility_ratio": volatility_ratio,
        "volume_zscore": volume_zscore,
        "atr_ratio": atr_ratio,
        "breakout_gap": breakout_gap,
    }


def _estimate_boost_trim_streak(
    feature_history: pd.DataFrame,
    train_probs: np.ndarray,
    raw_train_probs: np.ndarray,
    improvement_threshold: float,
    base_expected: float,
    pos_mean: float,
    neg_mean: float,
    threshold: float,
    soft_threshold: float,
    escalation_margin: float,
    window: int,
) -> int:
    if feature_history.empty or len(train_probs) == 0:
        return 0

    usable = min(window, len(feature_history), len(train_probs))
    if usable <= 0:
        return 0

    start_index = len(train_probs) - usable
    streak = 0

    for idx in range(start_index, len(train_probs)):
        row = feature_history.iloc[idx]
        prob = float(train_probs[idx])
        raw_prob = float(raw_train_probs[idx]) if idx < len(raw_train_probs) else float(prob)

        improvement_raw = prob * pos_mean + (1.0 - prob) * neg_mean - base_expected
        boost, _ = _compute_probability_boost(row, improvement_raw, improvement_threshold)
        prob_with_boost = min(1.0, prob + boost)
        expected_after_boost = prob_with_boost * pos_mean + (1.0 - prob_with_boost) * neg_mean
        improvement_after_boost = expected_after_boost - base_expected

        soft_trigger = prob_with_boost >= soft_threshold and improvement_after_boost >= improvement_threshold
        if soft_trigger:
            raw_close_enough = raw_prob >= threshold - escalation_margin
            if raw_close_enough:
                streak = 0
            else:
                streak += 1
        else:
            streak = 0

    return streak


def short_cover_classifier_agent(state: AgentState, agent_id: str = "short_cover_classifier_agent"):
    """Detects upside squeeze risk via logistic classification and forces timely short covers."""

    data = state.get("data", {})
    tickers = data.get("tickers", [])
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    portfolio = data.get("portfolio", {})
    positions = portfolio.get("positions", {})
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    analyst_view: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Calibrating squeeze classifier")
        prices = get_prices(
            ticker=ticker,
            start_date=_extend_start(start_date) or start_date,
            end_date=end_date,
            api_key=api_key,
        )

        if not prices:
            analyst_view[ticker] = {
                "signal": "neutral",
                "confidence": 25,
                "reasoning": "No price data available for squeeze classifier.",
                "constraints": {},
                "metrics": {},
            }
            continue

        df = prices_to_df(prices)
        if df.empty or len(df) < MIN_OBSERVATIONS:
            analyst_view[ticker] = {
                "signal": "neutral",
                "confidence": 30,
                "reasoning": f"Need at least {MIN_OBSERVATIONS} observations to fit squeeze classifier.",
                "constraints": {},
                "metrics": {"observations": int(len(df))},
            }
            continue

        df = df.sort_index()
        features_df, labels, future_ret = _prepare_dataset(df)
        if features_df.empty or len(features_df) < MIN_TRAINING_SAMPLES:
            analyst_view[ticker] = {
                "signal": "neutral",
                "confidence": 30,
                "reasoning": "Insufficient calibrated samples to evaluate squeeze probability.",
                "constraints": {},
                "metrics": {"calibration_samples": int(len(features_df))},
            }
            continue

        feature_matrix = features_df.to_numpy(dtype=float)
        intercept = np.ones((feature_matrix.shape[0], 1))
        X = np.hstack([intercept, feature_matrix])
        y = labels.to_numpy(dtype=float)

        if labels.iloc[:-1].nunique() < 2:
            analyst_view[ticker] = {
                "signal": "neutral",
                "confidence": 32,
                "reasoning": "Training labels lack variation for squeeze classifier.",
                "constraints": {},
                "metrics": {"label_variation": int(labels.nunique())},
            }
            continue

        train_X = X[:-1]
        train_y = y[:-1]
        latest_features = X[-1]

        if len(train_y) < MIN_TRAINING_SAMPLES - 1 or np.all(train_y == train_y[0]):
            analyst_view[ticker] = {
                "signal": "neutral",
                "confidence": 33,
                "reasoning": "Training sample too small or degenerate for squeeze classifier.",
                "constraints": {},
                "metrics": {"train_size": int(len(train_y))},
            }
            continue

        weights = _fit_logistic(train_X, train_y)

        train_logits = train_X @ weights
        raw_train_probs = _sigmoid(train_logits)
        latest_logit = float(latest_features @ weights)
        raw_prob_squeeze = float(_sigmoid(np.array([latest_logit]))[0])

        calibration_params = get_platt_parameters("short_cover_classifier", ticker)
        if calibration_params:
            a = float(calibration_params.get("a", 0.0))
            b = float(calibration_params.get("b", 0.0))
            train_probs = _sigmoid(a * train_logits + b)
            prob_squeeze = float(apply_platt(latest_logit, calibration_params))
            calibration_applied = True
        else:
            train_probs = raw_train_probs
            prob_squeeze = raw_prob_squeeze
            calibration_applied = False

        base_rate = float(train_y.mean()) if len(train_y) else 0.0
        quantile_threshold = float(np.quantile(train_probs, TRIGGER_QUANTILE)) if len(train_probs) else prob_squeeze
        base_threshold = max(base_rate + 0.012, quantile_threshold)

        train_future = future_ret.iloc[:-1]
        train_labels = labels.iloc[:-1]
        positive_returns = train_future[train_labels == 1]
        negative_returns = train_future[train_labels == 0]

        pos_mean = float(positive_returns.mean()) if not positive_returns.empty else float(train_future[train_future > 0].mean() or 0.012)
        neg_mean = float(negative_returns.mean()) if not negative_returns.empty else float(train_future[train_future <= 0].mean() or -0.012)

        base_expected = base_rate * pos_mean + (1.0 - base_rate) * neg_mean
        pre_boost_probability = prob_squeeze
        expected_return_raw = pre_boost_probability * pos_mean + (1.0 - pre_boost_probability) * neg_mean
        improvement_raw = expected_return_raw - base_expected
        future_std = float(train_future.std(ddof=0)) if len(train_future) > 1 else 0.0

        current_position = positions.get(ticker, {}) or {}
        existing_short = int(current_position.get("short") or 0)

        signal = "neutral"
        constraints: dict[str, Any] = {}
        improvement_threshold = max(future_std * IMPROVEMENT_SIGMA, MIN_IMPROVEMENT)
        bias_delta = prob_squeeze - base_rate

        latest_row = features_df.iloc[-1]
        probability_boost_raw, boost_feature_metrics = _compute_probability_boost(
            latest_row,
            improvement_raw,
            improvement_threshold,
        )
        probability_boost = probability_boost_raw
        if probability_boost_raw:
            prob_squeeze = min(1.0, prob_squeeze + probability_boost_raw)
            probability_boost = prob_squeeze - pre_boost_probability

        boost_delta = prob_squeeze - raw_prob_squeeze

        precision_mask_threshold = max(quantile_threshold - PRECISION_WINDOW_LEEWAY, base_rate)
        precision_mask = train_probs >= precision_mask_threshold
        precision_samples = int(np.sum(precision_mask))
        historical_precision = float(train_y[precision_mask].mean()) if precision_samples else 0.0
        if math.isnan(historical_precision):
            historical_precision = 0.0

        precision_override = PRECISION_TICKER_OVERRIDES.get(ticker.upper(), {})
        precision_min_samples = max(
            1,
            int(precision_override.get("min_samples", PRECISION_MIN_SAMPLES)),
        )
        precision_min_precision = float(precision_override.get("min_precision", PRECISION_MIN_PRECISION))
        precision_min_precision = max(0.0, min(0.995, precision_min_precision))

        precision_bonus_active = precision_samples >= precision_min_samples and historical_precision >= precision_min_precision
        precision_threshold_shift_base = 0.0
        precision_margin_bonus_base = 0.0
        precision_threshold_shift = 0.0
        precision_margin_bonus = 0.0
        precision_boost_trim_streak = 0
        precision_threshold_adjustment = 1.0
        precision_margin_adjustment = 1.0
        precision_improvement_discount = 0.0
        precision_improvement_discount_raw = 0.0
        precision_improvement_discount_factor = 1.0
        precision_streak_improvement_discount = 0.0
        precision_trend_strength = 0.0
        precision_trend_gap = 0.0
        precision_trend_slope = 0.0
        precision_trend_breakout = 0.0
        improvement_threshold_effective = improvement_threshold
        excess_precision = 0.0
        precision_margin_floor = 0.0
        precision_margin_clamped = False
        precision_streak_shift_recovery = 0.0
        precision_override_active = False

        precision_threshold_shift_scale = PRECISION_THRESHOLD_SHIFT_SCALE
        precision_threshold_shift_cap = PRECISION_THRESHOLD_SHIFT_MAX
        precision_margin_scale = PRECISION_MARGIN_SCALE
        precision_margin_cap = PRECISION_MARGIN_MAX
        precision_streak_shift_scale = PRECISION_STREAK_SHIFT_SCALE
        precision_streak_shift_step = PRECISION_STREAK_SHIFT_STEP
        precision_streak_shift_recovery_cap = PRECISION_STREAK_SHIFT_RECOVERY
        precision_streak_margin_decay = PRECISION_STREAK_MARGIN_DECAY
        precision_streak_margin_min_ratio = PRECISION_STREAK_MARGIN_MIN_RATIO
        precision_streak_margin_floor = PRECISION_STREAK_MARGIN_FLOOR
        precision_improvement_discount_scale = PRECISION_IMPROVEMENT_DISCOUNT_SCALE
        precision_improvement_discount_cap = PRECISION_IMPROVEMENT_DISCOUNT_MAX
        precision_improvement_discount_min_factor = PRECISION_IMPROVEMENT_DISCOUNT_MIN_FACTOR
        precision_improvement_discount_trend_cap = PRECISION_IMPROVEMENT_DISCOUNT_TREND_CAP
        precision_streak_improvement_discount_scale = PRECISION_STREAK_IMPROVEMENT_DISCOUNT_SCALE
        precision_streak_improvement_discount_cap = PRECISION_STREAK_IMPROVEMENT_DISCOUNT_MAX
        precision_streak_trigger = PRECISION_STREAK_TRIGGER
        precision_streak_trim_scale = PRECISION_STREAK_TRIM_SCALE
        precision_streak_trim_cap = PRECISION_STREAK_TRIM_CAP
        precision_streak_trim_scale = max(0.0, precision_streak_trim_scale)
        precision_streak_trim_cap = max(0.0, min(1.0, precision_streak_trim_cap))
        overrides_allowed = not any(globals()[name] != baseline for name, baseline in _PRECISION_CONSTANT_BASELINE.items())

        if precision_bonus_active:
            excess_precision = historical_precision - precision_min_precision
            precision_threshold_shift_base = min(
                precision_threshold_shift_cap,
                max(0.0, excess_precision * precision_threshold_shift_scale),
            )
            precision_margin_bonus_base = min(
                precision_margin_cap,
                max(0.0, excess_precision * precision_margin_scale),
            )
            precision_threshold_shift = precision_threshold_shift_base
            precision_margin_bonus = precision_margin_bonus_base

            history_features = features_df.iloc[:-1]
            pre_threshold = max(base_rate + 0.008, base_threshold - precision_threshold_shift_base)
            pre_soft_threshold = max(base_rate + 0.008, pre_threshold - TRIGGER_LEEWAY)
            pre_escalation_margin = BOOST_ESCALATION_MARGIN + precision_margin_bonus_base

            precision_boost_trim_streak = _estimate_boost_trim_streak(
                history_features,
                train_probs,
                raw_train_probs,
                improvement_threshold,
                base_expected,
                pos_mean,
                neg_mean,
                pre_threshold,
                pre_soft_threshold,
                pre_escalation_margin,
                PRECISION_STREAK_WINDOW,
            )

            if precision_override and overrides_allowed:
                precision_streak_trigger = int(precision_override.get("streak_trigger", precision_streak_trigger))
            if precision_override and overrides_allowed and precision_boost_trim_streak >= precision_streak_trigger:
                precision_threshold_shift_scale = precision_override.get(
                    "threshold_shift_scale",
                    precision_threshold_shift_scale,
                )
                precision_threshold_shift_cap = precision_override.get(
                    "threshold_shift_cap",
                    precision_threshold_shift_cap,
                )
                precision_margin_scale = precision_override.get(
                    "margin_scale",
                    precision_margin_scale,
                )
                precision_margin_cap = precision_override.get(
                    "margin_cap",
                    precision_margin_cap,
                )
                precision_streak_shift_scale = precision_override.get(
                    "streak_shift_scale",
                    precision_streak_shift_scale,
                )
                precision_streak_shift_step = precision_override.get(
                    "streak_shift_step",
                    precision_streak_shift_step,
                )
                precision_streak_shift_recovery_cap = precision_override.get(
                    "streak_shift_recovery",
                    precision_streak_shift_recovery_cap,
                )
                precision_streak_margin_decay = precision_override.get(
                    "streak_margin_decay",
                    precision_streak_margin_decay,
                )
                precision_streak_margin_min_ratio = precision_override.get(
                    "streak_margin_min_ratio",
                    precision_streak_margin_min_ratio,
                )
                precision_streak_margin_floor = precision_override.get(
                    "streak_margin_floor",
                    precision_streak_margin_floor,
                )
                precision_improvement_discount_scale = precision_override.get(
                    "improvement_discount_scale",
                    precision_improvement_discount_scale,
                )
                precision_improvement_discount_cap = precision_override.get(
                    "improvement_discount_cap",
                    precision_improvement_discount_cap,
                )
                precision_improvement_discount_min_factor = precision_override.get(
                    "improvement_discount_min_factor",
                    precision_improvement_discount_min_factor,
                )
                precision_improvement_discount_trend_cap = precision_override.get(
                    "improvement_discount_trend_cap",
                    precision_improvement_discount_trend_cap,
                )
                precision_streak_improvement_discount_scale = precision_override.get(
                    "streak_improvement_discount_scale",
                    precision_streak_improvement_discount_scale,
                )
                precision_streak_improvement_discount_cap = precision_override.get(
                    "streak_improvement_discount_cap",
                    precision_streak_improvement_discount_cap,
                )
                precision_streak_trim_scale = precision_override.get(
                    "streak_trim_scale",
                    precision_streak_trim_scale,
                )
                precision_streak_trim_cap = precision_override.get(
                    "streak_trim_cap",
                    precision_streak_trim_cap,
                )
                precision_streak_trim_scale = max(0.0, precision_streak_trim_scale)
                precision_streak_trim_cap = max(0.0, min(1.0, precision_streak_trim_cap))

                precision_threshold_shift_base = min(
                    precision_threshold_shift_cap,
                    max(0.0, excess_precision * precision_threshold_shift_scale),
                )
                precision_margin_bonus_base = min(
                    precision_margin_cap,
                    max(0.0, excess_precision * precision_margin_scale),
                )
                precision_threshold_shift = precision_threshold_shift_base
                precision_margin_bonus = precision_margin_bonus_base
                precision_override_active = True

            if precision_boost_trim_streak >= precision_streak_trigger:
                streak_factor = precision_boost_trim_streak - precision_streak_trigger + 1

                if precision_margin_bonus_base > 0.0:
                    precision_margin_floor = max(
                        precision_streak_margin_floor,
                        precision_margin_bonus_base * precision_streak_margin_min_ratio,
                    )
                    decayed_margin = precision_margin_bonus_base * (precision_streak_margin_decay**streak_factor)
                    precision_margin_bonus = max(decayed_margin, precision_margin_floor)
                    precision_margin_adjustment = precision_margin_bonus / precision_margin_bonus_base
                    precision_margin_clamped = precision_margin_bonus == precision_margin_floor
                else:
                    precision_margin_bonus = 0.0
                    precision_margin_adjustment = 0.0
                    precision_margin_floor = 0.0
                    precision_margin_clamped = False

                additional_shift = precision_threshold_shift_base * precision_streak_shift_scale * streak_factor + precision_streak_shift_step * streak_factor
                precision_threshold_shift_candidate = precision_threshold_shift_base + additional_shift
                precision_threshold_shift = min(
                    precision_threshold_shift_cap,
                    precision_threshold_shift_candidate,
                )
                if precision_threshold_shift_base > 0.0:
                    precision_threshold_adjustment = precision_threshold_shift / precision_threshold_shift_base
                else:
                    precision_threshold_adjustment = 1.0 if precision_threshold_shift > 0.0 else 0.0

                if precision_margin_clamped and precision_threshold_shift < precision_threshold_shift_cap:
                    precision_streak_shift_recovery = min(
                        precision_threshold_shift_cap - precision_threshold_shift,
                        precision_streak_shift_recovery_cap * streak_factor,
                    )
                    precision_threshold_shift += precision_streak_shift_recovery
                    if precision_threshold_shift_base > 0.0:
                        precision_threshold_adjustment = precision_threshold_shift / precision_threshold_shift_base
                    else:
                        precision_threshold_adjustment = 1.0 if precision_threshold_shift > 0.0 else 0.0
            elif precision_margin_bonus_base > 0.0:
                precision_margin_floor = max(
                    precision_streak_margin_floor,
                    precision_margin_bonus_base * precision_streak_margin_min_ratio,
                )

            precision_improvement_discount_raw = max(0.0, excess_precision * precision_improvement_discount_scale)
            precision_improvement_discount_raw *= max(1.0, precision_threshold_adjustment)
            if 0.0 < precision_margin_adjustment < 1.0:
                precision_improvement_discount_raw *= 1.0 / max(precision_margin_adjustment, 1e-6)

            if precision_boost_trim_streak >= precision_streak_trigger:
                streak_factor = precision_boost_trim_streak - precision_streak_trigger + 1
                clamp_bonus = 1.0
                if precision_margin_clamped:
                    margin_adjustment_safe = precision_margin_adjustment if precision_margin_adjustment > 0 else 1.0
                    clamp_bonus += max(0.0, 1.0 - min(1.0, margin_adjustment_safe))
                precision_streak_improvement_discount = min(
                    precision_streak_improvement_discount_cap,
                    streak_factor * precision_streak_improvement_discount_scale * clamp_bonus,
                )
                precision_improvement_discount_raw += precision_streak_improvement_discount

            precision_improvement_discount = min(precision_improvement_discount_cap, precision_improvement_discount_raw)

            precision_trend_gap = max(0.0, float(latest_row.get("ema_gap", 0.0)))
            precision_trend_slope = max(0.0, float(latest_row.get("ema_slope", 0.0)))
            precision_trend_breakout = max(0.0, float(latest_row.get("breakout_gap", 0.0)))
            precision_trend_strength = 0.0
            precision_improvement_discount_factor = 1.0

            if precision_improvement_discount > 0.0:
                precision_trend_strength = min(
                    precision_improvement_discount_trend_cap,
                    precision_trend_gap * PRECISION_IMPROVEMENT_DISCOUNT_TREND_GAP_WEIGHT + precision_trend_slope * PRECISION_IMPROVEMENT_DISCOUNT_TREND_SLOPE_WEIGHT + precision_trend_breakout * PRECISION_IMPROVEMENT_DISCOUNT_TREND_BREAKOUT_WEIGHT,
                )
                precision_improvement_discount_factor = max(
                    precision_improvement_discount_min_factor,
                    1.0 - precision_trend_strength,
                )
                precision_improvement_discount *= precision_improvement_discount_factor

            if precision_improvement_discount > 0.0:
                improvement_threshold_effective = max(
                    MIN_IMPROVEMENT,
                    improvement_threshold * (1.0 - precision_improvement_discount),
                )

        threshold = max(base_rate + 0.008, base_threshold - precision_threshold_shift)
        soft_threshold = max(base_rate + 0.008, threshold - TRIGGER_LEEWAY)
        raw_threshold_gap = threshold - raw_prob_squeeze
        soft_threshold_gap = soft_threshold - raw_prob_squeeze

        boost_clamp_factor = 1.0
        boost_clamp_penalty = 0.0
        boost_volatility_penalty = 0.0
        boost_volume_penalty = 0.0
        boost_atr_penalty = 0.0
        boost_threshold_gap_penalty = 0.0

        if probability_boost_raw > 0.0:
            volatility_ratio = float(boost_feature_metrics.get("volatility_ratio", 0.0))
            volume_zscore = float(boost_feature_metrics.get("volume_zscore", 0.0))
            atr_ratio = float(boost_feature_metrics.get("atr_ratio", 0.0))

            boost_volatility_penalty = max(0.0, volatility_ratio - BOOST_VOLATILITY_CLAMP_START) * BOOST_VOLATILITY_CLAMP_WEIGHT
            boost_volume_penalty = max(0.0, volume_zscore - BOOST_VOLUME_CLAMP_START) * BOOST_VOLUME_CLAMP_WEIGHT
            boost_atr_penalty = max(0.0, atr_ratio - BOOST_ATR_CLAMP_START) * BOOST_ATR_CLAMP_WEIGHT
            boost_threshold_gap_penalty = max(0.0, raw_threshold_gap - BOOST_THRESHOLD_GAP_CLAMP_START) * BOOST_THRESHOLD_GAP_CLAMP_WEIGHT

            boost_clamp_penalty = min(
                BOOST_CLAMP_MAX_PENALTY,
                boost_volatility_penalty + boost_volume_penalty + boost_atr_penalty + boost_threshold_gap_penalty,
            )
            if boost_clamp_penalty > 0.0:
                boost_clamp_factor = max(BOOST_CLAMP_MIN_FACTOR, 1.0 - boost_clamp_penalty)

            probability_boost = probability_boost_raw * boost_clamp_factor
            prob_squeeze = min(1.0, raw_prob_squeeze + probability_boost)
            boost_delta = prob_squeeze - raw_prob_squeeze
        else:
            probability_boost = 0.0
            boost_delta = 0.0

        expected_return = prob_squeeze * pos_mean + (1.0 - prob_squeeze) * neg_mean
        improvement = expected_return - base_expected

        squeeze_trigger = prob_squeeze >= threshold and improvement > 0
        soft_trigger = prob_squeeze >= soft_threshold and improvement >= improvement_threshold_effective

        near_threshold_gap = threshold - prob_squeeze
        near_threshold = 0.0 < near_threshold_gap <= NEAR_THRESHOLD_BUFFER
        near_threshold_improvement = improvement >= max(
            improvement_threshold_effective * NEAR_THRESHOLD_IMPROVEMENT_SCALE,
            MIN_IMPROVEMENT * NEAR_THRESHOLD_IMPROVEMENT_SCALE,
        )
        near_threshold_bias = near_threshold and near_threshold_improvement and (raw_threshold_gap <= NEAR_THRESHOLD_RAW_GAP_LIMIT or boost_clamp_factor > NEAR_THRESHOLD_CLAMP_FACTOR_LIMIT)

        escalation_margin = BOOST_ESCALATION_MARGIN + precision_margin_bonus
        raw_close_enough = raw_prob_squeeze >= threshold - escalation_margin
        soft_raw_close_enough = raw_prob_squeeze >= soft_threshold - escalation_margin

        boost_driven_squeeze = squeeze_trigger and not raw_close_enough
        boost_driven_soft = soft_trigger and not soft_raw_close_enough

        boost_reclassification: str | None = None
        if existing_short > 0:
            improvement_margin = improvement - improvement_threshold_effective
            improvement_ratio = float("inf") if improvement_threshold_effective <= 0 else improvement / improvement_threshold_effective

            if (
                boost_driven_squeeze
                and raw_threshold_gap <= BOOST_RECLASSIFY_RAW_GAP
                and improvement_margin >= BOOST_RECLASSIFY_MIN_IMPROVEMENT_MARGIN
                and improvement_ratio >= BOOST_RECLASSIFY_MIN_IMPROVEMENT_FACTOR
            ):
                boost_driven_squeeze = False
                boost_reclassification = "squeeze_high_improvement"

            elif (
                boost_driven_soft
                and soft_threshold_gap <= BOOST_RECLASSIFY_RAW_GAP
                and improvement_margin >= BOOST_RECLASSIFY_MIN_IMPROVEMENT_MARGIN
                and improvement_ratio >= BOOST_RECLASSIFY_MIN_IMPROVEMENT_FACTOR
            ):
                boost_driven_soft = False
                boost_reclassification = "soft_high_improvement"

        metrics = {
            "probability": prob_squeeze,
            "raw_probability": raw_prob_squeeze,
            "probability_boost_raw": probability_boost_raw,
            "probability_boost": probability_boost,
            "boost_delta": boost_delta,
            "boost_clamp_factor": boost_clamp_factor,
            "boost_clamp_penalty": boost_clamp_penalty,
            "boost_volatility_penalty": boost_volatility_penalty,
            "boost_volume_penalty": boost_volume_penalty,
            "boost_atr_penalty": boost_atr_penalty,
            "boost_threshold_gap_penalty": boost_threshold_gap_penalty,
            "calibration_applied": calibration_applied,
            "boost_reclassification": boost_reclassification,
            "volatility_ratio": boost_feature_metrics["volatility_ratio"],
            "volume_zscore": boost_feature_metrics["volume_zscore"],
            "atr_ratio": boost_feature_metrics["atr_ratio"],
            "breakout_gap": boost_feature_metrics["breakout_gap"],
            "threshold": threshold,
            "base_threshold": base_threshold,
            "quantile_threshold": quantile_threshold,
            "soft_threshold": soft_threshold,
            "precision_bonus_active": precision_bonus_active,
            "historical_precision": historical_precision,
            "precision_samples": precision_samples,
            "precision_min_precision": precision_min_precision,
            "precision_min_samples": precision_min_samples,
            "precision_threshold_shift": precision_threshold_shift,
            "precision_margin_bonus": precision_margin_bonus,
            "precision_threshold_shift_base": precision_threshold_shift_base,
            "precision_margin_bonus_base": precision_margin_bonus_base,
            "precision_threshold_shift_cap": precision_threshold_shift_cap,
            "precision_margin_cap": precision_margin_cap,
            "precision_streak_shift_scale": precision_streak_shift_scale,
            "precision_streak_shift_step": precision_streak_shift_step,
            "precision_boost_trim_streak": precision_boost_trim_streak,
            "precision_threshold_adjustment": precision_threshold_adjustment,
            "precision_margin_adjustment": precision_margin_adjustment,
            "precision_margin_floor": precision_margin_floor,
            "precision_margin_clamped": precision_margin_clamped,
            "precision_streak_shift_recovery": precision_streak_shift_recovery,
            "precision_mask_threshold": precision_mask_threshold,
            "precision_override_active": precision_override_active,
            "precision_streak_trigger": precision_streak_trigger,
            "base_rate": base_rate,
            "expected_return": expected_return,
            "raw_expected_return": expected_return_raw,
            "base_expected": base_expected,
            "improvement": improvement,
            "raw_improvement": improvement_raw,
            "improvement_threshold": improvement_threshold,
            "improvement_threshold_effective": improvement_threshold_effective,
            "precision_improvement_discount_raw": precision_improvement_discount_raw,
            "precision_improvement_discount_factor": precision_improvement_discount_factor,
            "precision_trend_strength": precision_trend_strength,
            "precision_trend_gap": precision_trend_gap,
            "precision_trend_slope": precision_trend_slope,
            "precision_trend_breakout": precision_trend_breakout,
            "precision_improvement_discount": precision_improvement_discount,
            "precision_streak_improvement_discount": precision_streak_improvement_discount,
            "precision_improvement_discount_cap": precision_improvement_discount_cap,
            "squeeze_trigger": squeeze_trigger,
            "soft_trigger": soft_trigger,
            "boost_driven_squeeze": boost_driven_squeeze,
            "boost_driven_soft": boost_driven_soft,
            "positive_mean": pos_mean,
            "negative_mean": neg_mean,
            "future_std": future_std,
            "train_samples": int(len(train_y)),
            "near_threshold_gap": near_threshold_gap,
            "near_threshold_bias": near_threshold_bias,
            "raw_threshold_gap": raw_threshold_gap,
            "soft_threshold_gap": soft_threshold_gap,
            "bias_delta": bias_delta,
            "bias_mode": "neutral",
        }

        if squeeze_trigger and not boost_driven_squeeze:
            signal = "squeeze_risk"
            constraints.update(
                {
                    "preferred_direction": "long",
                    "block_new_shorts": True,
                    "allow_short": False,
                    "target_short_shares": 0,
                }
            )
            if existing_short > 0:
                signal = "squeeze_cover"
                constraints["force_cover_qty"] = existing_short
                constraints["force_cover_reason"] = "Classifier expects upside squeeze; unwind remaining shorts"

        elif soft_trigger and not boost_driven_soft:
            signal = "squeeze_risk"
            constraints.update(
                {
                    "preferred_direction": "long",
                    "block_new_shorts": True,
                    "allow_short": False,
                    "target_short_shares": 0,
                }
            )
            if existing_short > 0:
                signal = "squeeze_cover"
                constraints["force_cover_qty"] = existing_short
                constraints["force_cover_reason"] = "Classifier expects upside squeeze; unwind remaining shorts"

        elif boost_driven_squeeze or boost_driven_soft:
            metrics["caution_trigger"] = "boost_only"
            constraints["preferred_direction"] = "long"
            if existing_short > 0:
                signal = "trim_short"
                trim_ratio_base = min(
                    BOOST_CAUTIOUS_TRIM_RATIO,
                    max(NEAR_THRESHOLD_TRIM_RATIO, prob_squeeze - base_rate),
                )
                trim_ratio, trim_metrics = _scale_trim_ratio_for_precision(
                    trim_ratio_base=trim_ratio_base,
                    precision_override_active=precision_override_active,
                    precision_boost_trim_streak=precision_boost_trim_streak,
                    precision_streak_trigger=precision_streak_trigger,
                    precision_streak_trim_scale=precision_streak_trim_scale,
                    precision_streak_trim_cap=precision_streak_trim_cap,
                )
                trim_qty = max(1, int(np.ceil(existing_short * trim_ratio)))
                target_shares = max(0, existing_short - trim_qty)
                constraints.update(
                    {
                        "block_new_shorts": True,
                        "allow_short": True,
                        "max_additional_short_shares": 0,
                        "target_short_shares": target_shares,
                    }
                )
                metrics["trim_qty"] = trim_qty
                metrics["trim_ratio"] = trim_ratio
                metrics["trim_ratio_base"] = trim_ratio_base
                metrics.update(trim_metrics)
                metrics["bias_mode"] = "boost_trim"
            else:
                signal = "bias_long"
                constraints.update(
                    {
                        "block_new_shorts": True,
                        "allow_short": False,
                        "max_additional_short_shares": 0,
                        "target_short_shares": 0,
                    }
                )
                metrics["bias_mode"] = "boost_bias"

        elif near_threshold_bias:
            metrics["caution_trigger"] = "near_threshold"

            if existing_short > 0:
                constraints.update(
                    {
                        "preferred_direction": "long",
                        "block_new_shorts": True,
                        "allow_short": False,
                        "max_additional_short_shares": 0,
                    }
                )
                metrics["bias_mode"] = "near_threshold_trim"
                signal = "trim_short"
                trim_ratio = min(1.0, max(NEAR_THRESHOLD_TRIM_RATIO, prob_squeeze - base_rate))
                trim_qty = max(1, int(np.ceil(existing_short * trim_ratio)))
                constraints["target_short_shares"] = max(0, existing_short - trim_qty)
                metrics["trim_qty"] = trim_qty
                metrics["trim_ratio"] = trim_ratio
            else:
                signal = "bias_long"
                constraints["preferred_direction"] = "long"
                constraints.update(
                    {
                        "block_new_shorts": True,
                        "allow_short": False,
                        "max_additional_short_shares": 0,
                        "target_short_shares": 0,
                    }
                )
                metrics["bias_mode"] = "near_threshold_soft"

        elif (
            existing_short > 0
            and prob_squeeze > base_rate
            and improvement
            > max(
                improvement_threshold_effective * 0.5,
                MIN_IMPROVEMENT,
            )
        ):
            signal = "trim_short"
            trim_qty = max(1, int(np.ceil(existing_short * min(0.8, prob_squeeze - base_rate))))
            constraints.update(
                {
                    "preferred_direction": "long",
                    "block_new_shorts": True,
                    "allow_short": False,
                    "target_short_shares": max(0, existing_short - trim_qty),
                }
            )
            metrics["trim_qty"] = trim_qty
            metrics["bias_mode"] = "trim_soft"

        elif prob_squeeze > base_rate and improvement >= MIN_IMPROVEMENT:
            signal = "bias_long"
            if improvement >= improvement_threshold_effective or bias_delta >= BIAS_LOCK_MARGIN:
                constraints.update(
                    {
                        "preferred_direction": "long",
                        "block_new_shorts": True,
                        "allow_short": False,
                        "target_short_shares": 0,
                    }
                )
                metrics["bias_mode"] = "hard_bias"
            elif bias_delta >= SOFT_BIAS_MARGIN:
                constraints["preferred_direction"] = "long"
                metrics["bias_mode"] = "soft_bias"
            else:
                constraints["preferred_direction"] = "long"
                metrics["bias_mode"] = "minimal_bias"
        guardrail_active, guardrail_reasons = _assess_neutral_guardrail(metrics, existing_short)
        if guardrail_active and existing_short == 0 and signal in {"bias_long", "squeeze_risk"}:
            metrics["auto_signal_before_guardrail"] = signal
            previous_constraints = constraints.copy()
            signal = "neutral"
            constraints.clear()

            retained_constraints: dict[str, Any] = {}
            if previous_constraints.get("block_new_shorts"):
                retained_constraints["block_new_shorts"] = True
                retained_constraints["allow_short"] = False
                retained_constraints["target_short_shares"] = 0
                if "max_additional_short_shares" in previous_constraints:
                    retained_constraints["max_additional_short_shares"] = 0

            if previous_constraints.get("preferred_direction") or retained_constraints:
                retained_constraints["preferred_direction"] = "neutral"

            if retained_constraints:
                constraints.update(retained_constraints)
                metrics["guardrail_retained_constraints"] = sorted(retained_constraints.keys())

            if guardrail_reasons:
                metrics["guardrail_reasons"] = guardrail_reasons
            existing_mode = metrics.get("bias_mode")
            if existing_mode:
                metrics["bias_mode"] = f"{existing_mode}_guardrail"
            else:
                metrics["bias_mode"] = "guardrail_neutral"
        unblock_new_shorts_reason = None
        boost_only = False
        discount_unblock = False
        near_threshold_unblock = False
        if existing_short == 0 and constraints.get("block_new_shorts"):
            prob_gap = prob_squeeze - threshold
            boost_only = boost_driven_squeeze or boost_driven_soft
            discount_unblock = precision_bonus_active and precision_improvement_discount > 0.0 and precision_improvement_discount_factor < 1.0 and prob_squeeze >= soft_threshold and prob_gap <= BOOST_ONLY_NEW_SHORT_UNBLOCK_MARGIN
            near_threshold_unblock = not boost_only and not discount_unblock and prob_gap <= BOOST_ONLY_NEW_SHORT_UNBLOCK_MARGIN and prob_gap > -BOOST_ONLY_NEW_SHORT_UNBLOCK_MARGIN and not squeeze_trigger
            if boost_only:
                unblock_new_shorts_reason = "boost_guidance"
            elif discount_unblock:
                unblock_new_shorts_reason = "precision_discount"
            elif near_threshold_unblock:
                unblock_new_shorts_reason = "near_threshold_guidance"

        long_bias_strength = max(0.0, max(prob_squeeze - base_threshold, bias_delta))
        long_bias_neutralized = False

        if existing_short == 0 and constraints.get("block_new_shorts") and constraints.get("preferred_direction") == "long":
            if long_bias_strength < LONG_BIAS_NEUTRALIZE_MARGIN:
                constraints["preferred_direction"] = "neutral"
                long_bias_neutralized = True
                existing_mode = metrics.get("bias_mode")
                if existing_mode:
                    metrics["bias_mode"] = f"{existing_mode}_neutralized"
                else:
                    metrics["bias_mode"] = "neutralized"

        if unblock_new_shorts_reason:
            hold_block_due_to_precision = precision_bonus_active and precision_margin_clamped and (boost_only or discount_unblock or near_threshold_unblock)
            if hold_block_due_to_precision:
                metrics["new_short_unblock_reason"] = "precision_margin_clamped"
                existing_mode = metrics.get("bias_mode")
                metrics["bias_mode"] = f"{existing_mode}_clamped" if existing_mode else "guidance_clamped"
            else:
                constraints["block_new_shorts"] = False
                constraints.pop("max_additional_short_shares", None)
                constraints.pop("target_short_shares", None)
                if signal == "bias_long":
                    constraints["preferred_direction"] = "neutral"
                    constraints["allow_short"] = True
                else:
                    constraints.pop("preferred_direction", None)
                    constraints.pop("allow_short", None)
                    constraints.pop("block_new_shorts", None)
                metrics["new_short_unblock_reason"] = unblock_new_shorts_reason
                existing_mode = metrics.get("bias_mode")
                metrics["bias_mode"] = f"{existing_mode}_unblocked" if existing_mode else "guidance_unblocked"

                if not constraints:
                    metrics.setdefault("constraints_cleared", True)

        metrics["long_bias_neutralized"] = long_bias_neutralized
        metrics["long_bias_strength"] = long_bias_strength

        auto_confidence = int(np.clip(prob_squeeze * 100.0, 0, 100))
        reasoning = _build_reasoning(
            prob_squeeze,
            base_rate,
            expected_return,
            base_expected,
            threshold,
            existing_short,
            improvement,
            improvement_threshold,
        )

        auto_constraints = constraints.copy() if constraints else {}

        recommendation_context = {
            "signal": signal,
            "confidence": auto_confidence,
            "reasoning": reasoning,
            "constraints": auto_constraints,
        }

        persona_context = _prepare_llm_context(
            ticker=ticker,
            existing_short=existing_short,
            metrics=metrics,
            recommendation=recommendation_context,
        )

        decision = _request_llm_decision(state, agent_id, persona_context)

        final_signal = decision.signal
        final_confidence = int(np.clip(decision.confidence, 0, 100))
        final_reasoning = decision.reasoning

        persona_constraints_raw = decision.constraints
        if persona_constraints_raw is None:
            final_constraints = dict(auto_constraints)
            persona_constraints = None
        else:
            persona_constraints = _to_builtin(persona_constraints_raw)
            final_constraints = dict(persona_constraints)

        metrics["auto_signal"] = signal
        metrics["auto_confidence"] = auto_confidence
        metrics["persona_signal"] = final_signal
        metrics["persona_confidence"] = final_confidence
        metrics["persona_overrode_auto_signal"] = final_signal != signal
        metrics["persona_constraints_source"] = "persona" if persona_constraints_raw is not None else "suggestion"

        payload: dict[str, Any] = {
            "signal": final_signal,
            "confidence": final_confidence,
            "reasoning": final_reasoning,
            "metrics": metrics,
        }
        if final_constraints:
            payload["constraints"] = final_constraints

        analyst_view[ticker] = payload
        progress.update_status(
            agent_id,
            ticker,
            f"Persona squeeze view {prob_squeeze:.1%} (thr {threshold:.1%}) → {final_signal.upper()}",
        )

    analysis = {ticker: analyst_view.get(ticker, {}) for ticker in tickers}

    message = HumanMessage(content=json.dumps(analysis), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(analysis, "Short-Cover Classifier Agent")

    state["data"]["analyst_signals"][agent_id] = analysis

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }
