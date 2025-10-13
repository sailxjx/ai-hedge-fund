"""Allocate explicit short targets when crash signals align."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping

import numpy as np
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

LOOKBACK_DAYS = 120
PERSONA_NAME = "Orion"
PERSONA_ROLE = "the crash-flight allocator guarding our downside hedges"
PERSONA_BACKSTORY = "You are a battle-tested crash tactician who survived multiple liquidity shocks. " "You weigh quantitative crash odds against market microstructure to size short allocations."
ALLOWED_SIGNALS = ("monitor", "short_bias", "crash_short", "cover_short")
PERSONA_INSTRUCTIONS = (
    "Lean into crash_short only when severity and probability align with the caps provided.",
    "Choose cover_short when crash pressure fades or constraints imply forced de-risking.",
    "Reference allocation_pct, caps, and price_profile details when explaining decisions.",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed):
        return default
    return parsed


def _bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(np.clip(value, lower, upper))


def _series_return(series, periods: int) -> float:
    if periods <= 0 or len(series) <= 1:
        return 0.0
    span = min(periods, len(series) - 1)
    if span <= 0:
        return 0.0
    past = float(series.iloc[-(span + 1)])
    last = float(series.iloc[-1])
    if past == 0:
        return 0.0
    return float(last / past - 1.0)


def _window_drawdown(series, window: int) -> float:
    if window <= 0 or series.empty:
        return 0.0
    window = min(window, len(series))
    if window <= 0:
        return 0.0
    recent_high = float(series.iloc[-window:].max())
    if recent_high <= 0:
        return 0.0
    current = float(series.iloc[-1])
    return float(current / recent_high - 1.0)


def _annualized_std(returns, window: int) -> float:
    if window <= 1 or returns.empty:
        return 0.0
    window = min(window, len(returns))
    if window <= 1:
        return 0.0
    segment = returns.tail(window)
    std = float(segment.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        return 0.0
    return std * np.sqrt(252.0)


def _normalized_slope(series, max_points: int = 15) -> float:
    window = min(max_points, len(series))
    if window <= 1:
        return 0.0
    segment = series.tail(window)
    idx = np.arange(len(segment))
    try:
        slope, _ = np.polyfit(idx, segment, 1)
    except np.linalg.LinAlgError:
        return 0.0
    mean = float(np.mean(segment))
    return float(slope / max(abs(mean), 1e-9))


def _extend_start(start_date: str | None, lookback: int) -> str | None:
    if not start_date:
        return None
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        return start_date
    return (start_dt - timedelta(days=lookback)).strftime("%Y-%m-%d")


def _min_constraint_pct(
    analyst_signals: Mapping[str, Any],
    ticker: str,
    key: str,
    exclude_agent: str,
) -> tuple[float | None, str | None]:
    """Return the lowest constraint percentage emitted by other agents for the ticker."""

    cap_value: float | None = None
    cap_agent: str | None = None

    for agent_id, signals in analyst_signals.items():
        if agent_id == exclude_agent or not isinstance(signals, Mapping):
            continue
        payload = signals.get(ticker)
        if not isinstance(payload, Mapping):
            continue
        constraints = payload.get("constraints")
        if not isinstance(constraints, Mapping):
            continue
        raw_value = constraints.get(key)
        try:
            candidate = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(candidate) or candidate < 0:
            continue
        if cap_value is None or candidate < cap_value:
            cap_value = candidate
            cap_agent = agent_id
    return cap_value, cap_agent


def _infer_margin_cap_pct(
    portfolio: Mapping[str, Any],
    price_hint: float | None,
    total_equity: float,
) -> float | None:
    """Approximate short capacity imposed by available margin."""

    if price_hint is None or price_hint <= 0 or total_equity <= 0:
        return None

    margin_requirement = _safe_float(portfolio.get("margin_requirement"), 0.0)
    if margin_requirement <= 0:
        return None

    equity = _safe_float(portfolio.get("equity"), total_equity)
    margin_used = _safe_float(portfolio.get("margin_used"), 0.0)

    available_margin = max(0.0, (equity / max(margin_requirement, 1e-9)) - margin_used)
    if available_margin <= 0:
        return 0.0

    return min(available_margin / total_equity, 1.0)


def _select_allocation_cap(candidates: list[tuple[float | None, str]]) -> tuple[float, str]:
    """Pick the binding allocation cap from candidate percentages."""

    valid: list[tuple[float, str]] = []
    for value, label in candidates:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric) or numeric < 0:
            continue
        valid.append((numeric, label))

    if not valid:
        return 0.0, "allocator allocation"

    return min(valid, key=lambda item: item[0])


@dataclass(slots=True)
class PriceProfile:
    close: float
    severity: float
    stage: str
    ret_5: float
    ret_10: float
    drawdown_30: float
    drawdown_60: float
    vol_ratio: float
    slope_15: float


@dataclass(slots=True)
class CrashInputs:
    crash_prob: float
    macro_score: float
    downside_score: float
    trend_bias: float
    event_bias: float
    downside_stage: str
    macro_stage: str
    price_profile: PriceProfile | None


def _compute_price_profile(state: AgentState, ticker: str) -> PriceProfile | None:
    data = state.get("data", {})
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    fetch_start = _extend_start(start_date, LOOKBACK_DAYS)
    prices = get_prices(
        ticker=ticker,
        start_date=fetch_start or start_date,
        end_date=end_date,
        api_key=api_key,
    )
    if not prices:
        return None

    df = prices_to_df(prices)
    if df.empty:
        return None

    close = df["close"].astype(float).dropna()
    if close.empty:
        return None

    ret_5 = _series_return(close, 5)
    ret_10 = _series_return(close, 10)

    drawdown_30 = _window_drawdown(close, 30)
    drawdown_60 = _window_drawdown(close, 60)

    returns = close.pct_change().dropna()
    vol_5 = _annualized_std(returns, 5)
    vol_20 = _annualized_std(returns, 20)
    vol_ratio = float(vol_5 / max(vol_20, 1e-9) - 1.0) if vol_20 > 0 else 0.0

    slope_15 = _normalized_slope(close, 15)

    ret5_component = _bounded(max(0.0, -ret_5) / 0.05, 0.0, 1.3)
    ret10_component = _bounded(max(0.0, -ret_10) / 0.08, 0.0, 1.2)
    drawdown_component = _bounded(max(0.0, -drawdown_60) / 0.2, 0.0, 1.2)
    vol_component = _bounded(max(0.0, vol_ratio) / 0.45, 0.0, 1.2)
    slope_component = _bounded(max(0.0, -slope_15) / 0.015, 0.0, 1.0)

    severity = float(
        np.clip(
            0.30 * ret5_component + 0.22 * ret10_component + 0.25 * drawdown_component + 0.15 * vol_component + 0.08 * slope_component,
            0.0,
            1.6,
        )
    )

    if severity >= 0.9 or ret_5 <= -0.07 or drawdown_60 <= -0.18:
        stage = "crash"
    elif severity >= 0.45 or ret_10 <= -0.04 or drawdown_30 <= -0.1:
        stage = "downtrend"
    else:
        stage = "calm"

    return PriceProfile(
        close=float(close.iloc[-1]),
        severity=severity,
        stage=stage,
        ret_5=ret_5,
        ret_10=ret_10,
        drawdown_30=drawdown_30,
        drawdown_60=drawdown_60,
        vol_ratio=vol_ratio,
        slope_15=slope_15,
    )


async def _compute_price_profile_async(state: AgentState, ticker: str) -> PriceProfile | None:
    data = state.get("data", {})
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    fetch_start = _extend_start(start_date, LOOKBACK_DAYS)
    prices = await get_prices_async(
        ticker=ticker,
        start_date=fetch_start or start_date,
        end_date=end_date,
        api_key=api_key,
    )
    if not prices:
        return None

    df = await prices_to_df_async(prices)
    if df.empty:
        return None

    close = df["close"].astype(float).dropna()
    if close.empty:
        return None

    ret_5 = _series_return(close, 5)
    ret_10 = _series_return(close, 10)

    drawdown_30 = _window_drawdown(close, 30)
    drawdown_60 = _window_drawdown(close, 60)

    returns = close.pct_change().dropna()
    vol_5 = _annualized_std(returns, 5)
    vol_20 = _annualized_std(returns, 20)
    vol_ratio = float(vol_5 / max(vol_20, 1e-9) - 1.0) if vol_20 > 0 else 0.0

    slope_15 = _normalized_slope(close, 15)

    ret5_component = _bounded(max(0.0, -ret_5) / 0.05, 0.0, 1.3)
    ret10_component = _bounded(max(0.0, -ret_10) / 0.08, 0.0, 1.2)
    drawdown_component = _bounded(max(0.0, -drawdown_60) / 0.2, 0.0, 1.2)
    vol_component = _bounded(max(0.0, vol_ratio) / 0.45, 0.0, 1.2)
    slope_component = _bounded(max(0.0, -slope_15) / 0.015, 0.0, 1.0)

    severity = float(
        np.clip(
            0.30 * ret5_component + 0.22 * ret10_component + 0.25 * drawdown_component + 0.15 * vol_component + 0.08 * slope_component,
            0.0,
            1.6,
        )
    )

    if severity >= 0.9 or ret_5 <= -0.07 or drawdown_60 <= -0.18:
        stage = "crash"
    elif severity >= 0.45 or ret_10 <= -0.04 or drawdown_30 <= -0.1:
        stage = "downtrend"
    else:
        stage = "calm"

    return PriceProfile(
        close=float(close.iloc[-1]),
        severity=severity,
        stage=stage,
        ret_5=ret_5,
        ret_10=ret_10,
        drawdown_30=drawdown_30,
        drawdown_60=drawdown_60,
        vol_ratio=vol_ratio,
        slope_15=slope_15,
    )


def _collect_inputs(
    analyst_signals: Mapping[str, Any],
    ticker: str,
    price_profile: PriceProfile | None,
) -> CrashInputs:
    regime_payload = (analyst_signals.get("regime_meta_agent") or {}).get(ticker, {})
    macro_payload = (analyst_signals.get("macro_volatility_sentinel_agent") or {}).get(ticker, {})
    downside_payload = (analyst_signals.get("downside_flow_sentinel_agent") or {}).get(ticker, {})
    trend_payload = (analyst_signals.get("trend_regime_agent") or {}).get(ticker, {})
    event_payload = (analyst_signals.get("event_catalyst_agent") or {}).get(ticker, {})

    probabilities = (regime_payload.get("indicators") or {}).get("probabilities", {})
    crash_prob = _bounded(_safe_float(probabilities.get("crash"), 0.0))

    macro_score = _bounded(_safe_float(macro_payload.get("score"), 0.0), 0.0, 1.4)
    downside_score = _bounded(_safe_float(downside_payload.get("score"), 0.0), 0.0, 1.4)

    trend_signal = str(trend_payload.get("signal") or "").lower()
    trend_conf = _safe_float(trend_payload.get("confidence"), 0.0)
    trend_bias = 0.0
    if trend_signal == "bearish" and trend_conf >= 65:
        trend_bias = _bounded((trend_conf - 60.0) / 100.0, 0.0, 0.4)

    downside_stage = str((downside_payload.get("signal") or "stable")).lower()
    macro_stage = str((macro_payload.get("signal") or "calm")).lower()

    event_signal = str(event_payload.get("signal") or "").lower()
    if event_signal == "bearish":
        event_bias = 0.12
    elif event_signal == "bullish":
        event_bias = -0.08
    else:
        event_bias = 0.0

    return CrashInputs(
        crash_prob=crash_prob,
        macro_score=macro_score,
        downside_score=downside_score,
        trend_bias=trend_bias,
        event_bias=event_bias,
        downside_stage=downside_stage,
        macro_stage=macro_stage,
        price_profile=price_profile,
    )


def _composite_score(inputs: CrashInputs) -> float:
    price_severity = inputs.price_profile.severity if inputs.price_profile else 0.0

    base = 0.35 * inputs.crash_prob
    base += 0.25 * price_severity
    base += 0.18 * _bounded(inputs.macro_score, 0.0, 1.0)
    base += 0.15 * _bounded(inputs.downside_score, 0.0, 1.0)
    base += 0.12 * inputs.trend_bias
    base += inputs.event_bias

    if inputs.price_profile:
        if inputs.price_profile.stage == "crash":
            base += 0.12
        elif inputs.price_profile.stage == "downtrend":
            base += 0.05

    if inputs.downside_stage == "crash_flow":
        base += 0.08
    elif inputs.downside_stage == "downside_trend":
        base += 0.04

    if inputs.macro_stage == "crash_alert":
        base += 0.1
    elif inputs.macro_stage == "vol_watch":
        base += 0.04

    if inputs.price_profile:
        # Penalise the composite score when prices rebound, so persistent shorts decay quickly.
        relief = 0.0
        if inputs.price_profile.stage == "calm":
            relief += 0.08
        elif inputs.price_profile.stage == "downtrend":
            relief += 0.03

        if inputs.price_profile.ret_5 > 0:
            relief += 0.12 * min(inputs.price_profile.ret_5 / 0.05, 1.0)
        if inputs.price_profile.ret_10 > 0:
            relief += 0.1 * min(inputs.price_profile.ret_10 / 0.08, 1.0)
        if inputs.price_profile.slope_15 > 0:
            relief += 0.06 * min(inputs.price_profile.slope_15 / 0.015, 1.0)

        base -= relief

    return float(np.clip(base, 0.0, 1.8))


def _short_allocation(score: float, price_severity: float, inputs: CrashInputs) -> float:
    """Translate composite conviction into a short allocation percentage."""

    crash_prob = inputs.crash_prob
    downside_score = inputs.downside_score
    macro_score = inputs.macro_score
    stage = inputs.price_profile.stage if inputs.price_profile else "calm"

    score_gate = 0.18
    severity_gate = 0.25

    if crash_prob >= 0.45:
        score_gate -= min(0.06, (crash_prob - 0.45) * 0.3)
        severity_gate -= min(0.05, (crash_prob - 0.45) * 0.25)

    if downside_score >= 0.4:
        score_gate -= min(0.03, (downside_score - 0.4) * 0.12)

    if macro_score >= 0.55:
        score_gate -= 0.015

    if stage in {"downtrend", "crash"}:
        severity_gate -= 0.04

    score_gate = max(0.1, score_gate)
    severity_gate = max(0.15, severity_gate)

    base = 0.03 + 0.3 * score + 0.18 * price_severity
    if crash_prob >= 0.5:
        base += 0.04
    if downside_score >= 0.7:
        base += 0.03
    if stage == "crash":
        base += 0.05
    elif stage == "downtrend":
        base += 0.02

    if score < score_gate and price_severity < severity_gate:
        if crash_prob >= 0.46:
            preload = 0.02 + max(0.0, crash_prob - 0.45) * 0.45
            base = max(base, preload)
        else:
            return 0.0

    return float(np.clip(base, 0.0, 0.45))


def _should_release_short(existing_short: int, inputs: CrashInputs, score: float) -> tuple[bool, str]:
    """Determine whether existing short exposure should be covered."""

    if existing_short <= 0:
        return False, ""

    profile = inputs.price_profile
    if profile is None:
        if score < 0.25 and inputs.crash_prob < 0.3:
            return True, "composite cooled below 0.25 with no crash diagnostics"
        return False, ""

    severity = profile.severity
    positive_ret_5 = profile.ret_5 >= 0.01
    positive_ret_10 = profile.ret_10 >= -0.005
    slope_positive = profile.slope_15 >= 0.0

    downside_relaxed = inputs.downside_stage not in {"crash_flow", "downside_trend"}
    macro_relaxed = inputs.macro_stage != "crash_alert"

    if profile.stage == "calm" and score < 0.55:
        return True, "price stage back to calm while composite <0.55"
    if severity <= 0.32 and score < 0.4 and (positive_ret_5 or slope_positive):
        return True, "severity cooled below 0.32 with upside momentum"
    if positive_ret_5 and positive_ret_10 and score < 0.5:
        return True, "5-10 day rebound eroded crash conviction"
    if downside_relaxed and macro_relaxed and score < 0.38:
        return True, "macro and downside sentinels neutral with composite <0.38"

    return False, ""


def _prepare_crash_observation(
    *,
    ticker: str,
    analyst_signals: Mapping[str, Any],
    portfolio: Mapping[str, Any],
    positions: Mapping[str, Any],
    price_profile: PriceProfile | None,
    inputs: CrashInputs,
    score: float,
    price_severity: float,
    fallback_total: float,
    cash_balance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build observation payloads for the crash short allocator persona."""

    position = positions.get(ticker, {}) or {}
    existing_long = int(_safe_float(position.get("long"), 0.0))
    existing_short = int(_safe_float(position.get("short"), 0.0))

    peer_short_cap_pct, peer_short_cap_source = _min_constraint_pct(analyst_signals, ticker, "max_short_exposure_pct", "crash_short_allocator_agent")
    peer_position_cap_pct, peer_position_cap_source = _min_constraint_pct(analyst_signals, ticker, "max_long_exposure_pct", "crash_short_allocator_agent")

    release_short, release_reason = _should_release_short(existing_short, inputs, score)
    raw_allocation_pct = 0.0 if release_short else _short_allocation(score, price_severity, inputs)

    regime_payload = (analyst_signals.get("regime_meta_agent") or {}).get(ticker, {})
    macro_payload = (analyst_signals.get("macro_volatility_sentinel_agent") or {}).get(ticker, {})
    downside_payload = (analyst_signals.get("downside_flow_sentinel_agent") or {}).get(ticker, {})

    price_hint = None
    if price_profile and price_profile.close > 0:
        price_hint = price_profile.close
    if price_hint is None:
        price_hint = _extract_close(regime_payload, macro_payload, downside_payload)

    total_equity = fallback_total
    if total_equity <= 0 and price_hint:
        total_equity = cash_balance + existing_long * price_hint
    if total_equity <= 0:
        total_equity = max(cash_balance, 100_000.0)

    long_cap_pct = float(np.clip(0.05 - 0.025 * (score + price_severity), 0.0, 0.05))
    margin_cap_pct = _infer_margin_cap_pct(portfolio, price_hint, total_equity)

    cap_candidates: list[tuple[float | None, str]] = [
        (raw_allocation_pct, "model_allocation"),
        (float(np.clip(0.14 + score * 0.25 + price_severity * 0.15, 0.14, 0.55)), "allocator_short_cap"),
        (peer_short_cap_pct, f"peer_short_cap:{peer_short_cap_source}" if peer_short_cap_source else "peer_short_cap"),
        (margin_cap_pct, "margin_capacity"),
    ]

    effective_allocation_pct, allocation_limiter = _select_allocation_cap(cap_candidates)
    if release_short:
        effective_allocation_pct = 0.0

    raw_target_short_shares: int | None = None
    if price_hint is not None and raw_allocation_pct > 0:
        raw_target_notional = total_equity * raw_allocation_pct
        raw_target_short_shares = max(1, int(np.floor(raw_target_notional / price_hint)))

    target_short_shares_estimate: int | None = None
    if price_hint is not None and effective_allocation_pct > 0:
        target_notional = total_equity * effective_allocation_pct
        target_short_shares_estimate = max(1, int(np.floor(target_notional / price_hint)))

    long_cap_effective = long_cap_pct
    if peer_position_cap_pct is not None:
        long_cap_effective = min(long_cap_effective, peer_position_cap_pct)
    long_cap_effective = max(0.0, long_cap_effective)

    observation = {
        "ticker": ticker,
        "composite_score": score,
        "price_severity": price_severity,
        "existing_positions": {"long": existing_long, "short": existing_short},
        "crash_inputs": {
            "crash_prob": inputs.crash_prob,
            "macro_score": inputs.macro_score,
            "downside_score": inputs.downside_score,
            "trend_bias": inputs.trend_bias,
            "event_bias": inputs.event_bias,
            "downside_stage": inputs.downside_stage,
            "macro_stage": inputs.macro_stage,
        },
        "price_profile": (
            {
                "close": price_profile.close,
                "severity": price_profile.severity,
                "stage": price_profile.stage,
                "ret_5": price_profile.ret_5,
                "ret_10": price_profile.ret_10,
                "drawdown_30": price_profile.drawdown_30,
                "drawdown_60": price_profile.drawdown_60,
                "vol_ratio": price_profile.vol_ratio,
                "slope_15": price_profile.slope_15,
            }
            if price_profile
            else None
        ),
        "release_analysis": {
            "should_release": release_short,
            "reason": release_reason,
        },
        "allocation_model": {
            "raw_allocation_pct": raw_allocation_pct,
            "effective_allocation_pct": effective_allocation_pct,
            "allocation_limiter": allocation_limiter,
            "allocation_candidates": [
                {"label": label, "value": value} for value, label in cap_candidates
            ],
            "raw_target_short_shares_estimate": raw_target_short_shares,
            "target_short_shares_estimate": target_short_shares_estimate,
            "price_hint": price_hint,
            "total_equity_estimate": total_equity,
        },
        "caps": {
            "long_cap_pct": long_cap_pct,
            "long_cap_effective": long_cap_effective,
            "margin_cap_pct": margin_cap_pct,
            "peer_short_cap_pct": peer_short_cap_pct,
            "peer_short_cap_source": peer_short_cap_source,
            "peer_position_cap_pct": peer_position_cap_pct,
            "peer_position_cap_source": peer_position_cap_source,
        },
        "financials": {
            "cash_balance": cash_balance,
            "fallback_total": fallback_total,
        },
    }

    indicators = {
        "crash_prob": inputs.crash_prob,
        "macro_score": inputs.macro_score,
        "downside_score": inputs.downside_score,
        "trend_bias": inputs.trend_bias,
        "event_bias": inputs.event_bias,
        "downside_stage": inputs.downside_stage,
        "macro_stage": inputs.macro_stage,
        "price_stage": price_profile.stage if price_profile else None,
        "price_severity": price_severity,
        "composite_score": score,
        "raw_allocation_pct": raw_allocation_pct,
        "effective_allocation_pct": effective_allocation_pct,
        "allocation_limiter": allocation_limiter,
        "raw_target_short_shares_estimate": raw_target_short_shares,
        "target_short_shares_estimate": target_short_shares_estimate,
        "release_short": release_short,
        "price_hint": price_hint,
        "long_cap_pct": long_cap_pct,
        "long_cap_effective": long_cap_effective,
        "margin_cap_pct": margin_cap_pct,
        "peer_short_cap_pct": peer_short_cap_pct,
        "peer_short_cap_source": peer_short_cap_source,
        "peer_position_cap_pct": peer_position_cap_pct,
        "peer_position_cap_source": peer_position_cap_source,
    }

    return observation, indicators


def crash_short_allocator_agent(state: AgentState, agent_id: str = "crash_short_allocator_agent"):
    """Convert crash diagnostics into explicit short targets for risk management."""

    data = state.get("data", {})
    tickers = data.get("tickers", [])
    portfolio = data.get("portfolio", {}) or {}
    positions = portfolio.get("positions", {}) or {}
    analyst_signals = data.setdefault("analyst_signals", {})

    cash_balance = _safe_float(portfolio.get("cash"), 0.0)
    fallback_total = _safe_float(portfolio.get("total_value"), cash_balance)

    allocator_payload: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Evaluating crash allocation")

        price_profile = _compute_price_profile(state, ticker)
        inputs = _collect_inputs(analyst_signals, ticker, price_profile)
        score = _composite_score(inputs)
        price_severity = price_profile.severity if price_profile else 0.0

        observation, indicators = _prepare_crash_observation(
            ticker=ticker,
            analyst_signals=analyst_signals,
            portfolio=portfolio,
            positions=positions,
            price_profile=price_profile,
            inputs=inputs,
            score=score,
            price_severity=price_severity,
            fallback_total=fallback_total,
            cash_balance=cash_balance,
        )

        decision = persona_from_observations(
            state=state,
            agent_id=agent_id,
            persona_name=PERSONA_NAME,
            persona_role=PERSONA_ROLE,
            persona_backstory=PERSONA_BACKSTORY,
            allowed_signals=ALLOWED_SIGNALS,
            observations=observation,
            persona_instructions=PERSONA_INSTRUCTIONS,
            default_signal="monitor",
            default_confidence=55.0,
            default_reasoning="Defaulted to monitor after observation-only fallback.",
        )

        allocation_pct = 0.0
        if decision.constraints:
            for key in ("allocation_pct", "max_short_exposure_pct"):
                if key in decision.constraints:
                    try:
                        allocation_pct = float(decision.constraints[key])
                    except (TypeError, ValueError):
                        allocation_pct = 0.0
                    break

        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": int(max(0, min(round(decision.confidence), 100))),
            "allocation_pct": allocation_pct,
            "reasoning": decision.reasoning,
            "constraints": decision.constraints or {},
            "indicators": indicators,
            "meta": {"observations": observation},
        }

        payload["indicators"]["allocation_pct"] = allocation_pct

        progress.update_status(agent_id, ticker, f"{decision.signal.upper()} @ {payload['confidence']}/100 | alloc {allocation_pct:.2%}")

        allocator_payload[ticker] = payload

    message = HumanMessage(content=json.dumps(allocator_payload), name=agent_id)

    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(allocator_payload, "Crash Short Allocator")

    state.setdefault("data", {}).setdefault("analyst_signals", {})[agent_id] = allocator_payload
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state.get("messages", []) + [message],
        "data": state["data"],
    }


async def crash_short_allocator_agent_async(state: AgentState, agent_id: str = "crash_short_allocator_agent"):
    """Convert crash diagnostics into explicit short targets for risk management."""

    data = state.get("data", {})
    tickers = data.get("tickers", [])
    portfolio = data.get("portfolio", {}) or {}
    positions = portfolio.get("positions", {}) or {}
    analyst_signals = data.setdefault("analyst_signals", {})

    cash_balance = _safe_float(portfolio.get("cash"), 0.0)
    fallback_total = _safe_float(portfolio.get("total_value"), cash_balance)

    allocator_payload: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Evaluating crash allocation")

        price_profile = await _compute_price_profile_async(state, ticker)
        inputs = _collect_inputs(analyst_signals, ticker, price_profile)
        score = _composite_score(inputs)
        price_severity = price_profile.severity if price_profile else 0.0

        observation, indicators = _prepare_crash_observation(
            ticker=ticker,
            analyst_signals=analyst_signals,
            portfolio=portfolio,
            positions=positions,
            price_profile=price_profile,
            inputs=inputs,
            score=score,
            price_severity=price_severity,
            fallback_total=fallback_total,
            cash_balance=cash_balance,
        )

        decision = await async_persona_from_observations(
            state=state,
            agent_id=agent_id,
            persona_name=PERSONA_NAME,
            persona_role=PERSONA_ROLE,
            persona_backstory=PERSONA_BACKSTORY,
            allowed_signals=ALLOWED_SIGNALS,
            observations=observation,
            persona_instructions=PERSONA_INSTRUCTIONS,
            default_signal="monitor",
            default_confidence=55.0,
            default_reasoning="Defaulted to monitor after observation-only fallback.",
        )

        allocation_pct = 0.0
        if decision.constraints:
            for key in ("allocation_pct", "max_short_exposure_pct"):
                if key in decision.constraints:
                    try:
                        allocation_pct = float(decision.constraints[key])
                    except (TypeError, ValueError):
                        allocation_pct = 0.0
                    break

        confidence = int(max(0, min(round(decision.confidence), 100)))
        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": confidence,
            "allocation_pct": allocation_pct,
            "reasoning": decision.reasoning,
            "constraints": decision.constraints or {},
            "indicators": indicators,
            "meta": {"observations": observation},
        }

        payload["indicators"]["allocation_pct"] = allocation_pct

        progress.update_status(agent_id, ticker, f"{decision.signal.upper()} @ {confidence}/100 | alloc {allocation_pct:.2%}")

        allocator_payload[ticker] = payload

    message = HumanMessage(content=json.dumps(allocator_payload), name=agent_id)

    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(allocator_payload, "Crash Short Allocator")

    await update_analyst_signals_async(state, agent_id, allocator_payload)
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state.get("messages", []) + [message],
        "data": state["data"],
    }

__all__ = ["crash_short_allocator_agent", "crash_short_allocator_agent_async"]
