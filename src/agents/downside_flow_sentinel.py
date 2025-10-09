"""Sentinel detecting persistent downside flows to enforce defensive positioning."""

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
PERSONA_NAME = "Vale"
PERSONA_ROLE = "a downside-flow sentinel who leans short when crash pressure builds"
PERSONA_BACKSTORY = (
    "Vale is a risk sentry forged during 2008, now combining flow statistics with trader instinct to decide when the book must lean short."
)
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


def _clip_ratio(value: float, scale: float, *, cap: float = 1.0) -> float:
    if scale <= 0 or not np.isfinite(value):
        return 0.0
    ratio = value / scale
    return float(np.clip(ratio, 0.0, cap))


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


def _score_downside(metrics: dict[str, float]) -> tuple[str, float]:
    ret5_component = _clip_ratio(max(0.0, -metrics.get("ret_5", 0.0)), 0.07, cap=1.2)
    ret10_component = _clip_ratio(max(0.0, -metrics.get("ret_10", 0.0)), 0.12, cap=1.2)
    drawdown_component = _clip_ratio(max(0.0, -metrics.get("drawdown_40", 0.0)), 0.18, cap=1.15)
    vol_component = _clip_ratio(max(0.0, metrics.get("vol_ratio", 0.0)), 0.5, cap=1.1)
    downside_component = float(np.clip(metrics.get("downside_share10", 0.0) / 0.7, 0.0, 1.0))
    slope_component = _clip_ratio(max(0.0, -metrics.get("trend_slope10", 0.0)), 0.012, cap=1.0)
    tail_component = _clip_ratio(max(0.0, -metrics.get("tail_loss20", 0.0)), 0.04, cap=1.0)

    severity = float(
        np.clip(
            0.26 * ret5_component
            + 0.20 * ret10_component
            + 0.24 * drawdown_component
            + 0.12 * vol_component
            + 0.10 * downside_component
            + 0.05 * slope_component
            + 0.03 * tail_component,
            0.0,
            1.35,
        )
    )

    ret_5 = metrics.get("ret_5", 0.0)
    ret_10 = metrics.get("ret_10", 0.0)
    drawdown_40 = metrics.get("drawdown_40", 0.0)

    if severity >= 0.85 or ret_5 <= -0.08 or drawdown_40 <= -0.14:
        return "crash_flow", severity
    if (
        severity >= 0.5
        or ret_5 <= -0.04
        or ret_10 <= -0.03
        or drawdown_40 <= -0.06
        or metrics.get("drawdown_20", 0.0) <= -0.05
    ):
        return "downside_trend", severity
    return "stable", severity


def _confidence_from_severity(stage: str, severity: float) -> int:
    if stage == "crash_flow":
        base = 72
        scale = 28
        ceiling = 99
    elif stage == "downside_trend":
        base = 58
        scale = 26
        ceiling = 92
    else:
        base = 45
        scale = 18
        ceiling = 78
    conf = base + severity * scale
    return int(np.clip(conf, 35, ceiling))


##### Downside Flow Sentinel #####
def downside_flow_sentinel_agent(
    state: AgentState, agent_id: str = "downside_flow_sentinel_agent"
):
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

        metrics: dict[str, float] = {}
        severity = 0.0
        position = positions.get(ticker, {}) or {}
        existing_long = _safe_int(position.get("long"))

        auto_signal = "no_data"
        auto_confidence = 35
        auto_reasoning = "No price history available to evaluate downside pressure."
        auto_constraints: dict[str, Any] = {"max_long_exposure_pct": 0.16}

        if not prices:
            progress.update_status(agent_id, ticker, "Failed: Missing price data")
        else:
            df = prices_to_df(prices)
            if df.empty or len(df) < 40:
                auto_signal = "insufficient_history"
                auto_confidence = 38
                auto_reasoning = "Require at least 40 observations to gauge flow regime."
                progress.update_status(agent_id, ticker, "Failed: Insufficient history")
            else:
                metrics = _compute_metrics(df)
                if not metrics:
                    auto_signal = "insufficient_history"
                    auto_confidence = 38
                    auto_reasoning = "Unable to derive downside metrics from recent prices."
                    progress.update_status(agent_id, ticker, "Failed: Metrics unavailable")
                else:
                    auto_signal, severity = _score_downside(metrics)
                    auto_confidence = _confidence_from_severity(auto_signal, severity)

                    reasoning_parts = [
                        f"5d return {metrics['ret_5']:.1%}",
                        f"10d return {metrics['ret_10']:.1%}",
                        f"40d drawdown {metrics['drawdown_40']:.1%}",
                        f"vol spike {metrics['vol_ratio']:+.2f}",
                        f"downside hit-rate {metrics['downside_share10']:.0%}",
                    ]
                    auto_reasoning = "; ".join(reasoning_parts)

                    auto_constraints = {"max_long_exposure_pct": 0.16}
                    if auto_signal == "crash_flow":
                        trimmed_target = max(0, int(np.floor(existing_long * 0.35)))
                        auto_constraints.update(
                            {
                                "preferred_direction": "short",
                                "max_long_exposure_pct": 0.04,
                                "max_additional_long_shares": 0,
                                "max_long_shares": trimmed_target,
                                "reduce_position_change": True,
                            }
                        )
                    elif auto_signal == "downside_trend":
                        auto_constraints.update(
                            {
                                "preferred_direction": "short",
                                "max_long_exposure_pct": 0.08,
                            }
                        )
                        if existing_long > 0:
                            trimmed_target = max(0, int(np.floor(existing_long * 0.6)))
                            additional_cap = max(1, int(np.floor(existing_long * 0.2)))
                            auto_constraints.update(
                                {
                                    "max_additional_long_shares": additional_cap,
                                    "max_long_shares": trimmed_target,
                                }
                            )
                    else:
                        auto_constraints = {"max_long_exposure_pct": 0.16}
                        if existing_long > 0:
                            additional_cap = max(1, int(np.floor(existing_long * 0.4)))
                            auto_constraints["max_additional_long_shares"] = additional_cap

        observations = {
            "ticker": ticker,
            "existing_long_shares": existing_long,
            "downside_metrics": {key: round(val, 6) for key, val in metrics.items()},
            "downside_severity": round(severity, 3),
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
            default_reasoning="Defaulted to monitor after pattern fallback.",
        )

        metrics_payload = dict(observations["downside_metrics"])
        metrics_payload.update(
            {
                "severity": round(severity, 3),
                "existing_long_shares": existing_long,
            }
        )

        confidence = int(max(0, min(round(decision.confidence), 100)))
        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": confidence,
            "score": round(severity, 3),
            "reasoning": decision.reasoning,
            "metrics": metrics_payload,
            "constraints": decision.constraints or {},
            "meta": {"observations": observations},
        }

        progress.update_status(
            agent_id,
            ticker,
            f"{payload['signal'].upper()} @ {confidence}/100 | score {payload['score']:.2f}",
        )

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


__all__ = ["downside_flow_sentinel_agent"]
