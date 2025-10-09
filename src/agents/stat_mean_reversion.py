from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.agents.persona_utils import persona_from_observations

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices, prices_to_df
from src.utils.api_key import get_api_key_from_state
from src.utils.calibration import apply_platt, get_platt_parameters
from src.utils.progress import progress


def _extend_start(start_date: str | None, buffer_days: int = 120) -> str | None:
    if not start_date:
        return None
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        return start_date
    return (start_dt - timedelta(days=buffer_days)).strftime("%Y-%m-%d")


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))



PERSONA_NAME = "Mira"
PERSONA_ROLE = "a mean-reversion sleuth translating z-scores into positioning guidance"
PERSONA_BACKSTORY = (
    "Mira built statistical arbitrage desks and now narrates when stretched prices should mean-revert."
)
PERSONA_INSTRUCTIONS = (
    "Use the z-score and reversion probabilities to decide the stance.",
    "When probabilities are marginal, feel free to keep things neutral instead of forcing trades.",
    "Explain the judgement in first person, referencing the quantitative evidence.",
)
ALLOWED_SIGNALS = ["bullish", "bearish", "neutral"]

##### Statistical Mean Reversion Analyst #####
def stat_mean_reversion_agent(state: AgentState, agent_id: str = "stat_mean_reversion_agent"):
    """Uses z-score deviations from rolling means to assess reversion probabilities."""

    data = state.get("data", {})
    tickers = data.get("tickers", [])
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    extended_start = _extend_start(start_date)

    signals: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching price history")

        prices = get_prices(
            ticker=ticker,
            start_date=extended_start or start_date,
            end_date=end_date,
            api_key=api_key,
        )

        if not prices:
            signals[ticker] = {
                "signal": "neutral",
                "confidence": 40,
                "reasoning": "Missing price data for mean reversion analysis.",
            }
            continue

        df = prices_to_df(prices)
        if df.empty or len(df) < 30:
            signals[ticker] = {
                "signal": "neutral",
                "confidence": 42,
                "reasoning": "Need at least 30 observations to evaluate mean deviation.",
            }
            continue

        df = df.sort_index()
        closes = df["close"].astype(float)

        rolling_window = min(63, len(closes))
        mean = closes.rolling(rolling_window).mean()
        std = closes.rolling(rolling_window).std(ddof=0)

        latest_mean = mean.iloc[-1]
        latest_std = std.iloc[-1]
        latest_close = closes.iloc[-1]

        if np.isnan(latest_mean) or np.isnan(latest_std) or latest_std == 0:
            signals[ticker] = {
                "signal": "neutral",
                "confidence": 45,
                "reasoning": "Rolling statistics unavailable; staying neutral.",
            }
            continue

        z_score = (latest_close - latest_mean) / latest_std
        raw_prob_revert_up = _norm_cdf(-z_score)

        calibration_params = get_platt_parameters("stat_mean_reversion", ticker)
        if calibration_params:
            prob_revert_up = apply_platt(z_score, calibration_params)
        else:
            prob_revert_up = raw_prob_revert_up
        prob_revert_down = 1.0 - prob_revert_up

        indicators = {
            "z_score": float(z_score),
            "rolling_mean": float(latest_mean),
            "rolling_std": float(latest_std),
            "prob_revert_up": float(prob_revert_up),
            "prob_revert_down": float(prob_revert_down),
            "raw_prob_revert_up": float(raw_prob_revert_up),
            "calibration_applied": bool(calibration_params),
        }
        if calibration_params:
            indicators["calibration"] = calibration_params

        thresholds = {
            "bullish_prob": 0.6,
            "bearish_prob": 0.4,
        }

        observations = {
            "ticker": ticker,
            "z_score": float(z_score),
            "probabilities": {
                "prob_revert_up": float(prob_revert_up),
                "prob_revert_down": float(prob_revert_down),
                "raw_prob_revert_up": float(raw_prob_revert_up),
            },
            "calibration_applied": bool(calibration_params),
            "thresholds": thresholds,
            "indicators": indicators,
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
            default_signal="neutral",
            default_confidence=55.0,
            default_reasoning="Defaulted to neutral after observation-only fallback.",
        )

        confidence = int(np.clip(decision.confidence, 0, 100))
        constraints = decision.constraints or {}
        payload: dict[str, Any] = {
            "signal": decision.signal,
            "confidence": confidence,
            "reasoning": decision.reasoning,
            "constraints": constraints,
            "indicators": indicators,
            "meta": {
                "observations": observations,
            },
        }

        progress.update_status(agent_id, ticker, f"z={z_score:.2f} → {payload['signal'].upper()} ({payload['confidence']})")

        signals[ticker] = payload

    message = HumanMessage(content=json.dumps(signals), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(signals, "Statistical Mean Reversion Analyst")

    state["data"].setdefault("analyst_signals", {})[agent_id] = signals
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }
