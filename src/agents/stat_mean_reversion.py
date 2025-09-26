from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices, prices_to_df
from src.utils.api_key import get_api_key_from_state
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
        prob_revert_up = _norm_cdf(-z_score)
        prob_revert_down = 1.0 - prob_revert_up

        signal = "neutral"
        constraints: dict[str, Any] = {}
        reasoning = (
            f"Price deviation z-score {z_score:.2f}; reversion prob up {prob_revert_up:.2%}, down {prob_revert_down:.2%}."
        )

        if z_score <= -0.5:
            signal = "bullish"
            constraints["preferred_direction"] = "long"
            constraints["max_short_exposure_pct"] = float(0.25 * (1.0 - prob_revert_up))
            constraints["max_additional_short_shares"] = 0
            confidence = int(np.clip(prob_revert_up * 120, 55, 99))
        elif z_score >= 0.5:
            signal = "bearish"
            constraints["preferred_direction"] = "short"
            constraints["max_short_exposure_pct"] = float(0.25 * prob_revert_down)
            confidence = int(np.clip(prob_revert_down * 120, 55, 99))
        else:
            confidence = int(np.clip(50 + z_score * 20, 30, 60))

        indicators = {
            "z_score": float(z_score),
            "rolling_mean": float(latest_mean),
            "rolling_std": float(latest_std),
            "prob_revert_up": float(prob_revert_up),
            "prob_revert_down": float(prob_revert_down),
        }

        payload: dict[str, Any] = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "indicators": indicators,
        }
        if constraints:
            payload["constraints"] = constraints

        progress.update_status(
            agent_id,
            ticker,
            f"z={z_score:.2f} → {signal.upper()} ({confidence})"
        )

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

