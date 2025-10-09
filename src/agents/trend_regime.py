from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.agents.persona_utils import persona_from_observations
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices, prices_to_df
from src.utils.api_key import get_api_key_from_state
from src.utils.progress import progress


def _extended_start(start_date: str | None, buffer_days: int = 120) -> str | None:
    if not start_date:
        return None
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        return start_date
    return (start_dt - timedelta(days=buffer_days)).strftime("%Y-%m-%d")


PERSONA_NAME = "Helios"
PERSONA_ROLE = "a trend cartographer who narrates regime direction"
PERSONA_BACKSTORY = (
    "Helios traded systematic trend strategies and now weighs deterministic momentum features before advising the desk."
)
PERSONA_INSTRUCTIONS = (
    "Inspect EMA stacks, slopes, and momentum metrics before locking in a stance.",
    "Do not force bullish or bearish calls when evidence is mixed; explain why you stay neutral if so.",
    "Speak in first person and cite the decisive indicators.",
)
ALLOWED_SIGNALS = ["bullish", "bearish", "neutral"]

##### Trend Regime Analyst #####
def trend_regime_agent(state: AgentState, agent_id: str = "trend_regime_agent"):
    """Determines directional bias using deterministic price momentum and trend structure."""

    data = state["data"]
    tickers = data.get("tickers", [])
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    extended_start = _extended_start(start_date)

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
                "reasoning": "Insufficient price history to determine regime.",
            }
            continue

        df = prices_to_df(prices)
        if df.empty or len(df) < 30:
            signals[ticker] = {
                "signal": "neutral",
                "confidence": 45,
                "reasoning": "Not enough observations for trend diagnostics.",
            }
            continue

        df = df.sort_index()
        closes = df["close"].astype(float)

        ema_short = closes.ewm(span=10, adjust=False).mean()
        ema_mid = closes.ewm(span=21, adjust=False).mean()
        ema_long = closes.ewm(span=55, adjust=False).mean()

        recent_close = closes.iloc[-1]
        recent_mid = ema_mid.iloc[-1]
        recent_long = ema_long.iloc[-1]

        # Percentage slopes across the last 5 observations
        slope_mid = (ema_mid.iloc[-1] - ema_mid.iloc[-5]) / max(1e-6, ema_mid.iloc[-5]) if len(ema_mid) > 5 else 0.0
        slope_long = (ema_long.iloc[-1] - ema_long.iloc[-10]) / max(1e-6, ema_long.iloc[-10]) if len(ema_long) > 10 else 0.0

        # Momentum measures (scaled returns)
        mom_10 = closes.pct_change(10).iloc[-1] if len(closes) > 10 else 0.0
        mom_20 = closes.pct_change(20).iloc[-1] if len(closes) > 20 else mom_10

        rolling_high_40 = closes.rolling(window=40).max().iloc[-1] if len(closes) >= 40 else closes.max()
        rolling_low_40 = closes.rolling(window=40).min().iloc[-1] if len(closes) >= 40 else closes.min()

        trend_strength = float((slope_mid + slope_long) / 2)
        momentum_factor = float((mom_10 + mom_20) / 2)

        bullish_stack = (
            recent_close > recent_mid > recent_long
            and slope_mid >= -0.005
            and slope_long >= -0.002
            and momentum_factor >= -0.05
        )
        bearish_stack = (
            recent_close < recent_mid < recent_long
            and slope_mid <= 0.005
            and slope_long <= 0.002
            and momentum_factor <= 0.05
        )

        indicators = {
            "price": float(recent_close),
            "ema10": float(ema_short.iloc[-1]),
            "ema21": float(recent_mid),
            "ema55": float(recent_long),
            "mom_10": float(mom_10),
            "mom_20": float(mom_20),
            "slope_mid": float(slope_mid),
            "slope_long": float(slope_long),
            "range_high_40": float(rolling_high_40),
            "range_low_40": float(rolling_low_40),
        }

        observations = {
            "ticker": ticker,
            "indicators": indicators,
            "trend_strength": trend_strength,
            "momentum_factor": momentum_factor,
            "bullish_stack": bool(bullish_stack),
            "bearish_stack": bool(bearish_stack),
            "rolling_high_40": float(rolling_high_40),
            "rolling_low_40": float(rolling_low_40),
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
            default_reasoning="Defaulted to neutral when persona output was unavailable.",
        )

        payload = {
            "signal": decision.signal,
            "confidence": int(max(0, min(round(decision.confidence), 100))),
            "reasoning": decision.reasoning,
            "constraints": decision.constraints or {},
            "indicators": indicators,
            "meta": {"observations": observations},
        }

        progress.update_status(
            agent_id,
            ticker,
            f"Signal {payload['signal']} @ {payload['confidence']}/100 | mom10 {mom_10:.2%}",
        )

        signals[ticker] = payload

    message = HumanMessage(content=json.dumps(signals), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(signals, "Trend Regime Analyst")

    state["data"].setdefault("analyst_signals", {})[agent_id] = signals
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }
