import json
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.agents.persona_utils import persona_from_observations
from src.tools.api import get_prices, prices_to_df
from src.utils.api_key import get_api_key_from_state
from src.utils.progress import progress


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, (float, int)):
            return float(value)
        return float(value)
    except (TypeError, ValueError):
        return default


def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


PERSONA_NAME = "Orion"
PERSONA_ROLE = "a momentum guardian who protects the book from fighting dominant trends"
PERSONA_BACKSTORY = (
    "Orion spent a decade trading trend-following strategies and now advises when our short book must respect momentum."
)
PERSONA_INSTRUCTIONS = (
    "Use the EMAs, ROC, RSI, and derived score to judge whether shorts should be capped.",
    "If momentum is bullish, do not hesitate to block shorts; if bearish, encourage flexibility.",
    "Explain the verdict in first person, referencing the metrics that mattered.",
)
ALLOWED_SIGNALS = ["bullish_momentum", "neutral", "bearish_momentum"]


##### Momentum Regime Analyst #####
def momentum_guardian_agent(state: AgentState, agent_id: str = "momentum_guardian_agent"):
    """Evaluate momentum regime and gate short exposure when trends turn bullish."""

    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    momentum_view: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Assessing momentum regime")
        prices = get_prices(ticker=ticker, start_date=start_date, end_date=end_date, api_key=api_key)
        if not prices:
            momentum_view[ticker] = {
                "signal": "neutral",
                "confidence": 0,
                "reasoning": "No price history available",
                "constraints": {},
            }
            continue

        prices_df = prices_to_df(prices)
        if prices_df.empty or len(prices_df) < 30:
            momentum_view[ticker] = {
                "signal": "neutral",
                "confidence": 0,
                "reasoning": "Insufficient lookback for momentum guard",
                "constraints": {},
            }
            continue

        close = prices_df["close"].dropna()
        current_price = _safe_float(close.iloc[-1])
        ema_fast = close.ewm(span=21, adjust=False).mean()
        ema_slow = close.ewm(span=55, adjust=False).mean()
        ema_long = close.ewm(span=200, adjust=False).mean() if len(close) >= 200 else None
        roc_10 = close.pct_change(periods=10)
        rsi_14 = _calculate_rsi(close, period=14)

        fast_val = _safe_float(ema_fast.iloc[-1], current_price)
        slow_val = _safe_float(ema_slow.iloc[-1], current_price)
        if ema_long is not None:
            long_val = _safe_float(ema_long.iloc[-1], slow_val)
            ema200_metric = long_val
        else:
            long_val = slow_val
            ema200_metric = None
        roc_val = _safe_float(roc_10.iloc[-1], 0.0)
        rsi_val = _safe_float(rsi_14.iloc[-1], 50.0)

        score = 0.0
        score += -0.35 if current_price < slow_val else 0.35
        score += -0.20 if fast_val < slow_val else 0.20
        score += -0.20 if ema_long is not None and current_price < long_val else 0.20 if ema_long is not None else 0.0
        if roc_val < -0.01:
            score += -0.15
        elif roc_val > 0.01:
            score += 0.15
        if rsi_val < 45:
            score += -0.1
        elif rsi_val > 60:
            score += 0.1

        score = float(np.clip(score, -1.0, 1.0))

        bullish_stack = (
            current_price > slow_val > long_val
            and slope_mid >= -0.005
            and slope_long >= -0.002
            and roc_val >= -0.01
        )
        bearish_stack = (
            current_price < slow_val < long_val
            and slope_mid <= 0.005
            and slope_long <= 0.002
            and roc_val <= 0.01
        )

        ema200_text = f"{ema200_metric:.2f}" if ema200_metric is not None else "N/A"

        metrics = {
            "price": current_price,
            "ema21": fast_val,
            "ema55": slow_val,
            "ema200": ema200_metric,
            "roc_10": roc_val,
            "rsi_14": rsi_val,
            "momentum_score": round(score, 3),
        }

        suggestions = {
            "block_shorts_hint": score >= 0.2,
            "preferred_direction_hint": "bullish" if score >= 0.2 else "bearish" if score <= -0.2 else "neutral",
            "max_short_pct_hint": (
                0.0 if score >= 0.2 else 0.15 if score <= -0.6 else 0.1 if score <= -0.2 else 0.05
            ),
        }

        observations = {
            "ticker": ticker,
            "metrics": metrics,
            "momentum_components": {
                "bullish_stack": bool(bullish_stack),
                "bearish_stack": bool(bearish_stack),
                "ema200_text": ema200_text,
            },
            "suggestions": suggestions,
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
            default_reasoning="Defaulted to neutral after missing persona response.",
        )

        momentum_view[ticker] = {
            "signal": decision.signal,
            "confidence": int(np.clip(decision.confidence, 0, 100)),
            "reasoning": decision.reasoning,
            "constraints": decision.constraints or {},
            "metrics": metrics,
            "meta": {"observations": observations},
        }

    message = HumanMessage(content=json.dumps(momentum_view), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(momentum_view, "Momentum Regime Analyst")

    state["data"].setdefault("analyst_signals", {})[agent_id] = momentum_view
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }
