import json
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
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
        long_val = _safe_float(ema_long.iloc[-1], slow_val if ema_long is not None else slow_val)
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

        if score <= -0.2:
            signal = "bearish_momentum"
        elif score >= 0.2:
            signal = "bullish_momentum"
        else:
            signal = "neutral"

        # Default exposure caps based on regime
        if score >= 0.4:
            allow_short = False
            max_short_pct = 0.0
            confidence = int(min(100, 60 + abs(score) * 40))
        elif score >= 0.2:
            allow_short = False
            max_short_pct = 0.0
            confidence = int(min(100, 50 + abs(score) * 50))
        elif score >= 0.0:
            allow_short = True
            max_short_pct = 0.05
            confidence = int(min(100, 40 + abs(score) * 40))
        elif score <= -0.6:
            allow_short = True
            max_short_pct = 0.15
            confidence = int(min(100, 60 + abs(score) * 40))
        else:
            allow_short = True
            max_short_pct = 0.10
            confidence = int(min(100, 50 + abs(score) * 40))

        reasoning = (
            f"Price {current_price:.2f} vs EMA55 {slow_val:.2f}, EMA200 {long_val:.2f}; "
            f"ROC10 {roc_val:.2%}, RSI14 {rsi_val:.1f}."
        )

        constraints: dict[str, Any] = {
            "allow_short": allow_short,
            "max_short_exposure_pct": round(max_short_pct, 4) if allow_short else 0.0,
            "momentum_score": round(score, 3),
            "trend_bias": signal,
        }
        if not allow_short:
            constraints["block_new_shorts"] = True

        momentum_view[ticker] = {
            "signal": signal,
            "confidence": int(np.clip(confidence, 0, 100)),
            "reasoning": reasoning,
            "constraints": constraints,
            "metrics": {
                "price": current_price,
                "ema21": fast_val,
                "ema55": slow_val,
                "ema200": long_val,
                "roc_10": roc_val,
                "rsi_14": rsi_val,
            },
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
