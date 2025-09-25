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


def _calculate_atr(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = prices_df["high"].astype(float)
    low = prices_df["low"].astype(float)
    close = prices_df["close"].astype(float)
    prev_close = close.shift(1)

    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr.bfill()


##### Stop-Loss Guardian Analyst #####
def stop_loss_guardian_agent(state: AgentState, agent_id: str = "stop_loss_guardian_agent"):
    """Monitors active positions and recommends forced covers when losses breach thresholds."""

    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    portfolio = data["portfolio"]
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    position_monitor: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Evaluating stop-loss bands")

        position = (portfolio.get("positions") or {}).get(ticker, {})
        short_shares = int(position.get("short", 0) or 0)
        avg_entry = _safe_float(position.get("short_cost_basis", 0.0), 0.0)

        if short_shares <= 0 or avg_entry <= 0:
            position_monitor[ticker] = {
                "signal": "no_short_position",
                "confidence": 0,
                "reasoning": "No active short to monitor",
                "constraints": {},
            }
            continue

        prices = get_prices(ticker=ticker, start_date=start_date, end_date=end_date, api_key=api_key)
        if not prices:
            position_monitor[ticker] = {
                "signal": "data_unavailable",
                "confidence": 50,
                "reasoning": "Missing price data; maintain manual vigilance",
                "constraints": {"block_new_shorts": True},
            }
            continue

        prices_df = prices_to_df(prices)
        if prices_df.empty:
            position_monitor[ticker] = {
                "signal": "data_unavailable",
                "confidence": 50,
                "reasoning": "Missing price data; maintain manual vigilance",
                "constraints": {"block_new_shorts": True},
            }
            continue

        close = prices_df["close"].dropna()
        current_price = _safe_float(close.iloc[-1], avg_entry)
        atr_series = _calculate_atr(prices_df)
        atr_latest = _safe_float(atr_series.iloc[-1], 0.0)
        atr_pct = atr_latest / current_price if current_price > 0 else 0.0

        loss_pct = (current_price - avg_entry) / avg_entry if avg_entry > 0 else 0.0
        loss_pct = float(loss_pct)

        base_threshold = max(0.03, atr_pct * 1.5)
        escalated_threshold = base_threshold * 1.5

        constraints: dict[str, Any] = {
            "loss_pct": round(loss_pct, 4),
            "threshold_pct": round(base_threshold, 4),
            "atr_pct": round(atr_pct, 4),
        }

        if loss_pct >= escalated_threshold:
            force_qty = short_shares
            constraints.update(
                {
                    "force_cover": True,
                    "force_cover_qty": force_qty,
                    "block_new_shorts": True,
                    "target_short_shares": 0,
                }
            )
            signal = "force_exit"
            confidence = 95
            reasoning = (
                f"Loss {loss_pct:.2%} exceeds escalated threshold {escalated_threshold:.2%}; cover entire short"
            )
        elif loss_pct >= base_threshold:
            force_qty = max(1, int(np.ceil(short_shares * 0.5)))
            remaining = max(0, short_shares - force_qty)
            constraints.update(
                {
                    "force_cover": True,
                    "force_cover_qty": force_qty,
                    "block_new_shorts": True,
                    "target_short_shares": remaining,
                }
            )
            signal = "trim_position"
            confidence = 85
            reasoning = (
                f"Loss {loss_pct:.2%} breached stop {base_threshold:.2%}; cover {force_qty} shares"
            )
        else:
            constraints.update({"force_cover": False, "target_short_shares": short_shares})
            signal = "within_tolerance"
            confidence = int(max(40, 70 - (loss_pct / base_threshold * 30))) if base_threshold > 0 else 60
            reasoning = f"Loss {loss_pct:.2%} within stop band {base_threshold:.2%}; monitor"

        position_monitor[ticker] = {
            "signal": signal,
            "confidence": int(np.clip(confidence, 0, 100)),
            "reasoning": reasoning,
            "constraints": constraints,
            "metrics": {
                "current_price": current_price,
                "avg_entry": avg_entry,
                "short_shares": short_shares,
                "loss_pct": loss_pct,
                "base_threshold": base_threshold,
                "escalated_threshold": escalated_threshold,
                "atr_pct": atr_pct,
            },
        }

    message = HumanMessage(content=json.dumps(position_monitor), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(position_monitor, "Stop-Loss Guardian")

    state["data"].setdefault("analyst_signals", {})[agent_id] = position_monitor
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }
