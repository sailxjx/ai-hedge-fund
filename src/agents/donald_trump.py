from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Iterable

import pandas as pd
from langchain_core.messages import HumanMessage

from src.agents.persona_utils import async_persona_from_observations, persona_from_observations
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import (
    get_company_news,
    get_company_news_async,
    get_prices,
    get_prices_async,
    prices_to_df,
    prices_to_df_async,
)
from src.utils.api_key import get_api_key_from_state
from src.utils.async_state import update_analyst_signals_async
from src.utils.progress import progress


PERSONA_NAME = "Donald Trump"
PERSONA_ROLE = "a politically charged dealmaker who trades policy shocks and media momentum"
PERSONA_BACKSTORY = (
    "You campaign from the trading desk, triangulating tariffs, tax policy, and public sentiment to stay ahead of market-moving headlines."
)
PERSONA_INSTRUCTIONS: Iterable[str] = (
    "Speak in first person with the cadence of a campaign rally—confident, direct, and decisive.",
    "Focus on catalysts such as tariffs, regulation, energy policy, and social media narrative velocity.",
    "Weigh both the price momentum and media tone when assigning a signal, and specify constraints when risk feels one-sided.",
    "Limit output to the provided JSON keys and do not invent data beyond the supplied observations.",
)
ALLOWED_SIGNALS = ["bullish", "bearish", "neutral"]

POLICY_KEYWORDS = ["tariff", "tax", "sanction", "deal", "china", "regulation", "energy", "oil", "border"]
SOCIAL_KEYWORDS = ["tweet", "post", "truth social", "rally", "poll"]


def _shift_date(date_str: str | None, days: int) -> str | None:
    if not date_str:
        return None
    try:
        base_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None
    return (base_date + timedelta(days=days)).strftime("%Y-%m-%d")


def _extend_start(start_date: str | None, buffer_days: int = 90) -> str | None:
    return _shift_date(start_date, -buffer_days) if start_date else None


def _build_price_context(price_df: pd.DataFrame) -> dict[str, Any]:
    if price_df.empty or "close" not in price_df.columns:
        return {"available": False, "note": "Insufficient price data"}

    closes = price_df["close"].astype(float)
    returns = closes.pct_change()
    latest_close = float(closes.iloc[-1])

    def compute_window_return(window: int) -> float | None:
        if len(closes) <= window:
            return None
        start_value = closes.iloc[-window - 1]
        if start_value == 0:
            return None
        return float(closes.iloc[-1] / start_value - 1)

    daily_ret = returns.iloc[-1] if len(returns) else None
    vol_window = returns.rolling(10).std(ddof=0)
    vol_10 = vol_window.iloc[-1] if len(vol_window.dropna()) else None

    return {
        "available": True,
        "last_close": latest_close,
        "daily_return": float(daily_ret) if pd.notna(daily_ret) else None,
        "weekly_return": compute_window_return(5),
        "monthly_return": compute_window_return(21),
        "rolling_volatility_10": float(vol_10) if pd.notna(vol_10) else None,
    }


def _build_news_summary(news_items: list) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not news_items:
        summary = {
            "total_items": 0,
            "sentiment_counts": {},
            "latest_timestamp": None,
            "policy_keyword_hits": {},
            "social_keyword_hits": {},
        }
        return summary, []

    sentiments = Counter()
    policy_hits = Counter()
    social_hits = Counter()
    latest_timestamp: str | None = None
    headline_samples: list[dict[str, Any]] = []

    sorted_items = sorted(news_items, key=lambda item: item.date or "", reverse=True)

    for item in sorted_items:
        sentiment = (item.sentiment or "unknown").lower()
        sentiments[sentiment] += 1

        title_text = getattr(item, "title", "") or ""
        summary_text = getattr(item, "summary", "") or ""
        combined_text = f"{title_text} {summary_text}".lower()
        for keyword in POLICY_KEYWORDS:
            if keyword in combined_text:
                policy_hits[keyword] += 1
        for keyword in SOCIAL_KEYWORDS:
            if keyword in combined_text:
                social_hits[keyword] += 1

        if not latest_timestamp:
            latest_timestamp = item.date

    for item in sorted_items[:5]:
        headline_samples.append(
            {
                "title": getattr(item, "title", None),
                "sentiment": item.sentiment,
                "date": item.date,
                "source": item.source,
                "summary": getattr(item, "summary", None),
            }
        )

    summary = {
        "total_items": len(news_items),
        "sentiment_counts": dict(sentiments),
        "latest_timestamp": latest_timestamp,
        "policy_keyword_hits": dict(policy_hits),
        "social_keyword_hits": dict(social_hits),
    }
    return summary, headline_samples


def _prepare_observations(
    *,
    ticker: str,
    price_df: pd.DataFrame,
    news_items: list,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    price_context = _build_price_context(price_df.sort_index()) if not price_df.empty else {"available": False}
    news_summary, headline_samples = _build_news_summary(news_items)
    observations = {
        "ticker": ticker,
        "price_context": price_context,
        "news_summary": news_summary,
        "headline_samples": headline_samples,
    }
    return observations, headline_samples


def _finalize_payload(
    *,
    decision,
    observations: dict[str, Any],
    headline_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    confidence = int(max(0, min(round(decision.confidence), 100)))
    constraints = decision.constraints or {}
    return {
        "signal": decision.signal,
        "confidence": confidence,
        "reasoning": decision.reasoning,
        "constraints": constraints,
        "meta": {
            "observations": observations,
            "headline_samples": headline_samples,
        },
    }


def donald_trump_agent(state: AgentState, agent_id: str = "donald_trump_agent"):
    """Synchronous Donald Trump persona leveraging observation-only prompting."""

    data = state.get("data", {})
    tickers = data.get("tickers", [])
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    extended_start = _extend_start(start_date, 120)
    news_start = _shift_date(end_date, -21) if end_date else None

    signals: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Gathering price context")
        prices = get_prices(
            ticker=ticker,
            start_date=extended_start or start_date,
            end_date=end_date,
            api_key=api_key,
        )
        price_df = prices_to_df(prices) if prices else pd.DataFrame()

        progress.update_status(agent_id, ticker, "Collecting news flow")
        news_items = get_company_news(
            ticker=ticker,
            end_date=end_date,
            start_date=news_start,
            limit=100,
            api_key=api_key,
        )

        observations, headline_samples = _prepare_observations(
            ticker=ticker,
            price_df=price_df,
            news_items=news_items,
        )

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

        payload = _finalize_payload(
            decision=decision,
            observations=observations,
            headline_samples=headline_samples,
        )
        progress.update_status(agent_id, ticker, f"Decision → {payload['signal'].upper()} ({payload['confidence']})")
        signals[ticker] = payload

    message = HumanMessage(
        content=json.dumps(signals),
        name=agent_id,
    )

    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(signals, "Donald Trump Analyst")

    state.setdefault("data", {}).setdefault("analyst_signals", {})[agent_id] = signals
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


async def donald_trump_agent_async(state: AgentState, agent_id: str = "donald_trump_agent"):
    """Async Donald Trump persona for async orchestration mode."""

    data = state.get("data", {})
    tickers = data.get("tickers", [])
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")

    extended_start = _extend_start(start_date, 120)
    news_start = _shift_date(end_date, -21) if end_date else None

    signals: dict[str, dict[str, Any]] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Gathering price context")
        prices = await get_prices_async(
            ticker=ticker,
            start_date=extended_start or start_date,
            end_date=end_date,
            api_key=api_key,
        )
        price_df = await prices_to_df_async(prices) if prices else pd.DataFrame()

        progress.update_status(agent_id, ticker, "Collecting news flow")
        news_items = await get_company_news_async(
            ticker=ticker,
            end_date=end_date,
            start_date=news_start,
            limit=100,
            api_key=api_key,
        )

        observations, headline_samples = _prepare_observations(
            ticker=ticker,
            price_df=price_df,
            news_items=news_items,
        )

        decision = await async_persona_from_observations(
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

        payload = _finalize_payload(
            decision=decision,
            observations=observations,
            headline_samples=headline_samples,
        )

        progress.update_status(agent_id, ticker, f"Decision → {payload['signal'].upper()} ({payload['confidence']})")
        signals[ticker] = payload

    message = HumanMessage(
        content=json.dumps(signals),
        name=agent_id,
    )

    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(signals, "Donald Trump Analyst")

    await update_analyst_signals_async(state, agent_id, signals)
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }
