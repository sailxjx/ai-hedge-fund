from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_company_news
from src.utils.api_key import get_api_key_from_state
from src.utils.llm import call_llm
from src.utils.progress import progress


MAX_NEWS_ITEMS = 25


CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "earnings": (
        "earnings",
        "guidance",
        "quarter",
        "q1",
        "q2",
        "q3",
        "q4",
        "eps",
        "revenue",
        "results",
    ),
    "product": (
        "product",
        "launch",
        "rollout",
        "innovation",
        "update",
        "chip",
        "platform",
        "release",
        "pipeline",
    ),
    "deal_activity": (
        "merger",
        "acquisition",
        "buy",
        "sell",
        "stake",
        "deal",
        "partnership",
        "investment",
    ),
    "regulation": (
        "regulation",
        "lawsuit",
        "probe",
        "investigation",
        "antitrust",
        "compliance",
        "sec",
        "ftc",
        "doj",
    ),
    "management": (
        "ceo",
        "cfo",
        "leadership",
        "resigns",
        "steps down",
        "executive",
        "board",
    ),
    "capital_markets": (
        "dividend",
        "buyback",
        "offering",
        "issuance",
        "capital",
        "debt",
        "rating",
        "downgrade",
        "upgrade",
    ),
}

POSITIVE_KEYWORDS: tuple[str, ...] = (
    "beat",
    "beats",
    "raises",
    "increase",
    "upgrades",
    "record",
    "strong",
    "improve",
    "growth",
    "surge",
    "boost",
    "approves",
    "wins",
)

NEGATIVE_KEYWORDS: tuple[str, ...] = (
    "miss",
    "misses",
    "cut",
    "cuts",
    "downgrade",
    "slump",
    "drops",
    "fall",
    "falls",
    "weak",
    "decline",
    "delay",
    "investigation",
    "probe",
    "lawsuit",
    "recall",
)

UPCOMING_KEYWORDS: tuple[str, ...] = (
    "will announce",
    "to announce",
    "ahead of",
    "expected to",
    "set to",
    "scheduled",
    "upcoming",
    "plans to",
    "previews",
    "forecast",
)

RECENT_WINDOW_DAYS = 3


class CatalystSnapshot(BaseModel):
    label: str
    category: str
    timing: Literal["upcoming", "recent", "past"]
    direction: Literal["positive", "negative", "uncertain"]
    conviction: float = 0.0
    detail: str


class EventCatalystSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(default=40.0)
    reasoning: str
    catalysts: list[CatalystSnapshot] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] | None = None


@dataclass
class ArticleFeature:
    title: str
    date: str
    source: str
    sentiment: str | None
    category: str
    timing: Literal["upcoming", "recent", "past"]
    tone: Literal["positive", "negative", "uncertain"]


def _safe_lower(value: str | None) -> str:
    return value.lower() if value else ""


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    sanitized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(sanitized)
    except ValueError:
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            return None


def _categorize_headline(title: str) -> str:
    lowered = _safe_lower(title)
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return "other"


def _infer_tone(title: str, sentiment: str | None) -> Literal["positive", "negative", "uncertain"]:
    if sentiment and sentiment.lower() in {"positive", "negative"}:
        return sentiment.lower()  # type: ignore[return-value]

    lowered = _safe_lower(title)
    if any(keyword in lowered for keyword in POSITIVE_KEYWORDS):
        return "positive"
    if any(keyword in lowered for keyword in NEGATIVE_KEYWORDS):
        return "negative"
    return "uncertain"


def _classify_timing(
    title: str,
    article_dt: datetime | None,
    reference_dt: datetime | None,
) -> Literal["upcoming", "recent", "past"]:
    lowered = _safe_lower(title)
    if any(phrase in lowered for phrase in UPCOMING_KEYWORDS):
        return "upcoming"

    if not article_dt or not reference_dt:
        return "past"

    article_dt_local = (
        article_dt.astimezone(timezone.utc).replace(tzinfo=None)
        if article_dt.tzinfo is not None
        else article_dt
    )
    reference_dt_local = (
        reference_dt.astimezone(timezone.utc).replace(tzinfo=None)
        if reference_dt.tzinfo is not None
        else reference_dt
    )
    delta = (reference_dt_local - article_dt_local).days
    if delta <= RECENT_WINDOW_DAYS:
        return "recent"
    return "past"


def _build_article_features(
    news_items: Iterable,
    reference_dt: datetime | None,
) -> tuple[list[ArticleFeature], dict[str, Any], dict[str, Any]]:
    features: list[ArticleFeature] = []
    category_summary: dict[str, Any] = {}
    tone_counter: Counter[str] = Counter()

    for item in news_items:
        article_dt = _parse_datetime(getattr(item, "date", None))
        category = _categorize_headline(getattr(item, "title", ""))
        tone = _infer_tone(getattr(item, "title", ""), getattr(item, "sentiment", None))
        timing = _classify_timing(getattr(item, "title", ""), article_dt, reference_dt)

        feature = ArticleFeature(
            title=getattr(item, "title", ""),
            date=getattr(item, "date", ""),
            source=getattr(item, "source", ""),
            sentiment=getattr(item, "sentiment", None),
            category=category,
            timing=timing,
            tone=tone,
        )
        features.append(feature)
        tone_counter[tone] += 1

        summary = category_summary.setdefault(
            category,
            {
                "articles": 0,
                "tone_counts": {"positive": 0, "negative": 0, "uncertain": 0},
                "timing_counts": {"upcoming": 0, "recent": 0, "past": 0},
                "headlines": [],
                "latest": None,
            },
        )
        summary["articles"] += 1
        summary["tone_counts"][tone] += 1
        summary["timing_counts"][timing] += 1
        if len(summary["headlines"]) < 3 and feature.title:
            summary["headlines"].append(feature.title)

        if article_dt:
            latest_dt = summary.get("_latest_dt")
            if not latest_dt or article_dt > latest_dt:
                summary["_latest_dt"] = article_dt
                summary["latest"] = feature.date

    overview = {
        "total_articles": len(features),
        "tone_counts": dict(tone_counter),
        "categories": list(category_summary.keys()),
        "upcoming_events": sum(
            cat_summary["timing_counts"].get("upcoming", 0)
            for cat_summary in category_summary.values()
        ),
    }

    for cat_summary in category_summary.values():
        if "_latest_dt" in cat_summary:
            del cat_summary["_latest_dt"]

    return features, category_summary, overview


def _summarise_news_signals(features: list[ArticleFeature]) -> dict[str, Any]:
    tone_counts: Counter[str] = Counter()
    upcoming_tone: Counter[str] = Counter()

    for feature in features:
        tone_counts[feature.tone] += 1
        if feature.timing == "upcoming":
            upcoming_tone[feature.tone] += 1

    total = sum(tone_counts.values())
    negative = tone_counts.get("negative", 0)
    positive = tone_counts.get("positive", 0)
    neutral = tone_counts.get("neutral", 0)
    upcoming_negative = upcoming_tone.get("negative", 0)
    upcoming_positive = upcoming_tone.get("positive", 0)

    negativity_ratio = float(negative / total) if total else 0.0
    positivity_ratio = float(positive / total) if total else 0.0

    return {
        "total": total,
        "negative": negative,
        "positive": positive,
        "neutral": neutral,
        "net": positive - negative,
        "negativity_ratio": negativity_ratio,
        "positivity_ratio": positivity_ratio,
        "upcoming_negative": upcoming_negative,
        "upcoming_positive": upcoming_positive,
    }


def _harmonise_constraints(
    *,
    signal: str | None,
    constraints: dict[str, Any] | None,
    news_stats: dict[str, Any],
) -> dict[str, Any] | None:
    signal_value = (signal or "neutral").lower()
    working: dict[str, Any] = dict(constraints or {})

    net = int(news_stats.get("net", 0))
    negativity_ratio = float(news_stats.get("negativity_ratio", 0.0))
    upcoming_negative = int(news_stats.get("upcoming_negative", 0))
    upcoming_positive = int(news_stats.get("upcoming_positive", 0))
    total = int(news_stats.get("total", 0))

    def _set_default(key: str, value: Any) -> None:
        if key not in working:
            working[key] = value

    if signal_value == "bearish":
        working["preferred_direction"] = "short"
        working["max_long_add"] = 0
        _set_default("max_short_exposure_pct", 0.12)
        if upcoming_negative > 0 or negativity_ratio >= 0.4:
            _set_default("reduce_position_change", True)
    elif signal_value == "bullish":
        working["preferred_direction"] = "long"
        if "max_short_exposure_pct" in working:
            try:
                working["max_short_exposure_pct"] = max(0.0, float(working["max_short_exposure_pct"]))
            except (TypeError, ValueError):
                working["max_short_exposure_pct"] = 0.05
        else:
            working["max_short_exposure_pct"] = 0.05
        if total and negativity_ratio >= 0.45:
            working["max_long_add"] = min(int(working.get("max_long_add", 0) or 0), 0)
            _set_default("reduce_position_change", True)
    else:
        if net <= -1 and total >= 3:
            working["preferred_direction"] = "avoid_large_long"
            working["max_long_add"] = min(int(working.get("max_long_add", 0) or 0), 0)
            _set_default("max_short_exposure_pct", 0.09)
            if upcoming_negative > upcoming_positive:
                _set_default("reduce_position_change", True)
        elif net >= 1 and total >= 3:
            working.setdefault("preferred_direction", "long")
            _set_default("max_short_exposure_pct", 0.05)
        elif negativity_ratio >= 0.5 and total >= 2:
            working["preferred_direction"] = "short"
            working["max_long_add"] = 0
            _set_default("max_short_exposure_pct", 0.11)

    preferred = working.get("preferred_direction")
    if signal_value == "bearish" and preferred == "long":
        working["preferred_direction"] = "short"
        working["max_long_add"] = 0
        _set_default("max_short_exposure_pct", 0.12)
    if signal_value == "bullish" and preferred == "short":
        working["preferred_direction"] = "long"

    cleaned = {key: value for key, value in working.items() if value is not None}
    return cleaned or None


def _construct_prompt(
    ticker: str,
    end_date: str | None,
    features: list[ArticleFeature],
    category_summary: dict[str, Any],
    overview: dict[str, Any],
    context: dict[str, Any],
) -> Any:
    digest_payload = [
        {
            "title": feature.title,
            "date": feature.date,
            "source": feature.source,
            "category": feature.category,
            "timing": feature.timing,
            "tone": feature.tone,
            "sentiment": feature.sentiment,
        }
        for feature in features
    ]

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an event-driven catalyst analyst for an equity long/short fund. "
                "Use the provided event flow to determine near-term directional bias. "
                "Highlight catalysts that can move the stock over the next 1-3 weeks, "
                "incorporating risk or regime context when available. Always return JSON without extra prose.",
            ),
            (
                "human",
                "Ticker: {ticker}\n"
                "Analysis end date: {end_date}\n\n"
                "Event overview:\n{overview}\n\n"
                "Category breakdown:\n{category_summary}\n\n"
                "Top headlines (max {max_news}):\n{digest}\n\n"
                "Regime & guard-rail context:\n{context}\n\n"
                "Produce JSON with the following shape:\n"
                "{{\n"
                '  "signal": "bullish|bearish|neutral",\n'
                '  "confidence": number 0-100,\n'
                '  "reasoning": "focus on catalysts",\n'
                '  "catalysts": [\n'
                "    {{\n"
                '      "label": "short title",\n'
                '      "category": "earnings|product|deal_activity|regulation|management|capital_markets|other",\n'
                '      "timing": "upcoming|recent|past",\n'
                '      "direction": "positive|negative|uncertain",\n'
                '      "conviction": 0.0-1.0,\n'
                '      "detail": "one sentence impact"\n'
                "    }}\n"
                "  ],\n"
                '  "risk_flags": ["concise risks"],\n'
                '  "constraints": {{"preferred_direction": str, "max_short_exposure_pct": float, "max_long_add": float}}\n'
                "}}\n"
                "Keep catalysts list to the 3-4 most material events and note if timing is upcoming versus in the past.",
            ),
        ]
    )

    prompt_variables = {
        "ticker": ticker,
        "end_date": end_date or "unknown",
        "overview": json.dumps(overview, ensure_ascii=False, separators=(",", ":")),
        "category_summary": json.dumps(category_summary, ensure_ascii=False, separators=(",", ":")),
        "digest": json.dumps(digest_payload, ensure_ascii=False, separators=(",", ":")),
        "context": json.dumps(context or {}, ensure_ascii=False, separators=(",", ":")),
        "max_news": len(digest_payload),
    }
    return template.invoke(prompt_variables)


def _gather_context(analyst_signals: dict[str, Any], ticker: str) -> dict[str, Any]:
    relevant_agents = {
        "regime_meta_agent",
        "trend_regime_agent",
        "short_squeeze_guardian_agent",
        "momentum_guardian_agent",
        "stat_mean_reversion_agent",
        "growth_momentum_agent",
        "risk_manager_agent",
    }

    context: dict[str, Any] = {}
    for agent_name, payload in analyst_signals.items():
        if agent_name not in relevant_agents:
            continue
        ticker_payload = payload.get(ticker)
        if not isinstance(ticker_payload, dict):
            continue
        snapshot = {
            key: ticker_payload.get(key)
            for key in ("signal", "confidence", "constraints")
            if key in ticker_payload
        }
        if snapshot:
            context[agent_name] = snapshot
    return context


def _fallback_signal(
    ticker: str,
    features: list[ArticleFeature],
    overview: dict[str, Any],
) -> EventCatalystSignal:
    tone_counts = overview.get("tone_counts", {})
    positive = tone_counts.get("positive", 0)
    negative = tone_counts.get("negative", 0)
    diff = positive - negative

    if diff > 1:
        signal = "bullish"
        confidence = min(75.0, 50.0 + diff * 6)
        rationale = f"Fallback: positive catalysts outweigh negative ({positive}:{negative})."
    elif diff < -1:
        signal = "bearish"
        confidence = min(75.0, 50.0 + abs(diff) * 6)
        rationale = f"Fallback: negative catalysts dominate positive ({negative}:{positive})."
    else:
        signal = "neutral"
        confidence = 40.0 + abs(diff) * 4
        rationale = "Fallback: mixed catalyst tone keeps stance neutral."

    catalysts_payload = []
    for feature in features[:3]:
        catalysts_payload.append(
            CatalystSnapshot(
                label=feature.title[:80] if feature.title else "Headline",
                category=feature.category,
                timing=feature.timing,
                direction=feature.tone,
                conviction=0.4 if feature.tone == "positive" else 0.35 if feature.tone == "negative" else 0.25,
                detail=f"{feature.source or 'Source'} {feature.date}: {feature.title}",
            )
        )

    risk_flags = []
    if overview.get("upcoming_events", 0) > 0:
        risk_flags.append("Upcoming catalyst requires position sizing discipline")

    return EventCatalystSignal(
        signal=signal,
        confidence=confidence,
        reasoning=rationale,
        catalysts=catalysts_payload,
        risk_flags=risk_flags,
    )


def event_catalyst_agent(state: AgentState, agent_id: str = "event_catalyst_agent"):
    data = state.get("data", {})
    tickers: list[str] = data.get("tickers", [])
    end_date: str | None = data.get("end_date")
    reference_dt = _parse_datetime(end_date)
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    analyst_signals = data.setdefault("analyst_signals", {})

    results: dict[str, Any] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching company news")
        news_items = get_company_news(
            ticker=ticker,
            end_date=end_date,
            limit=MAX_NEWS_ITEMS * 2,
            api_key=api_key,
        )

        sliced_news = list(news_items)[:MAX_NEWS_ITEMS]
        features, category_summary, overview = _build_article_features(sliced_news, reference_dt)

        context = _gather_context(analyst_signals, ticker)

        if features:
            progress.update_status(agent_id, ticker, "Evaluating catalysts")
            prompt = _construct_prompt(ticker, end_date, features, category_summary, overview, context)
        else:
            progress.update_status(agent_id, ticker, "No fresh news - using fallback")
            prompt = None

        def _default_factory() -> EventCatalystSignal:
            return _fallback_signal(ticker, features, overview)

        if prompt is not None and features:
            llm_result = call_llm(
                prompt=prompt,
                pydantic_model=EventCatalystSignal,
                agent_name=agent_id,
                state=state,
                default_factory=_default_factory,
            )
        else:
            llm_result = _default_factory()

        payload = llm_result.model_dump()
        payload["confidence"] = float(max(0.0, min(100.0, payload.get("confidence", 0.0))))

        catalysts = []
        for catalyst in payload.get("catalysts", []) or []:
            conv = float(catalyst.get("conviction", 0.0))
            catalyst["conviction"] = max(0.0, min(1.0, conv))
            timing = catalyst.get("timing")
            if timing not in {"upcoming", "recent", "past"}:
                catalyst["timing"] = "past"
            direction = catalyst.get("direction")
            if direction not in {"positive", "negative", "uncertain"}:
                catalyst["direction"] = "uncertain"
            catalysts.append(catalyst)
        payload["catalysts"] = catalysts

        news_stats = _summarise_news_signals(features)
        harmonised = _harmonise_constraints(
            signal=payload.get("signal"),
            constraints=payload.get("constraints"),
            news_stats=news_stats,
        )

        if harmonised is None and overview.get("upcoming_events", 0) > 0:
            harmonised = {"reduce_position_change": True}

        payload["constraints"] = harmonised

        results[ticker] = payload

        summary = f"{payload['signal'].upper()} @ {payload['confidence']:.0f}%"
        progress.update_status(agent_id, ticker, summary, analysis=payload.get("reasoning", ""))

    analyst_signals[agent_id] = results

    message = HumanMessage(content=json.dumps(results, ensure_ascii=False), name=agent_id)
    if state.get("metadata", {}).get("show_reasoning"):
        show_agent_reasoning(results, "Event Catalyst Agent")

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": data}


__all__ = [
    "event_catalyst_agent",
    "EventCatalystSignal",
]
