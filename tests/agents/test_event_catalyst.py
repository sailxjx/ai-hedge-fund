import json
from datetime import datetime, timedelta

import pytest

from src.agents.event_catalyst import event_catalyst_agent, EventCatalystSignal
from src.data.models import CompanyNews


def _base_state() -> dict:
    return {
        "messages": [],
        "data": {
            "tickers": ["TSLA"],
            "end_date": "2025-09-28",
            "analyst_signals": {},
        },
        "metadata": {"show_reasoning": False},
    }


def test_event_catalyst_agent_uses_llm_payload(monkeypatch):
    news_items = [
        CompanyNews(
            ticker="TSLA",
            title="Tesla raises production guidance ahead of Q3 earnings",
            author="Reporter",
            source="Newswire",
            date="2025-09-25T12:00:00Z",
            url="https://example.com/tesla-guidance",
            sentiment="positive",
        ),
    ]

    monkeypatch.setattr(
        "src.agents.event_catalyst.get_company_news",
        lambda *_, **__: news_items,
    )

    llm_output = EventCatalystSignal(
        signal="bullish",
        confidence=78.0,
        reasoning="Guidance hike ahead of earnings signals upside skew.",
        catalysts=[
            {
                "label": "Production guidance raised",
                "category": "earnings",
                "timing": "upcoming",
                "direction": "positive",
                "conviction": 0.8,
                "detail": "Newswire 2025-09-25: Tesla raises production guidance ahead of Q3 earnings",
            }
        ],
        risk_flags=["Watch for execution slippage"],
        constraints={"preferred_direction": "long"},
    )

    def _fake_call_llm(**kwargs):  # type: ignore
        return llm_output

    monkeypatch.setattr("src.agents.event_catalyst.call_llm", _fake_call_llm)

    state = _base_state()
    result = event_catalyst_agent(state)

    payload = result["data"]["analyst_signals"]["event_catalyst_agent"]["TSLA"]
    assert payload["signal"] == "bullish"
    assert payload["confidence"] == pytest.approx(78.0)
    assert payload["catalysts"][0]["direction"] == "positive"
    assert payload["constraints"]["preferred_direction"] == "long"


def test_event_catalyst_constraints_follow_bearish_signal(monkeypatch):
    news_items = [
        CompanyNews(
            ticker="TSLA",
            title="Tesla will announce recall review after fatal crash",
            author="Reporter",
            source="Newswire",
            date="2025-09-27T12:00:00Z",
            url="https://example.com/tesla-recall-review",
            sentiment="negative",
        ),
        CompanyNews(
            ticker="TSLA",
            title="Investigators escalate probe into Tesla safety systems",
            author="Reporter",
            source="Newswire",
            date="2025-09-26T09:00:00Z",
            url="https://example.com/tesla-probe",
            sentiment="negative",
        ),
    ]

    monkeypatch.setattr(
        "src.agents.event_catalyst.get_company_news",
        lambda *_, **__: news_items,
    )

    llm_output = EventCatalystSignal(
        signal="bearish",
        confidence=68.0,
        reasoning="Negative safety headlines dominate and raise recall risk.",
        catalysts=[
            {
                "label": "Recall review",
                "category": "regulation",
                "timing": "upcoming",
                "direction": "negative",
                "conviction": 0.9,
                "detail": "Tesla will announce recall review after fatal crash",
            }
        ],
        risk_flags=["Regulatory escalation"],
        constraints={"preferred_direction": "long", "max_long_add": 5},
    )

    def _fake_call_llm(**kwargs):  # type: ignore
        return llm_output

    monkeypatch.setattr("src.agents.event_catalyst.call_llm", _fake_call_llm)

    state = _base_state()
    result = event_catalyst_agent(state)

    payload = result["data"]["analyst_signals"]["event_catalyst_agent"]["TSLA"]
    constraints = payload["constraints"]

    assert payload["signal"] == "bearish"
    assert constraints["preferred_direction"] == "short"
    assert constraints["max_long_add"] == 0
    assert constraints["max_short_exposure_pct"] >= 0.11
    assert constraints.get("reduce_position_change") is True


def test_event_catalyst_agent_fallback_applies_when_llm_unavailable(monkeypatch):
    base_time = datetime(2025, 9, 26, 15, 0)
    positive_news = CompanyNews(
        ticker="TSLA",
        title="Tesla wins major fleet order from logistics customer",
        author="Reporter",
        source="Newswire",
        date=(base_time - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        url="https://example.com/tesla-order",
        sentiment="positive",
    )
    negative_news = CompanyNews(
        ticker="TSLA",
        title="Tesla faces investigation into autopilot incidents",
        author="Reporter",
        source="Newswire",
        date=(base_time - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        url="https://example.com/tesla-investigation",
        sentiment="negative",
    )

    monkeypatch.setattr(
        "src.agents.event_catalyst.get_company_news",
        lambda *_, **__: [positive_news, negative_news],
    )

    def _fallback_call_llm(**kwargs):  # type: ignore
        default_factory = kwargs["default_factory"]
        return default_factory()

    monkeypatch.setattr("src.agents.event_catalyst.call_llm", _fallback_call_llm)

    state = _base_state()
    output = event_catalyst_agent(state)
    payload = output["data"]["analyst_signals"]["event_catalyst_agent"]["TSLA"]

    assert payload["signal"] in {"bullish", "bearish", "neutral"}
    assert payload["confidence"] >= 40.0
    assert payload["reasoning"].startswith("Fallback:")
    assert json.loads(json.dumps(payload))  # Ensure payload is JSON serialisable
    assert payload["risk_flags"] or payload["catalysts"]
