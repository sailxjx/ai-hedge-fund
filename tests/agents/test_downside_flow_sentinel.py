from datetime import datetime, timedelta

from src.agents.downside_flow_sentinel import downside_flow_sentinel_agent
from src.agents.persona_utils import PersonaDecision
from src.data.models import Price


def _build_price_series(count: int = 60, base_price: float = 300.0, step: float = -1.0) -> list[Price]:
    start = datetime(2024, 1, 1)
    series: list[Price] = []
    for idx in range(count):
        level = base_price + step * idx
        stamp = (start + timedelta(days=idx)).strftime("%Y-%m-%dT00:00:00Z")
        series.append(
            Price(
                open=level,
                close=level,
                high=level * 1.01,
                low=level * 0.99,
                volume=1_000_000,
                time=stamp,
            )
        )
    return series


def _base_state(long_shares: int = 0, short_shares: int = 0) -> dict:
    return {
        "messages": [],
        "data": {
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "portfolio": {
                "cash": 125_000.0,
                "positions": {
                    "TSLA": {
                        "long": long_shares,
                        "short": short_shares,
                        "long_cost_basis": 0.0,
                        "short_cost_basis": 0.0,
                    }
                },
            },
            "analyst_signals": {},
        },
        "metadata": {"show_reasoning": False},
    }


def _persona_stub_factory(signal: str, confidence: float = 70.0, constraints: dict | None = None):
    captured: dict[str, dict] = {}

    def _stub(*, observations=None, **_):
        nonlocal captured
        captured["observations"] = observations or {}
        return PersonaDecision(
            signal=signal,
            confidence=confidence,
            reasoning=f"Stubbed {signal} decision",
            constraints=constraints or {},
        )

    return captured, _stub


def test_downside_metrics_forwarded_to_persona(monkeypatch):
    prices = _build_price_series()
    monkeypatch.setattr("src.agents.downside_flow_sentinel.get_prices", lambda *_, **__: prices)
    captured, persona_stub = _persona_stub_factory(
        signal="crash_flow",
        confidence=91.0,
        constraints={"preferred_direction": "short", "max_long_exposure_pct": 0.05},
    )
    monkeypatch.setattr("src.agents.downside_flow_sentinel.persona_from_observations", persona_stub)

    metrics = {
        "ret_1": -0.035,
        "ret_3": -0.062,
        "ret_5": -0.118,
        "ret_10": -0.185,
        "ret_20": -0.244,
        "drawdown_20": -0.19,
        "drawdown_40": -0.27,
        "vol_ratio": 0.74,
        "downside_share10": 0.9,
        "tail_loss20": -0.06,
        "trend_slope10": -0.018,
    }
    monkeypatch.setattr("src.agents.downside_flow_sentinel._compute_metrics", lambda df: metrics)

    state = _base_state(long_shares=120)
    result = downside_flow_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["downside_flow_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "crash_flow"
    assert payload["confidence"] == 91
    assert payload["constraints"] == {"preferred_direction": "short", "max_long_exposure_pct": 0.05}
    assert "score" not in payload

    observations = captured["observations"]
    assert observations["data_status"] == "data_ready"
    assert observations["downside_metrics"] == metrics
    assert observations["existing_long_shares"] == 120
    assert "auto_signal_hint" not in observations


def test_existing_position_context_is_preserved(monkeypatch):
    prices = _build_price_series(step=-0.5)
    monkeypatch.setattr("src.agents.downside_flow_sentinel.get_prices", lambda *_, **__: prices)
    captured, persona_stub = _persona_stub_factory(
        signal="downside_trend",
        confidence=73.0,
        constraints={"preferred_direction": "short"},
    )
    monkeypatch.setattr("src.agents.downside_flow_sentinel.persona_from_observations", persona_stub)

    metrics = {
        "ret_1": -0.012,
        "ret_3": -0.028,
        "ret_5": -0.052,
        "ret_10": -0.081,
        "ret_20": -0.11,
        "drawdown_20": -0.08,
        "drawdown_40": -0.13,
        "vol_ratio": 0.38,
        "downside_share10": 0.7,
        "tail_loss20": -0.03,
        "trend_slope10": -0.009,
    }
    monkeypatch.setattr("src.agents.downside_flow_sentinel._compute_metrics", lambda df: metrics)

    state = _base_state(long_shares=80)
    result = downside_flow_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["downside_flow_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "downside_trend"
    assert payload["confidence"] == 73
    assert payload["constraints"] == {"preferred_direction": "short"}

    observations = captured["observations"]
    assert observations["downside_metrics"] == metrics
    assert observations["existing_long_shares"] == 80
    assert observations["diagnostic_notes"]
    assert "suggested_constraints" not in observations


def test_insufficient_history_sets_status(monkeypatch):
    prices = _build_price_series(count=20)
    monkeypatch.setattr("src.agents.downside_flow_sentinel.get_prices", lambda *_, **__: prices)
    captured, persona_stub = _persona_stub_factory(signal="insufficient_history", confidence=40.0)
    monkeypatch.setattr("src.agents.downside_flow_sentinel.persona_from_observations", persona_stub)

    state = _base_state(long_shares=10)
    result = downside_flow_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["downside_flow_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "insufficient_history"
    assert payload["confidence"] == 40
    assert payload["constraints"] == {}

    observations = captured["observations"]
    assert observations["data_status"] == "insufficient_history"
    assert observations["downside_metrics"] == {}
    assert observations["existing_long_shares"] == 10
