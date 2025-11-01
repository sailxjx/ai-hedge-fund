from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable

import pytest

from src.agents.crash_short_allocator import crash_short_allocator_agent
from src.agents.persona_utils import PersonaDecision
from src.data.models import Price


def _build_price_series(count: int = 150, base: float = 320.0, step: float = -0.8) -> list[Price]:
    start = datetime(2024, 10, 1)
    series: list[Price] = []
    for idx in range(count):
        level = base + step * idx
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


def _persona_stub_factory(
    *,
    signal: str,
    confidence: float,
    constraints: dict | Callable[[dict], dict] | None = None,
):
    captured: dict[str, dict] = {}

    def _stub(*, observations=None, **_):
        obs = observations or {}
        captured["observations"] = obs
        if callable(constraints):
            resolved = constraints(obs)
        else:
            resolved = constraints or {}
        return PersonaDecision(
            signal=signal,
            confidence=confidence,
            reasoning=f"Stubbed {signal} decision",
            constraints=resolved,
        )

    return captured, _stub


def _build_rebound_series() -> list[Price]:
    """Construct a price path that tilts higher into the evaluation window."""

    series = _build_price_series(count=150, base=260.0, step=-0.4)
    # Inject a late positive reversal over the final 15 sessions.
    for idx in range(15):
        series[-(idx + 1)] = Price(
            open=300.0 + 3.0 * idx,
            close=300.0 + 3.0 * idx,
            high=(300.0 + 3.0 * idx) * 1.01,
            low=(300.0 + 3.0 * idx) * 0.99,
            volume=1_000_000,
            time=series[-(idx + 1)].time,
        )
    return series


def _base_state() -> dict:
    return {
        "messages": [],
        "data": {
            "tickers": ["TSLA"],
            "start_date": "2025-02-07",
            "end_date": "2025-03-10",
            "portfolio": {
                "cash": 120_000.0,
                "total_value": 120_000.0,
                "positions": {"TSLA": {"long": 0, "short": 0}},
            },
            "analyst_signals": {},
        },
        "metadata": {"show_reasoning": False},
    }


def test_allocator_targets_shorts_when_crash_conviction_high(monkeypatch):
    prices = _build_price_series()
    monkeypatch.setattr("src.agents.crash_short_allocator.get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(
        "src.agents.crash_short_allocator.get_api_key_from_state",
        lambda *_: None,
    )

    captured, persona_stub = _persona_stub_factory(
        signal="crash_short",
        confidence=92.0,
        constraints=lambda obs: {
            "preferred_direction": "short",
            "allow_short": True,
            "reference_price": obs["allocation_model"]["price_hint"],
            "allocation_pct": obs["allocation_model"]["effective_allocation_pct"] or 0.25,
            "max_short_exposure_pct": obs["allocation_model"]["effective_allocation_pct"] or 0.25,
            "target_short_shares": obs["allocation_model"]["target_short_shares_estimate"] or 1,
            "raw_target_short_shares": obs["allocation_model"]["raw_target_short_shares_estimate"],
        },
    )
    monkeypatch.setattr("src.agents.crash_short_allocator.persona_from_observations", persona_stub)

    state = _base_state()
    analyst_signals = state["data"]["analyst_signals"]
    analyst_signals["regime_meta_agent"] = {
        "TSLA": {
            "indicators": {"probabilities": {"rally": 0.22, "crash": 0.68}, "close": 240.0},
        }
    }
    analyst_signals["macro_volatility_sentinel_agent"] = {
        "TSLA": {
            "signal": "crash_alert",
            "score": 0.92,
            "metrics": {"close": 240.0},
        }
    }
    analyst_signals["downside_flow_sentinel_agent"] = {
        "TSLA": {"signal": "crash_flow", "score": 0.88, "metrics": {"ret_5": -0.07}},
    }
    analyst_signals["trend_regime_agent"] = {
        "TSLA": {"signal": "bearish", "confidence": 78},
    }

    result = crash_short_allocator_agent(state)
    payload = result["data"]["analyst_signals"]["crash_short_allocator_agent"]["TSLA"]
    observations = captured["observations"]

    assert payload["signal"] == "crash_short"
    constraints = payload["constraints"]
    assert constraints["preferred_direction"] == "short"
    assert constraints["target_short_shares"] >= 1
    assert constraints["max_short_exposure_pct"] >= 0
    assert constraints["reference_price"] > 0
    assert constraints["raw_target_short_shares"] >= constraints["target_short_shares"]
    assert payload["allocation_pct"] == pytest.approx(constraints["allocation_pct"])
    assert payload["indicators"]["target_short_shares_estimate"] == observations["allocation_model"]["target_short_shares_estimate"]
    assert "auto_signal_hint" not in observations


def test_allocator_targets_with_short_history(monkeypatch):
    prices = _build_price_series(count=20)
    monkeypatch.setattr("src.agents.crash_short_allocator.get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(
        "src.agents.crash_short_allocator.get_api_key_from_state",
        lambda *_: None,
    )

    captured, persona_stub = _persona_stub_factory(
        signal="short_bias",
        confidence=81.0,
        constraints=lambda obs: {
            "preferred_direction": "short",
            "allow_short": True,
            "allocation_pct": obs["allocation_model"]["effective_allocation_pct"] or 0.18,
            "max_short_exposure_pct": obs["allocation_model"]["effective_allocation_pct"] or 0.18,
            "target_short_shares": obs["allocation_model"]["target_short_shares_estimate"] or 1,
            "reference_price": obs["allocation_model"]["price_hint"],
        },
    )
    monkeypatch.setattr("src.agents.crash_short_allocator.persona_from_observations", persona_stub)

    state = _base_state()
    analyst_signals = state["data"]["analyst_signals"]
    analyst_signals["regime_meta_agent"] = {
        "TSLA": {
            "signal": "crash",
            "indicators": {"probabilities": {"rally": 0.2, "crash": 0.66}},
        }
    }
    analyst_signals["macro_volatility_sentinel_agent"] = {"TSLA": {"signal": "crash_alert", "score": 0.85}}
    analyst_signals["downside_flow_sentinel_agent"] = {"TSLA": {"signal": "crash_flow", "score": 0.8}}
    analyst_signals["trend_regime_agent"] = {"TSLA": {"signal": "bearish", "confidence": 78}}

    result = crash_short_allocator_agent(state)
    payload = result["data"]["analyst_signals"]["crash_short_allocator_agent"]["TSLA"]

    assert payload["signal"] == "short_bias"
    assert payload["allocation_pct"] > 0
    constraints = payload["constraints"]
    assert constraints.get("target_short_shares", 0) > 0
    assert constraints.get("reference_price", 0) > 0
    observations = captured["observations"]
    assert observations["allocation_model"]["raw_allocation_pct"] >= 0
    assert "auto_signal_hint" not in observations


def test_allocator_accelerates_when_crash_prob_rises(monkeypatch):
    prices = _build_price_series(count=140, base=120.0, step=-0.05)
    monkeypatch.setattr("src.agents.crash_short_allocator.get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(
        "src.agents.crash_short_allocator.get_api_key_from_state",
        lambda *_: None,
    )

    captured, persona_stub = _persona_stub_factory(
        signal="short_bias",
        confidence=77.0,
        constraints=lambda obs: {
            "preferred_direction": "short",
            "allow_short": True,
            "allocation_pct": obs["allocation_model"]["effective_allocation_pct"] or 0.12,
            "max_short_exposure_pct": obs["allocation_model"]["effective_allocation_pct"] or 0.12,
            "target_short_shares": obs["allocation_model"]["target_short_shares_estimate"] or 1,
            "reference_price": obs["allocation_model"]["price_hint"],
        },
    )
    monkeypatch.setattr("src.agents.crash_short_allocator.persona_from_observations", persona_stub)

    state = _base_state()
    analyst_signals = state["data"]["analyst_signals"]
    analyst_signals["regime_meta_agent"] = {
        "TSLA": {
            "signal": "crash",
            "indicators": {"probabilities": {"rally": 0.46, "crash": 0.49}, "close": 120.0},
        }
    }
    analyst_signals["macro_volatility_sentinel_agent"] = {"TSLA": {"signal": "calm", "score": 0.2}}
    analyst_signals["downside_flow_sentinel_agent"] = {"TSLA": {"signal": "stable", "score": 0.18}}
    analyst_signals["trend_regime_agent"] = {"TSLA": {"signal": "bullish", "confidence": 63}}

    result = crash_short_allocator_agent(state)
    payload = result["data"]["analyst_signals"]["crash_short_allocator_agent"]["TSLA"]

    assert payload["allocation_pct"] > 0.0
    constraints = payload["constraints"]
    assert constraints.get("target_short_shares", 0) > 0
    assert constraints.get("max_short_exposure_pct", 0) > 0
    observations = captured["observations"]
    assert observations["allocation_model"]["effective_allocation_pct"] is not None


def test_allocator_monitor_when_signals_muted(monkeypatch):
    prices = _build_price_series(base=310.0, step=0.1)
    monkeypatch.setattr("src.agents.crash_short_allocator.get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(
        "src.agents.crash_short_allocator.get_api_key_from_state",
        lambda *_: None,
    )

    captured, persona_stub = _persona_stub_factory(
        signal="monitor",
        confidence=60.0,
        constraints=lambda obs: {"reference_price": obs["allocation_model"].get("price_hint")},
    )
    monkeypatch.setattr("src.agents.crash_short_allocator.persona_from_observations", persona_stub)

    state = _base_state()
    analyst_signals = state["data"]["analyst_signals"]
    analyst_signals["regime_meta_agent"] = {"TSLA": {"indicators": {"probabilities": {"rally": 0.6, "crash": 0.18}}, "signal": "rally"}}
    analyst_signals["macro_volatility_sentinel_agent"] = {"TSLA": {"signal": "calm", "score": 0.12}}
    analyst_signals["downside_flow_sentinel_agent"] = {"TSLA": {"signal": "stable", "score": 0.1}}

    result = crash_short_allocator_agent(state)
    payload = result["data"]["analyst_signals"]["crash_short_allocator_agent"]["TSLA"]

    assert payload["signal"] == "monitor"
    assert set(payload["constraints"].keys()) <= {"reference_price"}
    observations = captured["observations"]
    assert observations["release_analysis"]["should_release"] is False
    assert "auto_signal_hint" not in observations


def test_allocator_releases_existing_short_on_rebound(monkeypatch):
    prices = _build_rebound_series()
    monkeypatch.setattr("src.agents.crash_short_allocator.get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(
        "src.agents.crash_short_allocator.get_api_key_from_state",
        lambda *_: None,
    )

    captured, persona_stub = _persona_stub_factory(
        signal="cover_short",
        confidence=85.0,
        constraints=lambda obs: {
            "preferred_direction": "neutral",
            "target_short_shares": 0,
            "max_additional_short_shares": 0,
            "max_short_exposure_pct": 0.0,
            "force_cover_qty": obs["existing_positions"]["short"],
            "block_new_shorts": True,
        },
    )
    monkeypatch.setattr("src.agents.crash_short_allocator.persona_from_observations", persona_stub)

    state = _base_state()
    state["data"]["portfolio"]["positions"]["TSLA"] = {"long": 0, "short": 6}
    analyst_signals = state["data"]["analyst_signals"]
    analyst_signals["regime_meta_agent"] = {
        "TSLA": {
            "indicators": {"probabilities": {"rally": 0.4, "crash": 0.28}},
            "signal": "neutral",
        }
    }
    analyst_signals["macro_volatility_sentinel_agent"] = {"TSLA": {"signal": "calm", "score": 0.18}}
    analyst_signals["downside_flow_sentinel_agent"] = {"TSLA": {"signal": "stable", "score": 0.2}}

    result = crash_short_allocator_agent(state)
    payload = result["data"]["analyst_signals"]["crash_short_allocator_agent"]["TSLA"]

    assert payload["signal"] == "cover_short"
    constraints = payload["constraints"]
    assert constraints["target_short_shares"] == 0
    assert constraints["force_cover_qty"] == 6
    assert constraints["max_additional_short_shares"] == 0
    assert constraints["block_new_shorts"] is True
    observations = captured["observations"]
    assert observations["release_analysis"]["should_release"] is True


def test_allocator_respects_peer_short_cap(monkeypatch):
    prices = _build_price_series()
    monkeypatch.setattr("src.agents.crash_short_allocator.get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(
        "src.agents.crash_short_allocator.get_api_key_from_state",
        lambda *_: None,
    )

    captured, persona_stub = _persona_stub_factory(
        signal="crash_short",
        confidence=90.0,
        constraints=lambda obs: {
            "preferred_direction": "short",
            "allow_short": True,
            "reference_price": obs["allocation_model"]["price_hint"],
            "max_short_exposure_pct": min(0.05, (obs["allocation_model"]["effective_allocation_pct"] or 0.05)),
            "allocation_pct": min(0.05, (obs["allocation_model"]["effective_allocation_pct"] or 0.05)),
            "target_short_shares": obs["allocation_model"]["target_short_shares_estimate"] or 1,
        },
    )
    monkeypatch.setattr("src.agents.crash_short_allocator.persona_from_observations", persona_stub)

    state = _base_state()
    analyst_signals = state["data"]["analyst_signals"]
    analyst_signals["regime_meta_agent"] = {
        "TSLA": {
            "indicators": {"probabilities": {"rally": 0.18, "crash": 0.7}, "close": 240.0},
        }
    }
    analyst_signals["macro_volatility_sentinel_agent"] = {
        "TSLA": {
            "signal": "crash_alert",
            "score": 0.9,
            "metrics": {"close": 240.0},
        }
    }
    analyst_signals["downside_flow_sentinel_agent"] = {
        "TSLA": {"signal": "crash_flow", "score": 0.82, "metrics": {"ret_5": -0.07}},
    }
    analyst_signals["trend_regime_agent"] = {
        "TSLA": {
            "signal": "bearish",
            "confidence": 78,
            "constraints": {"max_short_exposure_pct": 0.05},
        }
    }

    result = crash_short_allocator_agent(state)
    payload = result["data"]["analyst_signals"]["crash_short_allocator_agent"]["TSLA"]
    constraints = payload["constraints"]
    indicators = payload["indicators"]

    assert payload["signal"] == "crash_short"
    assert constraints["max_short_exposure_pct"] <= 0.05 + 1e-9

    price_hint = constraints["reference_price"]
    assert price_hint > 0

    total_equity = state["data"]["portfolio"]["total_value"]
    peer_cap_pct = indicators.get("peer_short_cap_pct")
    if peer_cap_pct is not None:
        assert constraints["target_short_shares"] * price_hint <= total_equity * (peer_cap_pct + 1e-6)

    assert payload["allocation_pct"] <= constraints["max_short_exposure_pct"] + 1e-4
    limiter_label = payload["indicators"].get("allocation_limiter")
    assert limiter_label is not None
    observations = captured["observations"]
    assert observations["allocation_model"]["allocation_candidates"]
