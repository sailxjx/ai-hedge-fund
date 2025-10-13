from datetime import datetime, timedelta

from src.agents.macro_volatility_sentinel import macro_volatility_sentinel_agent
from src.agents.persona_utils import PersonaDecision
from src.data.models import Price


def _build_price_series(count: int = 60, base_price: float = 250.0) -> list[Price]:
    start = datetime(2024, 1, 1)
    prices: list[Price] = []
    for offset in range(count):
        level = base_price + offset * 0.5
        stamp = (start + timedelta(days=offset)).strftime("%Y-%m-%dT00:00:00Z")
        prices.append(
            Price(
                open=level,
                close=level,
                high=level * 1.01,
                low=level * 0.99,
                volume=1_000_000,
                time=stamp,
            )
        )
    return prices


def _base_state(long_shares: int = 0) -> dict:
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
                        "short": 0,
                        "long_cost_basis": 0.0,
                        "short_cost_basis": 0.0,
                    }
                },
            },
            "analyst_signals": {},
        },
        "metadata": {"show_reasoning": False},
    }


def _persona_stub_factory(signal: str, confidence: float = 75.0, constraints: dict | None = None):
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


def test_crash_alert_blocks_long_adds(monkeypatch):
    prices = _build_price_series()
    monkeypatch.setattr("src.agents.macro_volatility_sentinel.get_prices", lambda *_, **__: prices)
    captured, persona_stub = _persona_stub_factory(
        signal="crash_alert",
        confidence=88.0,
        constraints={"preferred_direction": "short", "max_long_exposure_pct": 0.05},
    )
    monkeypatch.setattr("src.agents.macro_volatility_sentinel.persona_from_observations", persona_stub)

    features = {
        "close": 310.0,
        "ret_1": -0.035,
        "ret_5": -0.085,
        "ret_10": -0.12,
        "vol_5": 0.62,
        "vol_20": 0.32,
        "vol_60": 0.28,
        "vol_ratio": 0.94,
        "vol_trend": 0.30,
        "vol_accel": 0.34,
        "drawdown_20": -0.15,
        "atr_ratio": 0.42,
        "downside_tail": -0.07,
    }
    monkeypatch.setattr("src.agents.macro_volatility_sentinel._compute_features", lambda _: features)

    state = _base_state(long_shares=100)
    result = macro_volatility_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["macro_volatility_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "crash_alert"
    assert payload["confidence"] == 88
    assert payload["constraints"] == {"preferred_direction": "short", "max_long_exposure_pct": 0.05}
    assert "score" not in payload

    observations = captured["observations"]
    assert observations["data_status"] == "data_ready"
    assert "auto_signal_hint" not in observations
    assert observations["volatility_features"] == features
    assert observations["diagnostic_notes"]


def test_vol_watch_limits_adds(monkeypatch):
    prices = _build_price_series()
    monkeypatch.setattr("src.agents.macro_volatility_sentinel.get_prices", lambda *_, **__: prices)
    captured, persona_stub = _persona_stub_factory(
        signal="vol_watch",
        confidence=67.0,
        constraints={"max_long_exposure_pct": 0.12},
    )
    monkeypatch.setattr("src.agents.macro_volatility_sentinel.persona_from_observations", persona_stub)

    features = {
        "close": 280.0,
        "ret_1": -0.012,
        "ret_5": -0.032,
        "ret_10": -0.045,
        "vol_5": 0.38,
        "vol_20": 0.25,
        "vol_60": 0.24,
        "vol_ratio": 0.52,
        "vol_trend": 0.13,
        "vol_accel": 0.14,
        "drawdown_20": -0.06,
        "atr_ratio": 0.22,
        "downside_tail": -0.04,
    }
    monkeypatch.setattr("src.agents.macro_volatility_sentinel._compute_features", lambda _: features)

    state = _base_state(long_shares=80)
    result = macro_volatility_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["macro_volatility_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "vol_watch"
    assert payload["confidence"] == 67
    assert payload["constraints"] == {"max_long_exposure_pct": 0.12}
    assert "score" not in payload

    observations = captured["observations"]
    assert observations["data_status"] == "data_ready"
    assert observations["volatility_features"] == features
    assert "auto_confidence_hint" not in observations


def test_calm_mode_sets_baseline_cap(monkeypatch):
    prices = _build_price_series()
    monkeypatch.setattr("src.agents.macro_volatility_sentinel.get_prices", lambda *_, **__: prices)
    captured, persona_stub = _persona_stub_factory(signal="calm", confidence=42.0)
    monkeypatch.setattr("src.agents.macro_volatility_sentinel.persona_from_observations", persona_stub)

    features = {
        "close": 260.0,
        "ret_1": 0.004,
        "ret_5": 0.01,
        "ret_10": 0.02,
        "vol_5": 0.14,
        "vol_20": 0.16,
        "vol_60": 0.18,
        "vol_ratio": -0.12,
        "vol_trend": -0.02,
        "vol_accel": -0.04,
        "drawdown_20": -0.01,
        "atr_ratio": -0.05,
        "downside_tail": -0.01,
    }
    monkeypatch.setattr("src.agents.macro_volatility_sentinel._compute_features", lambda _: features)

    state = _base_state(long_shares=50)
    result = macro_volatility_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["macro_volatility_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "calm"
    assert payload["confidence"] == 42
    assert payload["constraints"] == {}
    assert "score" not in payload

    observations = captured["observations"]
    assert observations["data_status"] == "data_ready"
    assert observations["volatility_features"] == features
    assert observations["diagnostic_notes"]
