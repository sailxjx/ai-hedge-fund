from datetime import datetime, timedelta

from src.agents.breakout_cover_sentinel import breakout_cover_sentinel_agent
from src.agents.persona_utils import PersonaDecision
from src.data.models import Price


def _build_prices(count: int = 40, base_price: float = 250.0) -> list[Price]:
    base_dt = datetime(2024, 1, 1)
    prices: list[Price] = []
    for idx in range(count):
        price = base_price + idx * 0.5
        prices.append(
            Price(
                open=price,
                close=price,
                high=price * 1.003,
                low=price * 0.997,
                volume=2_000_000,
                time=(base_dt + timedelta(days=idx)).strftime("%Y-%m-%dT00:00:00Z"),
            )
        )
    return prices


def _base_state(short_shares: int = 0):
    return {
        "messages": [],
        "data": {
            "tickers": ["TSLA"],
            "portfolio": {
                "cash": 100_000.0,
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
                "positions": {
                    "TSLA": {
                        "long": 0,
                        "short": short_shares,
                        "long_cost_basis": 0.0,
                        "short_cost_basis": 0.0,
                    }
                },
            },
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
            "analyst_signals": {},
        },
        "metadata": {"show_reasoning": False},
    }


def _persona_stub_factory(signal: str, confidence: float, constraints: dict | None = None):
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


def test_breakout_cover_forces_full_cover(monkeypatch):
    prices = _build_prices()
    monkeypatch.setattr(
        "src.agents.breakout_cover_sentinel.get_prices",
        lambda *_, **__: prices,
    )
    captured, persona_stub = _persona_stub_factory(
        signal="breakout_cover",
        confidence=92.0,
        constraints={
            "preferred_direction": "long",
            "block_new_shorts": True,
            "allow_short": False,
            "target_short_shares": 0,
        },
    )
    monkeypatch.setattr("src.agents.breakout_cover_sentinel.persona_from_observations", persona_stub)

    features = {
        "close": 280.0,
        "ret_3": 0.05,
        "ret_5": 0.08,
        "ret_10": 0.12,
        "ema21_gap": 0.025,
        "ema_slope": 0.06,
        "breakout_gap": 0.03,
        "volume_ratio": 1.2,
        "atr_ratio": 0.01,
    }
    monkeypatch.setattr("src.agents.breakout_cover_sentinel._compute_features", lambda _: features)

    state = _base_state(short_shares=30)
    result = breakout_cover_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["breakout_cover_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "breakout_cover"
    assert payload["confidence"] == 92
    assert payload["constraints"] == {
        "preferred_direction": "long",
        "block_new_shorts": True,
        "allow_short": False,
        "target_short_shares": 0,
    }
    assert "score" not in payload

    observations = captured["observations"]
    assert observations["data_status"] == "data_ready"
    assert observations["current_short_shares"] == 30
    assert observations["breakout_features"] == features
    assert "auto_signal_hint" not in observations


def test_uptrend_defensive_trims_existing_shorts(monkeypatch):
    prices = _build_prices()
    monkeypatch.setattr(
        "src.agents.breakout_cover_sentinel.get_prices",
        lambda *_, **__: prices,
    )
    captured, persona_stub = _persona_stub_factory(
        signal="uptrend_defensive",
        confidence=71.0,
        constraints={"block_new_shorts": True, "max_short_exposure_pct": 0.02},
    )
    monkeypatch.setattr("src.agents.breakout_cover_sentinel.persona_from_observations", persona_stub)

    features = {
        "close": 265.0,
        "ret_3": 0.028,
        "ret_5": 0.045,
        "ret_10": 0.07,
        "ema21_gap": 0.015,
        "ema_slope": 0.035,
        "breakout_gap": 0.016,
        "volume_ratio": 0.6,
        "atr_ratio": 0.01,
    }
    monkeypatch.setattr("src.agents.breakout_cover_sentinel._compute_features", lambda _: features)

    state = _base_state(short_shares=10)
    result = breakout_cover_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["breakout_cover_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "uptrend_defensive"
    assert payload["confidence"] == 71
    assert payload["constraints"] == {"block_new_shorts": True, "max_short_exposure_pct": 0.02}

    observations = captured["observations"]
    assert observations["data_status"] == "data_ready"
    assert observations["breakout_features"] == features
    assert observations["current_short_shares"] == 10
    assert observations["diagnostic_notes"]
    assert "auto_confidence_hint" not in observations


def test_balanced_mode_sets_soft_cap(monkeypatch):
    prices = _build_prices()
    monkeypatch.setattr(
        "src.agents.breakout_cover_sentinel.get_prices",
        lambda *_, **__: prices,
    )
    captured, persona_stub = _persona_stub_factory(signal="balanced", confidence=48.0)
    monkeypatch.setattr("src.agents.breakout_cover_sentinel.persona_from_observations", persona_stub)

    features = {
        "close": 252.0,
        "ret_3": 0.005,
        "ret_5": 0.01,
        "ret_10": 0.015,
        "ema21_gap": 0.002,
        "ema_slope": 0.004,
        "breakout_gap": 0.003,
        "volume_ratio": -0.1,
        "atr_ratio": 0.01,
    }
    monkeypatch.setattr("src.agents.breakout_cover_sentinel._compute_features", lambda _: features)

    state = _base_state(short_shares=0)
    result = breakout_cover_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["breakout_cover_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "balanced"
    assert payload["confidence"] == 48
    assert payload["constraints"] == {}
    assert "score" not in payload

    observations = captured["observations"]
    assert observations["breakout_features"] == features
    assert observations["current_short_shares"] == 0
    assert observations["data_status"] == "data_ready"
