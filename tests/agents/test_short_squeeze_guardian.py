from datetime import datetime, timedelta

from src.agents.short_squeeze_guardian import short_squeeze_guardian_agent
from src.agents.persona_utils import PersonaDecision
from src.data.models import Price


def _build_flat_prices(count: int = 30, close: float = 300.0) -> list[Price]:
    base_dt = datetime(2024, 1, 1)
    prices: list[Price] = []
    for idx in range(count):
        price = float(close)
        prices.append(
            Price(
                open=price,
                close=price,
                high=price * 1.002,
                low=price * 0.998,
                volume=1_000_000,
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


def test_squeeze_warning_blocks_and_covers(monkeypatch):
    prices = _build_flat_prices()
    monkeypatch.setattr("src.agents.short_squeeze_guardian.get_prices", lambda *_, **__: prices)
    captured, persona_stub = _persona_stub_factory(
        signal="squeeze_warning",
        confidence=91.0,
        constraints={
            "block_new_shorts": True,
            "allow_short": False,
            "target_short_shares": 0,
            "force_cover_qty": 40,
        },
    )
    monkeypatch.setattr("src.agents.short_squeeze_guardian.persona_from_observations", persona_stub)

    features = {
        "close": 420.0,
        "ret_5": 0.12,
        "ret_10": 0.18,
        "ret_20": 0.10,
        "velocity_z": 3.0,
        "acceleration": 0.08,
        "gap_vs_high": 0.025,
        "gap_vs_ma": 0.12,
        "volume_ratio": 1.5,
        "range_ratio": 0.025,
        "std_5": 0.02,
        "std_20": 0.015,
    }
    monkeypatch.setattr("src.agents.short_squeeze_guardian._compute_features", lambda _: features)

    state = _base_state(short_shares=40)
    result = short_squeeze_guardian_agent(state)
    payload = result["data"]["analyst_signals"]["short_squeeze_guardian_agent"]["TSLA"]

    assert payload["signal"] == "squeeze_warning"
    assert payload["confidence"] == 91
    assert payload["constraints"] == {
        "block_new_shorts": True,
        "allow_short": False,
        "target_short_shares": 0,
        "force_cover_qty": 40,
    }
    assert "score" not in payload

    observations = captured["observations"]
    assert observations["data_status"] == "data_ready"
    assert observations["squeeze_features"] == features
    assert observations["current_short_shares"] == 40
    assert "auto_signal_hint" not in observations


def test_elevated_risk_trims_existing_short(monkeypatch):
    prices = _build_flat_prices()
    monkeypatch.setattr("src.agents.short_squeeze_guardian.get_prices", lambda *_, **__: prices)
    captured, persona_stub = _persona_stub_factory(
        signal="elevated_risk",
        confidence=72.0,
        constraints={"block_new_shorts": True, "allow_short": False, "target_short_shares": 4},
    )
    monkeypatch.setattr("src.agents.short_squeeze_guardian.persona_from_observations", persona_stub)

    features = {
        "close": 360.0,
        "ret_5": 0.045,
        "ret_10": 0.06,
        "ret_20": 0.02,
        "velocity_z": 1.0,
        "acceleration": 0.04,
        "gap_vs_high": 0.015,
        "gap_vs_ma": 0.04,
        "volume_ratio": 0.6,
        "range_ratio": 0.015,
        "std_5": 0.015,
        "std_20": 0.012,
    }
    monkeypatch.setattr("src.agents.short_squeeze_guardian._compute_features", lambda _: features)

    state = _base_state(short_shares=10)
    result = short_squeeze_guardian_agent(state)
    payload = result["data"]["analyst_signals"]["short_squeeze_guardian_agent"]["TSLA"]

    assert payload["signal"] == "elevated_risk"
    assert payload["confidence"] == 72
    assert payload["constraints"] == {"block_new_shorts": True, "allow_short": False, "target_short_shares": 4}

    observations = captured["observations"]
    assert observations["data_status"] == "data_ready"
    assert observations["squeeze_features"] == features
    assert observations["current_short_shares"] == 10
    assert observations["diagnostic_notes"]


def test_calm_mode_sets_soft_cap(monkeypatch):
    prices = _build_flat_prices()
    monkeypatch.setattr("src.agents.short_squeeze_guardian.get_prices", lambda *_, **__: prices)
    captured, persona_stub = _persona_stub_factory(signal="calm", confidence=48.0, constraints={})
    monkeypatch.setattr("src.agents.short_squeeze_guardian.persona_from_observations", persona_stub)

    features = {
        "close": 310.0,
        "ret_5": 0.005,
        "ret_10": 0.01,
        "ret_20": 0.008,
        "velocity_z": 0.1,
        "acceleration": 0.001,
        "gap_vs_high": -0.02,
        "gap_vs_ma": 0.01,
        "volume_ratio": -0.2,
        "range_ratio": 0.004,
        "std_5": 0.01,
        "std_20": 0.009,
    }
    monkeypatch.setattr("src.agents.short_squeeze_guardian._compute_features", lambda _: features)

    state = _base_state(short_shares=0)
    result = short_squeeze_guardian_agent(state)
    payload = result["data"]["analyst_signals"]["short_squeeze_guardian_agent"]["TSLA"]

    assert payload["signal"] == "calm"
    assert payload["confidence"] == 48
    assert payload["constraints"] == {}

    observations = captured["observations"]
    assert observations["data_status"] == "data_ready"
    assert observations["current_short_shares"] == 0
    assert observations["squeeze_features"] == features
