from datetime import datetime, timedelta

from src.agents.range_recovery_sentinel import range_recovery_sentinel_agent
from src.agents.persona_utils import PersonaDecision
from src.data.models import Price


def _build_prices(count: int = 60, base_price: float = 250.0) -> list[Price]:
    base_dt = datetime(2024, 1, 1)
    prices: list[Price] = []
    for idx in range(count):
        price = base_price + idx * 0.2
        prices.append(
            Price(
                open=price,
                close=price,
                high=price * 1.002,
                low=price * 0.998,
                volume=1_500_000,
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
            "end_date": "2024-03-15",
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


def test_compression_break_forces_full_cover(monkeypatch):
    prices = _build_prices()
    monkeypatch.setattr(
        "src.agents.range_recovery_sentinel.get_prices",
        lambda *_, **__: prices,
    )
    captured, persona_stub = _persona_stub_factory(
        signal="compression_break",
        confidence=93.0,
        constraints={
            "preferred_direction": "long",
            "block_new_shorts": True,
            "allow_short": False,
            "target_short_shares": 0,
            "force_cover_qty": 8,
        },
    )
    monkeypatch.setattr("src.agents.range_recovery_sentinel.persona_from_observations", persona_stub)

    features = {
        "close": 280.0,
        "ret_1": 0.01,
        "ret_3": 0.03,
        "ret_5": 0.06,
        "ret_10": 0.09,
        "drawdown_20": -0.01,
        "drawdown_40": -0.03,
        "rebound_from_low": 0.08,
        "range_width_pct": 0.05,
        "range_position": 0.88,
        "vol_ratio": -0.2,
        "atr_ratio": 0.01,
        "trend_slope_8": 0.005,
        "trend_slope_12": 0.004,
    }
    monkeypatch.setattr(
        "src.agents.range_recovery_sentinel._compute_features",
        lambda _: features,
    )

    state = _base_state(short_shares=8)
    result = range_recovery_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["range_recovery_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "compression_break"
    assert payload["confidence"] == 93
    assert payload["constraints"] == {
        "preferred_direction": "long",
        "block_new_shorts": True,
        "allow_short": False,
        "target_short_shares": 0,
        "force_cover_qty": 8,
    }
    assert "score" not in payload

    observations = captured["observations"]
    assert observations["data_status"] == "data_ready"
    assert observations["existing_short_shares"] == 8
    assert observations["range_features"] == features
    assert "auto_signal_hint" not in observations


def test_drift_recovery_trims_shorts(monkeypatch):
    prices = _build_prices()
    monkeypatch.setattr(
        "src.agents.range_recovery_sentinel.get_prices",
        lambda *_, **__: prices,
    )
    captured, persona_stub = _persona_stub_factory(
        signal="drift_recovery",
        confidence=76.0,
        constraints={
            "block_new_shorts": True,
            "allow_short": False,
            "target_short_shares": 3,
        },
    )
    monkeypatch.setattr("src.agents.range_recovery_sentinel.persona_from_observations", persona_stub)

    features = {
        "close": 265.0,
        "ret_1": 0.004,
        "ret_3": 0.015,
        "ret_5": 0.028,
        "ret_10": 0.045,
        "drawdown_20": -0.04,
        "drawdown_40": -0.07,
        "rebound_from_low": 0.06,
        "range_width_pct": 0.08,
        "range_position": 0.72,
        "vol_ratio": -0.12,
        "atr_ratio": 0.009,
        "trend_slope_8": 0.0032,
        "trend_slope_12": 0.0028,
    }
    monkeypatch.setattr(
        "src.agents.range_recovery_sentinel._compute_features",
        lambda _: features,
    )

    state = _base_state(short_shares=10)
    result = range_recovery_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["range_recovery_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "drift_recovery"
    assert payload["confidence"] == 76
    assert payload["constraints"] == {
        "block_new_shorts": True,
        "allow_short": False,
        "target_short_shares": 3,
    }

    observations = captured["observations"]
    assert observations["range_features"] == features
    assert observations["existing_short_shares"] == 10
    assert observations["diagnostic_notes"]


def test_range_monitor_caps_short_bias(monkeypatch):
    prices = _build_prices()
    monkeypatch.setattr(
        "src.agents.range_recovery_sentinel.get_prices",
        lambda *_, **__: prices,
    )
    captured, persona_stub = _persona_stub_factory(
        signal="range_monitor",
        confidence=58.0,
        constraints={"block_new_shorts": True, "allow_short": False},
    )
    monkeypatch.setattr("src.agents.range_recovery_sentinel.persona_from_observations", persona_stub)

    features = {
        "close": 255.0,
        "ret_1": 0.002,
        "ret_3": 0.007,
        "ret_5": 0.015,
        "ret_10": 0.022,
        "drawdown_20": -0.06,
        "drawdown_40": -0.08,
        "rebound_from_low": 0.04,
        "range_width_pct": 0.09,
        "range_position": 0.66,
        "vol_ratio": 0.05,
        "atr_ratio": 0.008,
        "trend_slope_8": 0.0016,
        "trend_slope_12": 0.0013,
    }
    monkeypatch.setattr(
        "src.agents.range_recovery_sentinel._compute_features",
        lambda _: features,
    )

    state = _base_state(short_shares=5)
    result = range_recovery_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["range_recovery_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "range_monitor"
    assert payload["confidence"] == 58
    assert payload["constraints"] == {"block_new_shorts": True, "allow_short": False}
    assert "score" not in payload

    observations = captured["observations"]
    assert observations["range_features"] == features
    assert observations["existing_short_shares"] == 5
    assert observations["data_status"] == "data_ready"
