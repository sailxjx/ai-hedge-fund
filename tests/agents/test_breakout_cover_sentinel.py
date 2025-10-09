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


def _persona_passthrough(**kwargs):
    default = kwargs.get("default_decision", {})
    constraints = default.get("constraints")
    if isinstance(constraints, dict):
        fallback_constraints = constraints
    else:
        fallback_constraints = {}
    return PersonaDecision(
        signal=str(default.get("signal", "neutral")),
        confidence=float(default.get("confidence", 0)),
        reasoning=str(default.get("reasoning", "")),
        constraints=fallback_constraints,
    )


def test_breakout_cover_forces_full_cover(monkeypatch):
    prices = _build_prices()
    monkeypatch.setattr(
        "src.agents.breakout_cover_sentinel.get_prices",
        lambda *_, **__: prices,
    )
    monkeypatch.setattr(
        "src.agents.breakout_cover_sentinel.invoke_persona",
        _persona_passthrough
    )

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
    monkeypatch.setattr(
        "src.agents.breakout_cover_sentinel._compute_features", lambda _: features
    )

    state = _base_state(short_shares=30)
    result = breakout_cover_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["breakout_cover_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "breakout_cover"
    constraints = payload["constraints"]
    assert constraints["block_new_shorts"] is True
    assert constraints["allow_short"] is False
    assert constraints["target_short_shares"] == 0
    assert constraints["max_short_exposure_pct"] == 0.0
    assert constraints["force_cover_qty"] == 30
    assert constraints["preferred_direction"] == "long"


def test_uptrend_defensive_trims_existing_shorts(monkeypatch):
    prices = _build_prices()
    monkeypatch.setattr(
        "src.agents.breakout_cover_sentinel.get_prices",
        lambda *_, **__: prices,
    )
    monkeypatch.setattr(
        "src.agents.breakout_cover_sentinel.invoke_persona",
        _persona_passthrough
    )

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
    monkeypatch.setattr(
        "src.agents.breakout_cover_sentinel._compute_features", lambda _: features
    )

    state = _base_state(short_shares=10)
    result = breakout_cover_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["breakout_cover_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "uptrend_defensive"
    constraints = payload["constraints"]
    assert constraints["block_new_shorts"] is True
    assert constraints["max_short_exposure_pct"] == 0.02
    assert constraints["target_short_shares"] == 3
    assert constraints["force_cover_qty"] == 7
    assert constraints["preferred_direction"] == "long"


def test_balanced_mode_sets_soft_cap(monkeypatch):
    prices = _build_prices()
    monkeypatch.setattr(
        "src.agents.breakout_cover_sentinel.get_prices",
        lambda *_, **__: prices,
    )
    monkeypatch.setattr(
        "src.agents.breakout_cover_sentinel.invoke_persona",
        _persona_passthrough
    )

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
    monkeypatch.setattr(
        "src.agents.breakout_cover_sentinel._compute_features", lambda _: features
    )

    state = _base_state(short_shares=0)
    result = breakout_cover_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["breakout_cover_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "balanced"
    constraints = payload["constraints"]
    assert constraints["max_short_exposure_pct"] == 0.06
    assert constraints.get("block_new_shorts") in (None, False)
    assert "force_cover_qty" not in constraints

