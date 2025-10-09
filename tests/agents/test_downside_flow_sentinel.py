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


def _persona_passthrough(**kwargs):
    default = kwargs.get("default_decision", {})
    constraints = default.get("constraints")
    fallback_constraints = constraints if isinstance(constraints, dict) else {}
    return PersonaDecision(
        signal=str(default.get("signal", "neutral")),
        confidence=float(default.get("confidence", 0)),
        reasoning=str(default.get("reasoning", "")),
        constraints=fallback_constraints,
    )


def test_crash_flow_freezes_long_exposure(monkeypatch):
    prices = _build_price_series()
    monkeypatch.setattr(
        "src.agents.downside_flow_sentinel.get_prices", lambda *_, **__: prices
    )
    monkeypatch.setattr(
        "src.agents.downside_flow_sentinel.invoke_persona",
        _persona_passthrough
    )

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
    monkeypatch.setattr(
        "src.agents.downside_flow_sentinel._compute_metrics", lambda df: metrics
    )

    state = _base_state(long_shares=120)
    result = downside_flow_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["downside_flow_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "crash_flow"
    constraints = payload["constraints"]
    assert constraints.get("preferred_direction") == "short"
    assert constraints.get("max_long_exposure_pct") <= 0.05
    assert constraints.get("max_additional_long_shares") == 0
    assert constraints.get("max_long_shares") < 120


def test_downside_trend_throttles_long_adds(monkeypatch):
    prices = _build_price_series(step=-0.5)
    monkeypatch.setattr(
        "src.agents.downside_flow_sentinel.get_prices", lambda *_, **__: prices
    )
    monkeypatch.setattr(
        "src.agents.downside_flow_sentinel.invoke_persona",
        _persona_passthrough
    )

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
    monkeypatch.setattr(
        "src.agents.downside_flow_sentinel._compute_metrics", lambda df: metrics
    )

    state = _base_state(long_shares=80)
    result = downside_flow_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["downside_flow_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "downside_trend"
    constraints = payload["constraints"]
    assert constraints.get("preferred_direction") == "short"
    assert 0 < constraints.get("max_additional_long_shares", -1) <= 16
    assert constraints.get("max_long_shares") <= 80


def test_downside_trend_with_no_existing_long_drops_share_caps(monkeypatch):
    prices = _build_price_series(step=-0.5)
    monkeypatch.setattr(
        "src.agents.downside_flow_sentinel.get_prices", lambda *_, **__: prices
    )
    monkeypatch.setattr(
        "src.agents.downside_flow_sentinel.invoke_persona",
        _persona_passthrough
    )

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
    monkeypatch.setattr(
        "src.agents.downside_flow_sentinel._compute_metrics", lambda df: metrics
    )

    state = _base_state(long_shares=0)
    result = downside_flow_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["downside_flow_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "downside_trend"
    constraints = payload["constraints"]
    assert constraints.get("preferred_direction") == "short"
    assert constraints.get("max_long_exposure_pct") == 0.08
    assert "max_additional_long_shares" not in constraints
    assert "max_long_shares" not in constraints


def test_stable_regime_retains_long_flex(monkeypatch):
    prices = _build_price_series(step=0.2)
    monkeypatch.setattr(
        "src.agents.downside_flow_sentinel.get_prices", lambda *_, **__: prices
    )
    monkeypatch.setattr(
        "src.agents.downside_flow_sentinel.invoke_persona",
        _persona_passthrough
    )

    metrics = {
        "ret_1": 0.004,
        "ret_3": 0.009,
        "ret_5": -0.012,
        "ret_10": -0.018,
        "ret_20": -0.021,
        "drawdown_20": -0.03,
        "drawdown_40": -0.04,
        "vol_ratio": -0.08,
        "downside_share10": 0.3,
        "tail_loss20": -0.005,
        "trend_slope10": 0.002,
    }
    monkeypatch.setattr(
        "src.agents.downside_flow_sentinel._compute_metrics", lambda df: metrics
    )

    state = _base_state(long_shares=50)
    result = downside_flow_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["downside_flow_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "stable"
    constraints = payload["constraints"]
    assert constraints.get("preferred_direction") is None
    assert constraints.get("max_long_exposure_pct") == 0.16
    assert constraints.get("max_additional_long_shares") >= 20
    assert "max_long_shares" not in constraints


def test_stable_regime_without_long_position_keeps_constraints_open(monkeypatch):
    prices = _build_price_series(step=0.2)
    monkeypatch.setattr(
        "src.agents.downside_flow_sentinel.get_prices", lambda *_, **__: prices
    )
    monkeypatch.setattr(
        "src.agents.downside_flow_sentinel.invoke_persona",
        _persona_passthrough
    )

    metrics = {
        "ret_1": 0.004,
        "ret_3": 0.009,
        "ret_5": -0.012,
        "ret_10": -0.018,
        "ret_20": -0.021,
        "drawdown_20": -0.03,
        "drawdown_40": -0.04,
        "vol_ratio": -0.08,
        "downside_share10": 0.3,
        "tail_loss20": -0.005,
        "trend_slope10": 0.002,
    }
    monkeypatch.setattr(
        "src.agents.downside_flow_sentinel._compute_metrics", lambda df: metrics
    )

    state = _base_state(long_shares=0)
    result = downside_flow_sentinel_agent(state)
    payload = result["data"]["analyst_signals"]["downside_flow_sentinel_agent"]["TSLA"]

    assert payload["signal"] == "stable"
    constraints = payload["constraints"]
    assert constraints.get("preferred_direction") is None
    assert constraints.get("max_long_exposure_pct") == 0.16
    assert "max_additional_long_shares" not in constraints
    assert "max_long_shares" not in constraints
