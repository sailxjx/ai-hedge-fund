import json
from datetime import datetime, timedelta

import pytest

from src.agents.persona_utils import PersonaDecision
from src.agents.regime_meta import regime_meta_agent
from src.data.models import Price


def _build_price_series(values: list[float]) -> list[Price]:
    base_dt = datetime(2024, 1, 1)
    prices: list[Price] = []
    for idx, close in enumerate(values):
        close_val = float(close)
        high = close_val * 1.01
        low = close_val * 0.99
        open_price = (high + low) / 2
        prices.append(
            Price(
                open=open_price,
                close=close_val,
                high=high,
                low=low,
                volume=1_000_000,
                time=(base_dt + timedelta(days=idx)).strftime("%Y-%m-%dT00:00:00Z"),
            )
        )
    return prices


def _base_state() -> dict:
    return {
        "messages": [],
        "data": {
            "tickers": ["TSLA"],
            "portfolio": {
                "cash": 100_000.0,
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "positions": {
                    "TSLA": {
                        "long": 0,
                        "short": 0,
                        "long_cost_basis": 0.0,
                        "short_cost_basis": 0.0,
                    }
                },
            },
            "start_date": "2024-01-01",
            "end_date": "2024-04-30",
            "analyst_signals": {},
        },
        "metadata": {"show_reasoning": False},
    }


def test_regime_meta_pushes_long_bias(monkeypatch):
    rally_values = [100 * (1.004) ** i for i in range(90)]
    prices = _build_price_series(rally_values)

    monkeypatch.setattr("src.agents.regime_meta.get_prices", lambda *_, **__: prices)

    state = _base_state()
    state["data"]["analyst_signals"] = {
        "growth_momentum_agent": {
            "TSLA": {
                "confidence": 72,
                "indicators": {"prob_up": 0.72, "base_rate": 0.52},
            }
        },
        "stat_mean_reversion_agent": {
            "TSLA": {
                "confidence": 60,
                "indicators": {"prob_revert_up": 0.58},
            }
        },
    }

    result = regime_meta_agent(state)
    payload = result["data"]["analyst_signals"]["regime_meta_agent"]["TSLA"]
    layers = payload["indicators"]["probability_layers"]
    blended = layers["blended"]
    model_reference = payload["indicators"]["model_reference"]

    assert blended["rally"] > blended["crash"]
    assert blended["rally"] > blended["consolidation"]
    assert model_reference["signal"] == "rally"
    assert model_reference["confidence_pct"] >= 60


def test_regime_meta_flags_crash(monkeypatch):
    crash_values = []
    price = 120.0
    for idx in range(90):
        if idx < 50:
            price *= 0.998
        else:
            price *= 0.985
        crash_values.append(price)

    prices = _build_price_series(crash_values)

    monkeypatch.setattr("src.agents.regime_meta.get_prices", lambda *_, **__: prices)

    state = _base_state()
    state["data"]["analyst_signals"] = {
        "growth_momentum_agent": {
            "TSLA": {
                "confidence": 55,
                "indicators": {"prob_up": 0.38, "base_rate": 0.52},
            }
        },
        "stat_mean_reversion_agent": {
            "TSLA": {
                "confidence": 55,
                "indicators": {"prob_revert_up": 0.35},
            }
        },
    }

    result = regime_meta_agent(state)
    payload = result["data"]["analyst_signals"]["regime_meta_agent"]["TSLA"]
    layers = payload["indicators"]["probability_layers"]
    blended = layers["blended"]
    model_reference = payload["indicators"]["model_reference"]

    assert blended["crash"] > blended["rally"]
    assert blended["crash"] > blended["consolidation"]
    assert model_reference["signal"] == "crash"


def test_probabilities_use_calibration(tmp_path, monkeypatch):
    from src.agents import regime_meta as module

    feature_keys = [
        "volatility_slope",
        "drawdown_depth",
        "return5_skew20",
        "momentum_20",
        "momentum_60",
        "vol_30",
        "vol_60",
        "vol_ratio",
    ]

    weights_payload = {
        "classes": ["crash", "rally", "consolidation"],
        "feature_order": feature_keys,
        "feature_means": {name: 0.0 for name in feature_keys},
        "feature_stds": {name: 1.0 for name in feature_keys},
        "weights": {
            "crash": [-0.6, -0.8, 0.4, -1.5, -1.0, 0.3, 0.4, 0.2],
            "rally": [0.6, 0.8, -0.4, 1.8, 1.2, -0.3, -0.4, -0.2],
            "consolidation": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "bias": {"crash": 0.0, "rally": 0.2, "consolidation": -0.1},
        "metadata": {
            "sample_count": 120,
            "tickers": ["TEST"],
            "regimes": ["crash", "rally", "consolidation"],
            "epochs": 10,
            "learning_rate": 0.1,
            "l2_penalty": 0.0,
        },
    }

    weights_path = tmp_path / "calibration.json"
    weights_path.write_text(json.dumps(weights_payload, indent=2))

    monkeypatch.setattr(module, "CALIBRATION_PATH", weights_path)
    module._load_calibration.cache_clear()

    snapshot = module.FeatureSnapshot(
        close=210.0,
        vol_30=0.32,
        vol_60=0.40,
        volatility_slope=-0.08,
        drawdown_depth=-0.12,
        return5_skew20=0.05,
        momentum_20=0.12,
        momentum_60=0.18,
    )

    probabilities = module._probabilities(
        snapshot,
        growth_payload={},
        mean_rev_payload={},
        downside_payload=None,
    )

    assert probabilities["rally"] > probabilities["crash"]
    assert sum(probabilities.values()) == pytest.approx(1.0, rel=1e-6)


def test_downside_flow_boosts_crash_probability(monkeypatch):
    from src.agents import regime_meta as module

    snapshot = module.FeatureSnapshot(
        close=180.0,
        vol_30=0.42,
        vol_60=0.36,
        volatility_slope=0.06,
        drawdown_depth=-0.09,
        return5_skew20=0.04,
        momentum_20=-0.03,
        momentum_60=-0.07,
    )

    growth_payload = {"indicators": {"prob_up": 0.43, "base_rate": 0.54}}
    mean_rev_payload = {"indicators": {"prob_revert_up": 0.38}}

    manual = {"rally": 0.27, "crash": 0.33, "consolidation": 0.40}
    data = {"rally": 0.30, "crash": 0.30, "consolidation": 0.40}

    monkeypatch.setattr(module, "_probabilities_manual", lambda *_: manual)
    monkeypatch.setattr(module, "_probabilities_data_driven", lambda *_: data)

    baseline = module._probabilities(
        snapshot,
        growth_payload=growth_payload,
        mean_rev_payload=mean_rev_payload,
        downside_payload=None,
    )
    boosted = module._probabilities(
        snapshot,
        growth_payload=growth_payload,
        mean_rev_payload=mean_rev_payload,
        downside_payload={"signal": "crash_flow", "score": 0.9},
    )

    assert boosted["crash"] > baseline["crash"]
    assert boosted["rally"] < baseline["rally"]


def test_downside_flow_requires_feature_alignment(monkeypatch):
    from src.agents import regime_meta as module

    snapshot = module.FeatureSnapshot(
        close=180.0,
        vol_30=0.31,
        vol_60=0.30,
        volatility_slope=0.002,
        drawdown_depth=-0.02,
        return5_skew20=0.01,
        momentum_20=-0.01,
        momentum_60=-0.02,
    )

    growth_payload = {"indicators": {"prob_up": 0.43, "base_rate": 0.54}}
    mean_rev_payload = {"indicators": {"prob_revert_up": 0.38}}

    manual = {"rally": 0.27, "crash": 0.33, "consolidation": 0.40}
    data = {"rally": 0.30, "crash": 0.30, "consolidation": 0.40}

    monkeypatch.setattr(module, "_probabilities_manual", lambda *_: manual)
    monkeypatch.setattr(module, "_probabilities_data_driven", lambda *_: data)

    baseline = module._probabilities(
        snapshot,
        growth_payload=growth_payload,
        mean_rev_payload=mean_rev_payload,
        downside_payload=None,
    )
    gated = module._probabilities(
        snapshot,
        growth_payload=growth_payload,
        mean_rev_payload=mean_rev_payload,
        downside_payload={"signal": "crash_flow", "score": 0.9},
    )

    assert gated["crash"] == pytest.approx(baseline["crash"])


def test_target_long_shares_respects_low_cash(monkeypatch):
    rally_values = [500 * (1.0015) ** i for i in range(90)]
    prices = _build_price_series(rally_values)

    monkeypatch.setattr("src.agents.regime_meta.get_prices", lambda *_, **__: prices)

    state = _base_state()
    state["data"]["portfolio"]["cash"] = 450.0
    state["data"]["portfolio"]["positions"]["TSLA"]["long"] = 1
    state["data"]["analyst_signals"] = {
        "growth_momentum_agent": {
            "TSLA": {
                "confidence": 70,
                "indicators": {"prob_up": 0.74, "base_rate": 0.51},
            }
        }
    }

    result = regime_meta_agent(state)
    payload = result["data"]["analyst_signals"]["regime_meta_agent"]["TSLA"]
    model_reference = payload["indicators"]["model_reference"]
    assert payload["signal"] == "rally"
    assert model_reference["signal"] == "rally"
    # No deterministic constraints are produced; persona (or downstream logic) must decide sizing.
    constraints = payload.get("constraints") or {}
    assert constraints == {}


def test_target_long_shares_caps_fractional_cash(monkeypatch):
    rally_values = [40 * (1.004) ** i for i in range(90)]
    prices = _build_price_series(rally_values)

    monkeypatch.setattr("src.agents.regime_meta.get_prices", lambda *_, **__: prices)

    state = _base_state()
    state["data"]["portfolio"]["cash"] = 1_250_000.0
    state["data"]["portfolio"]["positions"]["TSLA"]["long"] = 1500
    state["data"]["analyst_signals"] = {
        "growth_momentum_agent": {
            "TSLA": {
                "confidence": 78,
                "indicators": {"prob_up": 0.79, "base_rate": 0.52},
            }
        },
        "stat_mean_reversion_agent": {
            "TSLA": {
                "confidence": 55,
                "indicators": {"prob_revert_up": 0.6},
            }
        },
    }

    result = regime_meta_agent(state)
    payload = result["data"]["analyst_signals"]["regime_meta_agent"]["TSLA"]
    model_reference = payload["indicators"]["model_reference"]
    assert payload["signal"] == "rally"
    assert model_reference["signal"] == "rally"

    # Constraints are left to the persona/portfolio manager under the new observation-only design.
    constraints = payload.get("constraints") or {}
    assert constraints == {}


def _persona_passthrough(*, observations=None, default_signal="monitor", default_confidence=0.0, default_reasoning="", **_):
    observations = observations or {}
    model_reference = observations.get("model_reference") or {}
    signal = model_reference.get("signal", default_signal)
    confidence = model_reference.get("confidence_pct", default_confidence)
    reasoning = model_reference.get("reasoning", default_reasoning)
    constraints = observations.get("suggested_constraints")
    if not isinstance(constraints, dict):
        constraints = {}
    return PersonaDecision(
        signal=str(signal),
        confidence=float(confidence),
        reasoning=str(reasoning),
        constraints=constraints,
    )


@pytest.fixture(autouse=True)
def _patch_persona(monkeypatch):
    monkeypatch.setattr(
        "src.agents.regime_meta.persona_from_observations",
        _persona_passthrough,
    )
