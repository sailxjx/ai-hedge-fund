import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.agents.short_cover_classifier import short_cover_classifier_agent
from src.data.models import Price


@pytest.fixture(autouse=True)
def _stub_short_cover_persona(monkeypatch):
    """Force the short-cover agent to mirror its quantitative recommendation during tests."""

    from src.agents import short_cover_classifier as module

    def _fake_persona_decision(state, agent_id, context):
        recommendation = context["model_recommendation"]
        return module.ShortCoverDecision(
            signal=recommendation["signal"],
            confidence=float(recommendation["confidence"]),
            reasoning=recommendation["reasoning"],
            constraints=recommendation.get("constraints") or {},
        )

    monkeypatch.setattr(module, "_request_llm_decision", _fake_persona_decision)


def _build_prices(count: int = 160, base_price: float = 250.0) -> list[Price]:
    base_dt = datetime(2023, 1, 1)
    prices: list[Price] = []
    for idx in range(count):
        level = base_price + np.sin(idx / 5.0) * 5 + idx * 0.1
        prices.append(
            Price(
                open=level,
                close=level,
                high=level * 1.01,
                low=level * 0.99,
                volume=2_000_000 + idx * 10_000,
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
            "end_date": "2024-07-01",
            "analyst_signals": {},
        },
        "metadata": {"show_reasoning": False},
    }


def _mock_dataset():
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    features = pd.DataFrame(
        {
            "ret_1": [0.01, -0.02, 0.03, 0.04, 0.05, 0.08],
            "ret_3": [0.02, -0.01, 0.04, 0.05, 0.06, 0.10],
            "ret_5": [0.03, -0.005, 0.045, 0.06, 0.07, 0.12],
            "ret_10": [0.04, 0.0, 0.05, 0.07, 0.08, 0.13],
            "ema_gap": [0.01, -0.01, 0.015, 0.02, 0.03, 0.05],
            "ema_slope": [0.005, -0.002, 0.008, 0.012, 0.02, 0.035],
            "breakout_gap": [0.0, -0.01, 0.005, 0.01, 0.015, 0.03],
            "retrace_gap": [0.01, 0.02, 0.03, 0.035, 0.04, 0.05],
            "volume_ratio": [0.0, -0.1, 0.2, 0.25, 0.3, 0.6],
            "vol_slope": [-0.01, -0.005, 0.01, 0.015, 0.02, 0.04],
        },
        index=index,
    )
    future_returns = pd.Series([-0.02, -0.01, 0.03, 0.04, 0.05, 0.06], index=index, name="future_ret_5")
    labels = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0, 1.0], index=index)
    return features, labels, future_returns


def _near_threshold_dataset():
    index = pd.date_range("2024-02-01", periods=6, freq="D")
    features = pd.DataFrame(
        {
            "ret_1": [0.01, 0.012, 0.014, 0.016, 0.018, 0.02],
            "ret_3": [0.015, 0.017, 0.018, 0.02, 0.022, 0.024],
            "ret_5": [0.02, 0.021, 0.022, 0.024, 0.026, 0.028],
            "ret_10": [0.025, 0.026, 0.027, 0.028, 0.03, 0.032],
            "ema_gap": [0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
            "ema_slope": [0.003, 0.004, 0.005, 0.006, 0.007, 0.008],
            "breakout_gap": [0.0, 0.001, 0.0015, 0.002, 0.0022, 0.0],
            "retrace_gap": [0.01, 0.011, 0.012, 0.013, 0.014, 0.015],
            "volume_ratio": [0.05, 0.055, 0.06, 0.065, 0.07, 0.075],
            "vol_slope": [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035],
            "vol_ratio": [0.02, 0.022, 0.024, 0.026, 0.028, 0.03],
            "atr_ratio": [0.01, 0.011, 0.012, 0.013, 0.014, 0.015],
            "volume_zscore": [0.1, 0.12, 0.15, 0.18, 0.2, 0.22],
        },
        index=index,
    )
    labels = pd.Series([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], index=index)
    future_returns = pd.Series([-0.015, -0.012, -0.01, -0.008, 0.032, 0.035], index=index, name="future_ret_5")
    return features, labels, future_returns


def _boost_dataset():
    index = pd.date_range("2024-03-01", periods=6, freq="D")
    features = pd.DataFrame(
        {
            "ret_1": [0.008, 0.01, 0.012, 0.016, 0.018, 0.021],
            "ret_3": [0.012, 0.014, 0.016, 0.02, 0.023, 0.027],
            "ret_5": [0.016, 0.018, 0.02, 0.024, 0.028, 0.033],
            "ret_10": [0.02, 0.022, 0.024, 0.028, 0.032, 0.038],
            "ema_gap": [0.004, 0.005, 0.006, 0.008, 0.012, 0.016],
            "ema_slope": [0.002, 0.003, 0.004, 0.006, 0.009, 0.013],
            "breakout_gap": [0.0, 0.001, 0.006, 0.012, 0.018, 0.03],
            "retrace_gap": [0.012, 0.013, 0.014, 0.015, 0.016, 0.017],
            "volume_ratio": [0.02, 0.04, 0.1, 0.18, 0.24, 0.35],
            "vol_slope": [0.001, 0.003, 0.007, 0.012, 0.018, 0.028],
            "vol_ratio": [0.04, 0.05, 0.09, 0.14, 0.17, 0.2],
            "atr_ratio": [0.009, 0.011, 0.014, 0.017, 0.02, 0.025],
            "volume_zscore": [0.2, 0.35, 0.6, 0.9, 1.4, 1.6],
        },
        index=index,
    )
    labels = pd.Series([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], index=index)
    future_returns = pd.Series([-0.012, -0.01, -0.008, 0.032, 0.045, 0.055], index=index, name="future_ret_5")
    return features, labels, future_returns


def _precision_dataset():
    index = pd.date_range("2024-05-01", periods=31, freq="D")
    features = pd.DataFrame(
        {
            "ret_1": np.linspace(0.0, 0.012, 31),
            "ret_3": np.linspace(0.0, 0.018, 31),
            "ret_5": np.linspace(0.0, 0.022, 31),
            "ret_10": np.linspace(0.0, 0.03, 31),
            "ema_gap": np.linspace(-0.01, 0.02, 31),
            "ema_slope": np.linspace(-0.005, 0.015, 31),
            "breakout_gap": np.linspace(0.0, 0.02, 31),
            "retrace_gap": np.linspace(0.01, 0.02, 31),
            "volume_ratio": np.linspace(-0.1, 0.4, 31),
            "vol_slope": np.linspace(-0.02, 0.03, 31),
            "vol_ratio": np.linspace(0.0, 0.11, 31),
            "atr_ratio": np.linspace(0.005, 0.04, 31),
            "volume_zscore": np.linspace(-0.5, 3.5, 31),
        },
        index=index,
    )

    labels = pd.Series([0] * 25 + [1, 1, 1, 1, 1, 0], index=index, dtype=float)
    future_returns = pd.Series(
        [-0.025] * 25 + [0.01, 0.055, 0.06, 0.07, 0.08, -0.01],
        index=index,
        name="future_ret_5",
    )

    return features, labels, future_returns


def test_classifier_forces_cover_on_high_probability(monkeypatch):
    from src.agents import short_cover_classifier as module

    monkeypatch.setattr(module, "MIN_OBSERVATIONS", 10, raising=False)
    monkeypatch.setattr(module, "MIN_TRAINING_SAMPLES", 5, raising=False)

    prices = _build_prices()
    monkeypatch.setattr(module, "get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(module, "prices_to_df", lambda *_: pd.DataFrame({"close": np.linspace(250, 260, len(prices)), "high": np.linspace(252, 262, len(prices)), "low": np.linspace(248, 258, len(prices)), "volume": np.linspace(2_000_000, 2_100_000, len(prices))}))

    features, labels, future_returns = _mock_dataset()
    monkeypatch.setattr(module, "_prepare_dataset", lambda *_: (features, labels, future_returns))

    def _weights(X, _y):
        weights = np.zeros(X.shape[1])
        weights[0] = -2.5
        weights[3] = 40.0
        return weights

    monkeypatch.setattr(module, "_fit_logistic", _weights)

    state = _base_state(short_shares=6)
    result = short_cover_classifier_agent(state)
    payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["TSLA"]

    assert payload["signal"] == "squeeze_cover"
    constraints = payload["constraints"]
    assert constraints["force_cover_qty"] == 6
    assert constraints["preferred_direction"] == "long"
    assert constraints["block_new_shorts"] is True
    assert constraints["target_short_shares"] == 0


def test_classifier_blocks_new_shorts_without_position(monkeypatch):
    from src.agents import short_cover_classifier as module

    monkeypatch.setattr(module, "MIN_OBSERVATIONS", 10, raising=False)
    monkeypatch.setattr(module, "MIN_TRAINING_SAMPLES", 5, raising=False)

    prices = _build_prices()
    monkeypatch.setattr(module, "get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(module, "prices_to_df", lambda *_: pd.DataFrame({"close": np.linspace(250, 260, len(prices)), "high": np.linspace(252, 262, len(prices)), "low": np.linspace(248, 258, len(prices)), "volume": np.linspace(2_000_000, 2_100_000, len(prices))}))

    features, labels, future_returns = _mock_dataset()
    monkeypatch.setattr(module, "_prepare_dataset", lambda *_: (features, labels, future_returns))

    def _weights(X, _y):
        weights = np.zeros(X.shape[1])
        weights[0] = -2.5
        weights[3] = 35.0
        return weights

    monkeypatch.setattr(module, "_fit_logistic", _weights)

    state = _base_state(short_shares=0)
    result = short_cover_classifier_agent(state)
    payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["TSLA"]

    assert payload["signal"] == "squeeze_risk"
    constraints = payload["constraints"]
    assert constraints["block_new_shorts"] is True
    assert constraints["target_short_shares"] == 0
    assert constraints["allow_short"] is False
    metrics = payload["metrics"]
    assert metrics["long_bias_neutralized"] is False


def test_classifier_trims_position_when_risk_stalls_near_threshold(monkeypatch):
    from src.agents import short_cover_classifier as module

    monkeypatch.setattr(module, "MIN_OBSERVATIONS", 10, raising=False)
    monkeypatch.setattr(module, "MIN_TRAINING_SAMPLES", 5, raising=False)

    prices = _build_prices()
    monkeypatch.setattr(module, "get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(
        module,
        "prices_to_df",
        lambda *_: pd.DataFrame(
            {
                "close": np.linspace(250, 260, len(prices)),
                "high": np.linspace(252, 262, len(prices)),
                "low": np.linspace(248, 258, len(prices)),
                "volume": np.linspace(2_000_000, 2_050_000, len(prices)),
            }
        ),
    )

    features, labels, future_returns = _near_threshold_dataset()
    monkeypatch.setattr(module, "_prepare_dataset", lambda *_: (features, labels, future_returns))
    monkeypatch.setattr(module, "_fit_logistic", lambda X, _y: np.zeros(X.shape[1]))

    train_probs = np.array([0.16, 0.18, 0.2, 0.215, 0.223])

    def _sigmoid_stub(values):
        values = np.asarray(values)
        if values.shape and values.shape[0] == len(train_probs):
            return train_probs
        return np.array([0.209])

    monkeypatch.setattr(module, "_sigmoid", _sigmoid_stub)

    state = _base_state(short_shares=10)
    result = short_cover_classifier_agent(state)

    payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["TSLA"]
    assert payload["signal"] == "trim_short"

    constraints = payload["constraints"]
    assert constraints["block_new_shorts"] is True
    assert constraints["allow_short"] is False
    assert constraints["max_additional_short_shares"] == 0
    assert constraints["target_short_shares"] < 10

    metrics = payload["metrics"]
    assert metrics["caution_trigger"] == "near_threshold"
    assert metrics["near_threshold_bias"] is True


def test_boost_squeeze_with_existing_short_promotes_full_cover(monkeypatch):
    from src.agents import short_cover_classifier as module

    monkeypatch.setattr(module, "MIN_OBSERVATIONS", 10, raising=False)
    monkeypatch.setattr(module, "MIN_TRAINING_SAMPLES", 5, raising=False)

    prices = _build_prices()
    monkeypatch.setattr(module, "get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(
        module,
        "prices_to_df",
        lambda *_: pd.DataFrame(
            {
                "close": np.linspace(250, 260, len(prices)),
                "high": np.linspace(252, 262, len(prices)),
                "low": np.linspace(248, 258, len(prices)),
                "volume": np.linspace(2_000_000, 2_060_000, len(prices)),
            }
        ),
    )

    features, labels, future_returns = _boost_dataset()
    monkeypatch.setattr(module, "_prepare_dataset", lambda *_: (features, labels, future_returns))
    monkeypatch.setattr(module, "_fit_logistic", lambda X, _y: np.zeros(X.shape[1]))

    train_probs = np.array([0.16, 0.18, 0.2, 0.205, 0.21])

    def _sigmoid_stub(values):
        values = np.asarray(values)
        if values.shape and values.shape[0] == len(train_probs):
            return train_probs
        return np.array([0.211])

    monkeypatch.setattr(module, "_sigmoid", _sigmoid_stub)
    monkeypatch.setattr(module.np, "quantile", lambda *_, **__: 0.24)

    def _boost_stub(row, improvement_raw, improvement_threshold):
        return 0.07, {
            "volatility_ratio": 0.08,
            "volume_zscore": 0.3,
            "atr_ratio": 0.017,
            "breakout_gap": float(row.get("breakout_gap", 0.0)),
        }

    monkeypatch.setattr(module, "_compute_probability_boost", _boost_stub)

    state = _base_state(short_shares=25)
    result = short_cover_classifier_agent(state)

    payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["TSLA"]
    assert payload["signal"] == "squeeze_cover"

    metrics = payload["metrics"]
    assert metrics["boost_driven_squeeze"] is False
    assert metrics["boost_reclassification"] == "squeeze_high_improvement"
    assert metrics["probability"] > metrics["threshold"]
    assert metrics["raw_probability"] < metrics["threshold"]


def test_near_threshold_bias_without_position_unblocks_new_shorts(monkeypatch):
    from src.agents import short_cover_classifier as module

    monkeypatch.setattr(module, "MIN_OBSERVATIONS", 10, raising=False)
    monkeypatch.setattr(module, "MIN_TRAINING_SAMPLES", 5, raising=False)

    prices = _build_prices()
    monkeypatch.setattr(module, "get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(
        module,
        "prices_to_df",
        lambda *_: pd.DataFrame(
            {
                "close": np.linspace(250, 260, len(prices)),
                "high": np.linspace(252, 262, len(prices)),
                "low": np.linspace(248, 258, len(prices)),
                "volume": np.linspace(2_000_000, 2_050_000, len(prices)),
            }
        ),
    )

    features, labels, future_returns = _near_threshold_dataset()
    monkeypatch.setattr(module, "_prepare_dataset", lambda *_: (features, labels, future_returns))
    monkeypatch.setattr(module, "_fit_logistic", lambda X, _y: np.zeros(X.shape[1]))

    train_probs = np.array([0.16, 0.18, 0.2, 0.215, 0.223])

    def _sigmoid_stub(values):
        values = np.asarray(values)
        if values.shape and values.shape[0] == len(train_probs):
            return train_probs
        return np.array([0.209])

    monkeypatch.setattr(module, "_sigmoid", _sigmoid_stub)

    state = _base_state(short_shares=0)
    result = short_cover_classifier_agent(state)

    payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["TSLA"]
    assert payload["signal"] == "neutral"

    assert payload.get("constraints") is None

    metrics = payload["metrics"]
    assert metrics["caution_trigger"] == "near_threshold"
    assert metrics["near_threshold_bias"] is True
    assert metrics["neutral_guardrail"] is True
    assert metrics["neutral_guardrail_relaxed"] is False
    assert metrics["auto_signal_before_guardrail"] == "bias_long"
    reasons = set(metrics.get("guardrail_reasons", []))
    assert {"raw probability below trigger", "marginal probability gap"}.issubset(reasons)
    retained = metrics["guardrail_retained_constraints"]
    assert "block_new_shorts" in retained
    assert "preferred_direction" in retained
    assert metrics["new_short_unblock_reason"] == "near_threshold_guidance"


def test_caution_bias_without_position_neutralises_long_direction(monkeypatch):
    from src.agents import short_cover_classifier as module

    monkeypatch.setattr(module, "MIN_OBSERVATIONS", 10, raising=False)
    monkeypatch.setattr(module, "MIN_TRAINING_SAMPLES", 5, raising=False)
    prices = _build_prices()
    monkeypatch.setattr(module, "get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(
        module,
        "prices_to_df",
        lambda *_: pd.DataFrame(
            {
                "close": np.linspace(250, 260, len(prices)),
                "high": np.linspace(252, 262, len(prices)),
                "low": np.linspace(248, 258, len(prices)),
                "volume": np.linspace(2_000_000, 2_050_000, len(prices)),
            }
        ),
    )

    features, labels, future_returns = _near_threshold_dataset()
    monkeypatch.setattr(module, "_prepare_dataset", lambda *_: (features, labels, future_returns))
    monkeypatch.setattr(module, "_fit_logistic", lambda X, _y: np.zeros(X.shape[1]))

    train_probs = np.array([0.16, 0.18, 0.2, 0.215, 0.223])

    def _sigmoid_stub(values):
        values = np.asarray(values)
        if values.shape and values.shape[0] == len(train_probs):
            return train_probs
        return np.array([0.209])

    monkeypatch.setattr(module, "_sigmoid", _sigmoid_stub)
    monkeypatch.setattr(module.np, "quantile", lambda *_, **__: 0.226)

    state = _base_state(short_shares=0)
    result = short_cover_classifier_agent(state)

    payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["TSLA"]
    assert payload["signal"] == "neutral"

    constraints = payload["constraints"]
    assert constraints["block_new_shorts"] is True
    assert constraints["preferred_direction"] == "neutral"
    assert constraints["allow_short"] is False

    metrics = payload["metrics"]
    assert metrics["neutral_guardrail"] is True
    assert metrics["neutral_guardrail_relaxed"] is False
    assert metrics["auto_signal_before_guardrail"] == "bias_long"
    assert metrics["bias_mode"].endswith("guardrail")
    retained = metrics["guardrail_retained_constraints"]
    assert "block_new_shorts" in retained
    assert "preferred_direction" in retained


def test_probability_boost_escalates_to_cover(monkeypatch):
    from src.agents import short_cover_classifier as module

    monkeypatch.setattr(module, "MIN_OBSERVATIONS", 10, raising=False)
    monkeypatch.setattr(module, "MIN_TRAINING_SAMPLES", 5, raising=False)

    prices = _build_prices()
    monkeypatch.setattr(module, "get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(
        module,
        "prices_to_df",
        lambda *_: pd.DataFrame(
            {
                "close": np.linspace(250, 260, len(prices)),
                "high": np.linspace(252, 263, len(prices)),
                "low": np.linspace(248, 257, len(prices)),
                "volume": np.linspace(2_000_000, 2_200_000, len(prices)),
            }
        ),
    )

    features, labels, future_returns = _boost_dataset()
    monkeypatch.setattr(module, "_prepare_dataset", lambda *_: (features, labels, future_returns))

    monkeypatch.setattr(module, "_fit_logistic", lambda X, _y: np.zeros(X.shape[1]))

    train_probs = np.array([0.22, 0.226, 0.231, 0.234, 0.238])

    def _sigmoid_stub(values):
        values = np.asarray(values)
        if values.shape and values.shape[0] == len(train_probs):
            return train_probs
        return np.array([0.228])

    monkeypatch.setattr(module, "_sigmoid", _sigmoid_stub)
    monkeypatch.setattr(module.np, "quantile", lambda *_, **__: 0.235)

    state = _base_state(short_shares=12)
    result = short_cover_classifier_agent(state)

    payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["TSLA"]
    assert payload["signal"] == "squeeze_cover"
    constraints = payload["constraints"]
    assert constraints["force_cover_qty"] == 12
    assert constraints["block_new_shorts"] is True
    assert constraints["allow_short"] is False

    metrics = payload["metrics"]
    assert metrics["probability"] > metrics["threshold"]
    assert metrics["probability_boost"] > 0.0
    assert metrics["boost_driven_squeeze"] is False


def test_probability_boost_without_raw_support_trims_instead_of_cover(monkeypatch):
    from src.agents import short_cover_classifier as module

    monkeypatch.setattr(module, "MIN_OBSERVATIONS", 10, raising=False)
    monkeypatch.setattr(module, "MIN_TRAINING_SAMPLES", 5, raising=False)

    prices = _build_prices()
    monkeypatch.setattr(module, "get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(
        module,
        "prices_to_df",
        lambda *_: pd.DataFrame(
            {
                "close": np.linspace(250, 260, len(prices)),
                "high": np.linspace(252, 263, len(prices)),
                "low": np.linspace(248, 257, len(prices)),
                "volume": np.linspace(2_000_000, 2_200_000, len(prices)),
            }
        ),
    )

    features, labels, future_returns = _boost_dataset()
    monkeypatch.setattr(module, "_prepare_dataset", lambda *_: (features, labels, future_returns))
    monkeypatch.setattr(module, "_fit_logistic", lambda X, _y: np.zeros(X.shape[1]))

    train_probs = np.array([0.14, 0.16, 0.18, 0.19, 0.2])

    def _sigmoid_stub(values):
        values = np.asarray(values)
        if values.shape and values.shape[0] == len(train_probs):
            return train_probs
        return np.array([0.18])

    monkeypatch.setattr(module, "_sigmoid", _sigmoid_stub)
    monkeypatch.setattr(module.np, "quantile", lambda *_, **__: 0.26)

    state = _base_state(short_shares=10)
    result = short_cover_classifier_agent(state)

    payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["TSLA"]
    assert payload["signal"] == "neutral"
    metrics = payload["metrics"]
    assert metrics["boost_clamp_factor"] < 1.0
    assert metrics["probability_boost"] < metrics["probability_boost_raw"]
    assert metrics["boost_driven_squeeze"] is False
    assert metrics["boost_driven_soft"] is False
    assert metrics["near_threshold_bias"] is False


def test_probability_boost_without_position_unblocks_new_shorts(monkeypatch):
    from src.agents import short_cover_classifier as module

    monkeypatch.setattr(module, "MIN_OBSERVATIONS", 10, raising=False)
    monkeypatch.setattr(module, "MIN_TRAINING_SAMPLES", 5, raising=False)

    prices = _build_prices()
    monkeypatch.setattr(module, "get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(
        module,
        "prices_to_df",
        lambda *_: pd.DataFrame(
            {
                "close": np.linspace(250, 260, len(prices)),
                "high": np.linspace(252, 263, len(prices)),
                "low": np.linspace(248, 257, len(prices)),
                "volume": np.linspace(2_000_000, 2_200_000, len(prices)),
            }
        ),
    )

    features, labels, future_returns = _boost_dataset()
    monkeypatch.setattr(module, "_prepare_dataset", lambda *_: (features, labels, future_returns))
    monkeypatch.setattr(module, "_fit_logistic", lambda X, _y: np.zeros(X.shape[1]))

    train_probs = np.array([0.14, 0.16, 0.18, 0.19, 0.2])

    def _sigmoid_stub(values):
        values = np.asarray(values)
        if values.shape and values.shape[0] == len(train_probs):
            return train_probs
        return np.array([0.18])

    monkeypatch.setattr(module, "_sigmoid", _sigmoid_stub)
    monkeypatch.setattr(module.np, "quantile", lambda *_, **__: 0.26)

    state = _base_state(short_shares=0)
    result = short_cover_classifier_agent(state)

    payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["TSLA"]
    assert payload["signal"] == "neutral"
    assert payload.get("constraints") is None
    metrics = payload["metrics"]
    assert metrics["boost_clamp_factor"] < 1.0
    assert metrics["probability_boost"] < metrics["probability_boost_raw"]


def test_boost_guardrail_relaxes_after_precision_streak(monkeypatch):
    from src.agents import short_cover_classifier as module

    metrics = {
        "probability": 0.236,
        "raw_probability": 0.219,
        "threshold": 0.24,
        "improvement": 0.0025,
        "improvement_threshold_effective": 0.003,
        "boost_driven_squeeze": True,
        "boost_driven_soft": False,
        "caution_trigger": "boost_only",
        "precision_boost_trim_streak": 0,
    }

    guardrail_active, _ = module._assess_neutral_guardrail(dict(metrics), existing_short=0)
    assert guardrail_active is True

    metrics_with_streak = dict(metrics, precision_boost_trim_streak=module.NEUTRAL_GUARDRAIL_STREAK_RELAX)
    relaxed_active, _ = module._assess_neutral_guardrail(metrics_with_streak, existing_short=0)
    assert relaxed_active is False
    assert metrics_with_streak["neutral_guardrail"] is False
    assert metrics_with_streak["neutral_guardrail_relaxed"] is True
    assert "precision_boost_streak" in metrics_with_streak["neutral_guardrail_relaxed_reasons"]

    notes = module._build_persona_notes(metrics_with_streak, existing_short=0)
    assert any("NEUTRAL_GUARDRAIL RELAXED" in note for note in notes)


def test_soft_bias_margin_does_not_block_new_shorts(monkeypatch):
    from src.agents import short_cover_classifier as module

    monkeypatch.setattr(module, "MIN_OBSERVATIONS", 10, raising=False)
    monkeypatch.setattr(module, "MIN_TRAINING_SAMPLES", 5, raising=False)

    index = pd.date_range("2024-04-01", periods=6, freq="D")
    features = pd.DataFrame(
        {
            "ret_1": [0.0] * 6,
            "ret_3": [0.0] * 6,
            "ret_5": [0.0] * 6,
            "ret_10": [0.0] * 6,
            "ema_gap": [0.0] * 6,
            "ema_slope": [0.0] * 6,
            "breakout_gap": [0.0] * 6,
            "retrace_gap": [0.0] * 6,
            "volume_ratio": [0.0] * 6,
            "vol_slope": [0.0] * 6,
            "vol_ratio": [0.0] * 6,
            "atr_ratio": [0.0] * 6,
            "volume_zscore": [0.0] * 6,
        },
        index=index,
    )

    labels = pd.Series([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], index=index)
    future_returns = pd.Series([-0.04, -0.035, -0.03, -0.028, 0.135, 0.14], index=index, name="future_ret_5")

    monkeypatch.setattr(module, "get_prices", lambda *_, **__: _build_prices())
    monkeypatch.setattr(module, "prices_to_df", lambda *_: pd.DataFrame({"close": np.linspace(250, 255, 160), "high": np.linspace(251, 256, 160), "low": np.linspace(249, 254, 160), "volume": np.linspace(2_000_000, 2_100_000, 160)}))
    monkeypatch.setattr(module, "_prepare_dataset", lambda *_: (features, labels, future_returns))
    monkeypatch.setattr(module, "_fit_logistic", lambda X, _y: np.zeros(X.shape[1]))
    monkeypatch.setattr(module, "get_platt_parameters", lambda *_, **__: {})

    train_probs = np.array([0.12, 0.13, 0.14, 0.145, 0.16])

    def _sigmoid_stub(values):
        values = np.asarray(values)
        if values.shape and values.shape[0] == len(train_probs):
            return train_probs
        return np.array([0.21])

    monkeypatch.setattr(module, "_sigmoid", _sigmoid_stub)
    monkeypatch.setattr(module.np, "quantile", lambda *_args, **_kwargs: 0.24)

    state = _base_state(short_shares=0)
    result = short_cover_classifier_agent(state)
    payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["TSLA"]

    assert payload["signal"] == "neutral"
    constraints = payload["constraints"]
    assert constraints == {"preferred_direction": "neutral"}
    metrics = payload["metrics"]
    assert metrics["neutral_guardrail"] is True
    assert metrics["neutral_guardrail_relaxed"] is False
    assert metrics["auto_signal_before_guardrail"] == "bias_long"
    assert metrics["bias_delta"] == pytest.approx(0.01, rel=1e-3)
    assert metrics["guardrail_retained_constraints"] == ["preferred_direction"]


def test_precision_bonus_allows_boost_squeeze_to_cover(monkeypatch):
    from src.agents import short_cover_classifier as module

    monkeypatch.setattr(module, "MIN_OBSERVATIONS", 10, raising=False)
    monkeypatch.setattr(module, "MIN_TRAINING_SAMPLES", 5, raising=False)
    monkeypatch.setattr(module, "PRECISION_MIN_SAMPLES", 0, raising=False)
    monkeypatch.setattr(module, "PRECISION_MIN_PRECISION", 0.0, raising=False)
    monkeypatch.setitem(module.PRECISION_TICKER_OVERRIDES["NVDA"], "min_precision", 0.0)
    monkeypatch.setitem(module.PRECISION_TICKER_OVERRIDES["NVDA"], "min_samples", 0)
    monkeypatch.setitem(module.PRECISION_TICKER_OVERRIDES["NVDA"], "min_precision", 0.0)
    monkeypatch.setitem(module.PRECISION_TICKER_OVERRIDES["NVDA"], "min_samples", 0)
    monkeypatch.setitem(module.PRECISION_TICKER_OVERRIDES["NVDA"], "min_precision", 0.0)
    monkeypatch.setitem(module.PRECISION_TICKER_OVERRIDES["NVDA"], "min_samples", 0)

    index = pd.date_range("2024-05-01", periods=31, freq="D")
    features = pd.DataFrame(
        {
            "ret_1": np.linspace(0.0, 0.012, 31),
            "ret_3": np.linspace(0.0, 0.018, 31),
            "ret_5": np.linspace(0.0, 0.022, 31),
            "ret_10": np.linspace(0.0, 0.03, 31),
            "ema_gap": np.linspace(-0.01, 0.02, 31),
            "ema_slope": np.linspace(-0.005, 0.015, 31),
            "breakout_gap": np.linspace(0.0, 0.02, 31),
            "retrace_gap": np.linspace(0.01, 0.02, 31),
            "volume_ratio": np.linspace(-0.1, 0.4, 31),
            "vol_slope": np.linspace(-0.02, 0.03, 31),
            "vol_ratio": np.linspace(0.0, 0.11, 31),
            "atr_ratio": np.linspace(0.005, 0.04, 31),
            "volume_zscore": np.linspace(-0.5, 3.5, 31),
        },
        index=index,
    )

    labels = pd.Series([0] * 25 + [1, 1, 1, 1, 1, 0], index=index, dtype=float)
    future_returns = pd.Series(
        [-0.025] * 25 + [0.01, 0.055, 0.06, 0.07, 0.08, -0.01],
        index=index,
        name="future_ret_5",
    )

    monkeypatch.setattr(module, "get_prices", lambda *_, **__: _build_prices())
    monkeypatch.setattr(
        module,
        "prices_to_df",
        lambda *_: pd.DataFrame(
            {
                "close": np.linspace(250, 255, 160),
                "high": np.linspace(252, 258, 160),
                "low": np.linspace(248, 252, 160),
                "volume": np.linspace(2_000_000, 2_300_000, 160),
            }
        ),
    )
    monkeypatch.setattr(module, "_prepare_dataset", lambda *_: (features, labels, future_returns))
    monkeypatch.setattr(module, "_fit_logistic", lambda X, _y: np.zeros(X.shape[1]))

    train_probs = np.concatenate([np.linspace(0.12, 0.2, 25), np.array([0.225, 0.232, 0.24, 0.248, 0.255])])

    def _sigmoid_stub(values):
        values = np.asarray(values)
        if values.shape and values.shape[0] == len(train_probs):
            return train_probs
        return np.array([0.215])

    monkeypatch.setattr(module, "_sigmoid", _sigmoid_stub)
    monkeypatch.setattr(module.np, "quantile", lambda *_, **__: 0.235)

    state = _base_state(short_shares=8)
    state["data"]["tickers"] = ["NVDA"]
    state["data"]["portfolio"]["positions"]["NVDA"] = state["data"]["portfolio"]["positions"].pop("TSLA")

    result = short_cover_classifier_agent(state)
    payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["NVDA"]

    assert payload["signal"] == "squeeze_cover"
    constraints = payload["constraints"]
    assert constraints["force_cover_qty"] == 8
    assert constraints["block_new_shorts"] is True
    assert constraints["allow_short"] is False

    metrics = payload["metrics"]
    assert metrics["precision_bonus_active"] is True
    assert metrics["historical_precision"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["threshold"] < metrics["base_threshold"]
    assert metrics["precision_margin_bonus"] > 0.0
    assert metrics["boost_driven_squeeze"] is False


def test_precision_trim_streak_scales_precision_shift(monkeypatch):
    from src.agents import short_cover_classifier as module

    monkeypatch.setattr(module, "MIN_OBSERVATIONS", 10, raising=False)
    monkeypatch.setattr(module, "MIN_TRAINING_SAMPLES", 5, raising=False)
    monkeypatch.setattr(module, "PRECISION_MIN_SAMPLES", 0, raising=False)
    monkeypatch.setattr(module, "PRECISION_MIN_PRECISION", 0.0, raising=False)
    monkeypatch.setitem(module.PRECISION_TICKER_OVERRIDES["NVDA"], "min_precision", 0.0)
    monkeypatch.setitem(module.PRECISION_TICKER_OVERRIDES["NVDA"], "min_samples", 0)
    monkeypatch.setattr(module, "PRECISION_STREAK_MARGIN_DECAY", 0.0, raising=False)
    monkeypatch.setattr(module, "PRECISION_STREAK_SHIFT_RECOVERY", 0.0015, raising=False)
    target_streak = module.PRECISION_STREAK_TRIGGER + 1
    monkeypatch.setattr(module, "_estimate_boost_trim_streak", lambda *_, **__: target_streak, raising=False)

    features, labels, future_returns = _precision_dataset()

    monkeypatch.setattr(module, "get_prices", lambda *_, **__: _build_prices())
    monkeypatch.setattr(
        module,
        "prices_to_df",
        lambda *_: pd.DataFrame(
            {
                "close": np.linspace(250, 255, 160),
                "high": np.linspace(252, 258, 160),
                "low": np.linspace(248, 252, 160),
                "volume": np.linspace(2_000_000, 2_300_000, 160),
            }
        ),
    )
    monkeypatch.setattr(module, "_prepare_dataset", lambda *_: (features, labels, future_returns))
    monkeypatch.setattr(module, "_fit_logistic", lambda X, _y: np.zeros(X.shape[1]))

    train_probs = np.concatenate([np.linspace(0.12, 0.2, 25), np.array([0.225, 0.232, 0.24, 0.248, 0.255])])

    def _sigmoid_stub(values):
        values = np.asarray(values)
        if values.shape and values.shape[0] == len(train_probs):
            return train_probs
        return np.array([0.215])

    monkeypatch.setattr(module, "_sigmoid", _sigmoid_stub)
    monkeypatch.setattr(module.np, "quantile", lambda *_, **__: 0.235)

    state = _base_state(short_shares=8)
    state["data"]["tickers"] = ["NVDA"]
    state["data"]["portfolio"]["positions"]["NVDA"] = state["data"]["portfolio"]["positions"].pop("TSLA")

    result = short_cover_classifier_agent(state)
    payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["NVDA"]
    metrics = payload["metrics"]

    assert metrics["precision_boost_trim_streak"] == target_streak
    assert metrics["precision_margin_bonus_base"] > 0.0
    margin_floor = max(
        module.PRECISION_STREAK_MARGIN_FLOOR,
        metrics["precision_margin_bonus_base"] * module.PRECISION_STREAK_MARGIN_MIN_RATIO,
    )
    assert metrics["precision_margin_floor"] == pytest.approx(margin_floor, rel=1e-6)
    assert metrics["precision_margin_bonus"] >= margin_floor - 1e-9
    assert metrics["precision_margin_bonus"] <= metrics["precision_margin_bonus_base"]
    if math.isclose(metrics["precision_margin_bonus"], margin_floor, rel_tol=1e-6, abs_tol=1e-6):
        assert metrics["precision_margin_clamped"] is True
    else:
        assert metrics["precision_margin_clamped"] is False
    expected_adjustment = metrics["precision_margin_bonus"] / metrics["precision_margin_bonus_base"]
    assert metrics["precision_margin_adjustment"] == pytest.approx(expected_adjustment, rel=1e-6)
    assert metrics["precision_threshold_shift"] >= metrics["precision_threshold_shift_base"]
    assert metrics["precision_threshold_adjustment"] >= 1.0
    assert metrics["precision_streak_shift_recovery"] >= 0.0
    assert metrics["precision_streak_improvement_discount"] > 0.0


def test_precision_trim_streak_escalates_trim_ratio(monkeypatch):
    from src.agents import short_cover_classifier as module

    base_ratio = 0.35
    streak_active_ratio, streak_metrics = module._scale_trim_ratio_for_precision(
        trim_ratio_base=base_ratio,
        precision_override_active=True,
        precision_boost_trim_streak=module.PRECISION_STREAK_TRIGGER + 2,
        precision_streak_trigger=module.PRECISION_STREAK_TRIGGER,
        precision_streak_trim_scale=module.PRECISION_STREAK_TRIM_SCALE,
        precision_streak_trim_cap=module.PRECISION_STREAK_TRIM_CAP,
    )

    assert streak_metrics["precision_streak_trim_active"] is True
    assert streak_metrics["precision_streak_trim_factor"] == 3  # trigger + 2 ⇒ factor 3
    assert streak_metrics["precision_streak_trim_multiplier"] > 1.0
    assert streak_active_ratio > base_ratio
    assert streak_active_ratio <= module.PRECISION_STREAK_TRIM_CAP + 1e-9

    baseline_ratio, baseline_metrics = module._scale_trim_ratio_for_precision(
        trim_ratio_base=base_ratio,
        precision_override_active=False,
        precision_boost_trim_streak=module.PRECISION_STREAK_TRIGGER + 5,
        precision_streak_trigger=module.PRECISION_STREAK_TRIGGER,
        precision_streak_trim_scale=module.PRECISION_STREAK_TRIM_SCALE,
        precision_streak_trim_cap=module.PRECISION_STREAK_TRIM_CAP,
    )

    assert baseline_ratio == base_ratio
    assert baseline_metrics["precision_streak_trim_active"] is False
    assert baseline_metrics["precision_streak_trim_multiplier"] == 1.0


def test_precision_trim_streak_below_trigger_keeps_baseline(monkeypatch):
    from src.agents import short_cover_classifier as module

    monkeypatch.setattr(module, "MIN_OBSERVATIONS", 10, raising=False)
    monkeypatch.setattr(module, "MIN_TRAINING_SAMPLES", 5, raising=False)
    monkeypatch.setattr(module, "PRECISION_MIN_SAMPLES", 5, raising=False)
    monkeypatch.setattr(module, "PRECISION_MIN_PRECISION", 0.6, raising=False)

    features, labels, future_returns = _precision_dataset()

    monkeypatch.setattr(module, "get_prices", lambda *_, **__: _build_prices())
    monkeypatch.setattr(
        module,
        "prices_to_df",
        lambda *_: pd.DataFrame(
            {
                "close": np.linspace(250, 255, 160),
                "high": np.linspace(252, 258, 160),
                "low": np.linspace(248, 252, 160),
                "volume": np.linspace(2_000_000, 2_300_000, 160),
            }
        ),
    )
    monkeypatch.setattr(module, "_prepare_dataset", lambda *_: (features, labels, future_returns))
    monkeypatch.setattr(module, "_fit_logistic", lambda X, _y: np.zeros(X.shape[1]))

    train_probs = np.concatenate([np.linspace(0.12, 0.2, 25), np.array([0.225, 0.232, 0.24, 0.248, 0.255])])

    def _sigmoid_stub(values):
        values = np.asarray(values)
        if values.shape and values.shape[0] == len(train_probs):
            return train_probs
        return np.array([0.215])

    monkeypatch.setattr(module, "_sigmoid", _sigmoid_stub)
    monkeypatch.setattr(module.np, "quantile", lambda *_, **__: 0.235)

    target_streak = max(module.PRECISION_STREAK_TRIGGER - 1, 0)
    monkeypatch.setattr(module, "_estimate_boost_trim_streak", lambda *_, **__: target_streak, raising=False)

    state = _base_state(short_shares=8)
    state["data"]["tickers"] = ["NVDA"]
    state["data"]["portfolio"]["positions"]["NVDA"] = state["data"]["portfolio"]["positions"].pop("TSLA")

    result = short_cover_classifier_agent(state)
    payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["NVDA"]
    metrics = payload["metrics"]

    assert metrics["precision_boost_trim_streak"] == target_streak
    assert metrics["precision_margin_bonus_base"] > 0.0
    assert metrics["precision_margin_bonus"] == pytest.approx(metrics["precision_margin_bonus_base"], rel=1e-6)
    assert metrics["precision_margin_adjustment"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["precision_threshold_shift"] == pytest.approx(metrics["precision_threshold_shift_base"], rel=1e-6)
    assert metrics["precision_threshold_adjustment"] == pytest.approx(1.0, rel=1e-6)
    margin_floor = max(
        module.PRECISION_STREAK_MARGIN_FLOOR,
        metrics["precision_margin_bonus_base"] * module.PRECISION_STREAK_MARGIN_MIN_RATIO,
    )
    assert metrics["precision_margin_floor"] == pytest.approx(margin_floor, rel=1e-6)
    assert metrics["precision_margin_clamped"] is False
    assert metrics["precision_streak_shift_recovery"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["precision_streak_improvement_discount"] == pytest.approx(0.0, abs=1e-6)


def test_precision_improvement_discount_enables_soft_trigger(monkeypatch):
    from src.agents import short_cover_classifier as module

    monkeypatch.setattr(module, "MIN_OBSERVATIONS", 10, raising=False)
    monkeypatch.setattr(module, "MIN_TRAINING_SAMPLES", 5, raising=False)
    monkeypatch.setattr(module, "PRECISION_MIN_SAMPLES", 3, raising=False)
    monkeypatch.setattr(module, "PRECISION_MIN_PRECISION", 0.6, raising=False)
    monkeypatch.setattr(module, "TRIGGER_LEEWAY", 0.015, raising=False)
    monkeypatch.setattr(module, "PRECISION_THRESHOLD_SHIFT_MAX", 0.0, raising=False)
    monkeypatch.setattr(module, "PRECISION_THRESHOLD_SHIFT_SCALE", 0.0, raising=False)
    monkeypatch.setattr(module, "PRECISION_MARGIN_SCALE", 0.0, raising=False)
    monkeypatch.setattr(module, "PRECISION_MARGIN_MAX", 0.0, raising=False)

    features, labels, future_returns = _precision_dataset()
    prices = _build_prices()

    monkeypatch.setattr(module, "get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(
        module,
        "prices_to_df",
        lambda *_: pd.DataFrame(
            {
                "close": np.linspace(250, 255, len(prices)),
                "high": np.linspace(252, 258, len(prices)),
                "low": np.linspace(248, 252, len(prices)),
                "volume": np.linspace(2_000_000, 2_300_000, len(prices)),
            }
        ),
    )
    monkeypatch.setattr(module, "_prepare_dataset", lambda *_: (features, labels, future_returns))
    monkeypatch.setattr(module, "_fit_logistic", lambda X, _y: np.zeros(X.shape[1]))
    monkeypatch.setattr(
        module,
        "_compute_probability_boost",
        lambda *_: (
            0.0,
            {
                "volatility_ratio": 0.0,
                "volume_zscore": 0.0,
                "atr_ratio": 0.0,
                "breakout_gap": 0.0,
            },
        ),
    )

    train_probs = np.concatenate([
        np.linspace(0.13, 0.17, len(features) - 6),
        np.array([0.18, 0.182, 0.184, 0.187, 0.19]),
    ])

    def _sigmoid_stub(values):
        values = np.asarray(values)
        if values.shape and values.shape[0] == len(train_probs):
            return train_probs
        return np.array([0.186])

    monkeypatch.setattr(module, "_sigmoid", _sigmoid_stub)
    monkeypatch.setattr(module.np, "quantile", lambda *_: 0.195)

    def _make_state():
        state = _base_state(short_shares=6)
        state["data"]["tickers"] = ["NVDA"]
        position = state["data"]["portfolio"]["positions"].pop("TSLA")
        state["data"]["portfolio"]["positions"]["NVDA"] = position
        return state

    with monkeypatch.context() as ctx:
        ctx.setattr(module, "PRECISION_IMPROVEMENT_DISCOUNT_SCALE", 0.0, raising=False)
        ctx.setattr(module, "PRECISION_IMPROVEMENT_DISCOUNT_MAX", 0.0, raising=False)
        result = module.short_cover_classifier_agent(_make_state())
        payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["NVDA"]
        metrics = payload["metrics"]
        assert payload["signal"] == "trim_short"
        assert metrics["soft_trigger"] is False
        assert metrics["squeeze_trigger"] is False
        assert metrics["improvement_threshold"] == pytest.approx(metrics["improvement_threshold_effective"], rel=1e-6)

    with monkeypatch.context() as ctx:
        ctx.setattr(module, "PRECISION_IMPROVEMENT_DISCOUNT_SCALE", 1.6, raising=False)
        ctx.setattr(module, "PRECISION_IMPROVEMENT_DISCOUNT_MAX", 0.45, raising=False)
        result = module.short_cover_classifier_agent(_make_state())
        payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["NVDA"]
        metrics = payload["metrics"]

    assert payload["signal"] == "squeeze_cover"
    assert metrics["soft_trigger"] is True
    assert metrics["squeeze_trigger"] is False
    assert metrics["precision_improvement_discount"] > 0
    assert metrics["precision_improvement_discount_raw"] >= metrics["precision_improvement_discount"]
    assert metrics["precision_improvement_discount_factor"] <= 1.0
    assert metrics["precision_improvement_discount_factor"] >= module.PRECISION_IMPROVEMENT_DISCOUNT_MIN_FACTOR
    assert metrics["precision_trend_strength"] >= 0.0
    assert metrics["improvement_threshold_effective"] < metrics["improvement_threshold"]
    assert metrics["improvement"] >= metrics["improvement_threshold_effective"]


def test_precision_improvement_discount_trend_damping(monkeypatch):
    from src.agents import short_cover_classifier as module

    monkeypatch.setattr(module, "MIN_OBSERVATIONS", 10, raising=False)
    monkeypatch.setattr(module, "MIN_TRAINING_SAMPLES", 5, raising=False)
    monkeypatch.setattr(module, "PRECISION_MIN_SAMPLES", 3, raising=False)
    monkeypatch.setattr(module, "PRECISION_MIN_PRECISION", 0.6, raising=False)
    monkeypatch.setattr(module, "TRIGGER_LEEWAY", 0.012, raising=False)
    monkeypatch.setattr(module, "PRECISION_THRESHOLD_SHIFT_MAX", 0.0, raising=False)
    monkeypatch.setattr(module, "PRECISION_THRESHOLD_SHIFT_SCALE", 0.0, raising=False)
    monkeypatch.setattr(module, "PRECISION_MARGIN_SCALE", 0.0, raising=False)
    monkeypatch.setattr(module, "PRECISION_MARGIN_MAX", 0.0, raising=False)

    base_features, labels, future_returns = _precision_dataset()
    prices = _build_prices()

    monkeypatch.setattr(module, "get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(
        module,
        "prices_to_df",
        lambda *_: pd.DataFrame(
            {
                "close": np.linspace(250, 255, len(prices)),
                "high": np.linspace(252, 258, len(prices)),
                "low": np.linspace(248, 252, len(prices)),
                "volume": np.linspace(2_000_000, 2_300_000, len(prices)),
            }
        ),
    )
    monkeypatch.setattr(module, "_fit_logistic", lambda X, _y: np.zeros(X.shape[1]))
    monkeypatch.setattr(
        module,
        "_compute_probability_boost",
        lambda *_: (
            0.0,
            {
                "volatility_ratio": 0.0,
                "volume_zscore": 0.0,
                "atr_ratio": 0.0,
                "breakout_gap": 0.0,
            },
        ),
    )

    train_probs = np.concatenate([
        np.linspace(0.13, 0.17, len(base_features) - 6),
        np.array([0.18, 0.182, 0.184, 0.187, 0.19]),
    ])

    def _sigmoid_stub(values):
        values = np.asarray(values)
        if values.shape and values.shape[0] == len(train_probs):
            return train_probs
        return np.array([0.186])

    monkeypatch.setattr(module, "_sigmoid", _sigmoid_stub)
    monkeypatch.setattr(module.np, "quantile", lambda *_: 0.195)

    def _make_state():
        state = _base_state(short_shares=6)
        state["data"]["tickers"] = ["NVDA"]
        position = state["data"]["portfolio"]["positions"].pop("TSLA")
        state["data"]["portfolio"]["positions"]["NVDA"] = position
        return state

    bullish_features = base_features.copy()
    bullish_features.iloc[-1, bullish_features.columns.get_loc("ema_gap")] = 0.045
    bullish_features.iloc[-1, bullish_features.columns.get_loc("ema_slope")] = 0.03
    bullish_features.iloc[-1, bullish_features.columns.get_loc("breakout_gap")] = 0.028

    with monkeypatch.context() as ctx:
        ctx.setattr(module, "_prepare_dataset", lambda *_: (bullish_features, labels, future_returns))
        result = module.short_cover_classifier_agent(_make_state())
        payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["NVDA"]
        bullish_metrics = payload["metrics"]

    assert bullish_metrics["precision_improvement_discount_raw"] > bullish_metrics["precision_improvement_discount"]
    assert bullish_metrics["precision_improvement_discount_factor"] < 1.0
    assert bullish_metrics["precision_improvement_discount_factor"] >= module.PRECISION_IMPROVEMENT_DISCOUNT_MIN_FACTOR
    assert bullish_metrics["precision_trend_strength"] > 0.0

    neutral_features = base_features.copy()
    neutral_features.iloc[-1, neutral_features.columns.get_loc("ema_gap")] = -0.015
    neutral_features.iloc[-1, neutral_features.columns.get_loc("ema_slope")] = -0.01
    neutral_features.iloc[-1, neutral_features.columns.get_loc("breakout_gap")] = 0.0

    with monkeypatch.context() as ctx:
        ctx.setattr(module, "_prepare_dataset", lambda *_: (neutral_features, labels, future_returns))
        result = module.short_cover_classifier_agent(_make_state())
        payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["NVDA"]
        neutral_metrics = payload["metrics"]

    assert neutral_metrics["precision_improvement_discount"] == pytest.approx(
        min(
            neutral_metrics["precision_improvement_discount_raw"],
            module.PRECISION_IMPROVEMENT_DISCOUNT_MAX,
        )
    )
    assert neutral_metrics["precision_improvement_discount_factor"] == pytest.approx(1.0)
    assert neutral_metrics["precision_trend_strength"] == pytest.approx(0.0)

def test_boost_guidance_unblocks_new_shorts(monkeypatch):
    from src.agents import short_cover_classifier as module

    monkeypatch.setattr(module, "MIN_OBSERVATIONS", 10, raising=False)
    monkeypatch.setattr(module, "MIN_TRAINING_SAMPLES", 5, raising=False)

    features, labels, future_returns = _boost_dataset()
    prices = _build_prices()

    monkeypatch.setattr(module, "get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(
        module,
        "prices_to_df",
        lambda *_: pd.DataFrame(
            {
                "close": np.linspace(250, 255, len(prices)),
                "high": np.linspace(252, 258, len(prices)),
                "low": np.linspace(248, 252, len(prices)),
                "volume": np.linspace(2_000_000, 2_300_000, len(prices)),
            }
        ),
    )
    monkeypatch.setattr(module, "_prepare_dataset", lambda *_: (features, labels, future_returns))
    monkeypatch.setattr(module, "_fit_logistic", lambda X, _y: np.zeros(X.shape[1]))

    train_probs = np.concatenate([np.linspace(0.16, 0.19, len(features) - 1), np.array([0.19])])

    def _sigmoid_stub(values):
        values = np.asarray(values)
        if values.shape and values.shape[0] == len(train_probs):
            return train_probs
        return np.array([0.19])

    monkeypatch.setattr(module, "_sigmoid", _sigmoid_stub)
    monkeypatch.setattr(module.np, "quantile", lambda *_: 0.215)

    monkeypatch.setattr(
        module,
        "_compute_probability_boost",
        lambda *_: (
            0.04,
            {
                "volatility_ratio": 0.2,
                "volume_zscore": 1.2,
                "atr_ratio": 0.02,
                "breakout_gap": 0.015,
            },
        ),
    )

    state = _base_state(short_shares=0)
    result = short_cover_classifier_agent(state)
    payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["TSLA"]
    metrics = payload["metrics"]

    assert payload["signal"] == "neutral"
    assert "constraints" not in payload
    assert metrics["neutral_guardrail"] is True
    assert metrics["neutral_guardrail_relaxed"] is False
    assert metrics["auto_signal_before_guardrail"] == "bias_long"
    assert metrics["boost_clamp_factor"] < 1.0
    assert metrics["probability_boost"] < metrics["probability_boost_raw"]
    assert metrics["boost_driven_squeeze"] is False
    assert metrics["boost_driven_soft"] is False



def test_classifier_returns_neutral_on_empty_dataset(monkeypatch):
    from src.agents import short_cover_classifier as module

    prices = _build_prices(count=5)
    monkeypatch.setattr(module, "get_prices", lambda *_, **__: prices)
    monkeypatch.setattr(module, "prices_to_df", lambda *_: pd.DataFrame({"close": [250, 251, 252, 253, 254]}))
    monkeypatch.setattr(module, "_prepare_dataset", lambda *_: (pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)))

    state = _base_state(short_shares=4)
    result = short_cover_classifier_agent(state)
    payload = result["data"]["analyst_signals"]["short_cover_classifier_agent"]["TSLA"]

    assert payload["signal"] == "neutral"
    assert payload["confidence"] <= 35
    assert payload["constraints"] == {}


def test_llm_context_carries_guardrail_summary():
    from src.agents import short_cover_classifier as module

    metrics = {
        "probability": 0.23,
        "raw_probability": 0.21,
        "threshold": 0.25,
        "improvement": -0.001,
        "improvement_threshold_effective": 0.004,
    }
    context = module._prepare_llm_context(
        ticker="TSLA",
        existing_short=0,
        metrics=metrics,
        recommendation={
            "signal": "neutral",
            "confidence": 10,
            "reasoning": "Guardrail engaged",
            "constraints": {},
        },
    )

    assert context["guardrails"]["neutral_guardrail_active"] is True
    assert "raw probability below trigger" in context["guardrails"]["neutral_guardrail_reasons"]
    assert context["guardrails"]["neutral_guardrail_relaxed"] is False
    assert context["guardrails"]["neutral_guardrail_relaxed_reasons"] == []
