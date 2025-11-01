import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.agents.growth_momentum import growth_momentum_agent
from src.agents.stat_mean_reversion import stat_mean_reversion_agent
from src.data.models import Price
from src.utils import calibration


def _build_prices(length: int, start: float = 100.0, drift: float = 0.5) -> list[Price]:
    base = datetime(2023, 1, 1)
    prices: list[Price] = []
    level = start
    for idx in range(length):
        if idx % 9 == 0:
            level -= drift * 3
        else:
            level += drift
        level = max(level, 1.0)
        high = level * 1.002
        low = level * 0.998
        prices.append(
            Price(
                open=(high + low) / 2,
                close=level,
                high=high,
                low=low,
                volume=1_000_000,
                time=(base + timedelta(days=idx)).strftime("%Y-%m-%dT00:00:00Z"),
            )
        )
    return prices


def _base_state() -> dict:
    return {
        "messages": [],
        "data": {
            "tickers": ["TSLA"],
            "start_date": "2024-01-01",
            "end_date": "2024-04-30",
            "analyst_signals": {},
        },
        "metadata": {"show_reasoning": False},
    }


def test_growth_momentum_uses_calibration(tmp_path, monkeypatch):
    calibration_path = tmp_path / "calibration.json"
    calibration_payload = {
        "metadata": {},
        "growth_momentum": {
            "TSLA": {"a": 10.0, "b": -1.0},
        },
        "stat_mean_reversion": {},
    }
    calibration_path.write_text(json.dumps(calibration_payload), encoding="utf-8")

    monkeypatch.setattr(calibration, "CALIBRATION_PATH", calibration_path)
    calibration.reset_probability_calibration_cache()

    price_series = _build_prices(200)
    monkeypatch.setattr(
        "src.agents.growth_momentum.get_prices",
        lambda *_, **__: price_series,
    )

    monkeypatch.setattr(
        "src.agents.growth_momentum._logistic_regression",
        lambda X, y, **kwargs: np.array([0.2] + [0.0] * (X.shape[1] - 1)),
    )

    state = _base_state()
    growth_momentum_agent(state)

    payload = state["data"]["analyst_signals"]["growth_momentum_agent"]["TSLA"]
    indicators = payload["indicators"]

    assert pytest.approx(indicators["raw_prob_up"], rel=1e-6) == 1.0 / (1.0 + np.exp(-0.2))
    assert indicators["calibration_applied"] is True

    expected_prob = calibration.apply_platt(
        indicators["logit"],
        {"a": 10.0, "b": -1.0},
    )
    assert indicators["prob_up"] == pytest.approx(expected_prob, rel=1e-6)


def test_mean_reversion_uses_calibration(tmp_path, monkeypatch):
    calibration_path = tmp_path / "calibration.json"
    calibration_payload = {
        "metadata": {},
        "growth_momentum": {},
        "stat_mean_reversion": {
            "TSLA": {"a": -4.0, "b": 0.5},
        },
    }
    calibration_path.write_text(json.dumps(calibration_payload), encoding="utf-8")

    monkeypatch.setattr(calibration, "CALIBRATION_PATH", calibration_path)
    calibration.reset_probability_calibration_cache()

    price_series = _build_prices(140, drift=1.0)

    async def fake_get_prices_async(*_, **__):
        return price_series

    monkeypatch.setattr("src.agents.stat_mean_reversion.get_prices_async", fake_get_prices_async)

    state = _base_state()
    stat_mean_reversion_agent(state)

    payload = state["data"]["analyst_signals"]["stat_mean_reversion_agent"]["TSLA"]
    indicators = payload["indicators"]

    assert indicators["calibration_applied"] is True
    calibrated = indicators["prob_revert_up"]
    raw = indicators["raw_prob_revert_up"]

    assert calibrated == pytest.approx(
        calibration.apply_platt(indicators["z_score"], {"a": -4.0, "b": 0.5}),
        rel=1e-6,
    )
    assert calibrated != pytest.approx(raw, rel=1e-6)
