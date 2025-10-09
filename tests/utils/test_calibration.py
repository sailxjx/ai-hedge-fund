import json
import math
from pathlib import Path

import pytest

from src.utils import calibration


def _write_calibration(tmp_path: Path) -> Path:
    payload = {
        "metadata": {},
        "growth_momentum": {
            "TSLA": {"a": 1.0, "b": -0.1},
            "default": {"a": 2.0, "b": 0.0},
        },
        "stat_mean_reversion": {},
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_get_platt_parameters_prefers_specific(tmp_path, monkeypatch):
    path = _write_calibration(tmp_path)

    monkeypatch.setattr(calibration, "CALIBRATION_PATH", path)
    calibration.reset_probability_calibration_cache()

    params = calibration.get_platt_parameters("growth_momentum", "TSLA")
    assert params == {"a": 1.0, "b": -0.1}

    fallback = calibration.get_platt_parameters("growth_momentum", "MSFT")
    assert fallback == {"a": 2.0, "b": 0.0}

    missing = calibration.get_platt_parameters("unknown_agent", "TSLA")
    assert missing is None


@pytest.mark.parametrize(
    "score,a,b",
    [
        (0.0, 1.0, 0.0),
        (0.2, 10.0, -1.0),
        (-0.7, 2.0, 0.3),
    ],
)
def test_apply_platt_matches_sigmoid(score, a, b):
    prob = calibration.apply_platt(score, {"a": a, "b": b})
    expected = 1.0 / (1.0 + math.exp(-(a * score + b)))
    assert prob == pytest.approx(expected, rel=1e-6)
