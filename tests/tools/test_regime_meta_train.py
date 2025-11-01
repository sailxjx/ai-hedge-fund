from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.tools import regime_meta_train as trainer


def _create_feature_table(path: Path, seed: int, drawdown: float) -> None:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    close = 100 + rng.normal(loc=0.0, scale=1.5, size=len(dates)).cumsum()

    vol_30 = np.full(len(dates), 0.35 + 0.05 * rng.random())
    vol_60 = vol_30 - 0.05 * rng.random()

    frame = pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "return_1d": rng.normal(0.0, 0.02, size=len(dates)),
            "vol_30": vol_30,
            "vol_60": vol_60,
            "volatility_slope": vol_30 - vol_60,
            "drawdown_depth": drawdown,
            "return5_skew20": rng.normal(0.0, 0.2, size=len(dates)),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_build_calibration_from_feature_tables(tmp_path):
    features_dir = tmp_path / "regime_features"
    ticker = "TEST"
    for seed, regime, drawdown in [
        (1, "crash", -0.35),
        (2, "rally", -0.05),
        (3, "consolidation", -0.12),
    ]:
        table_path = features_dir / ticker / regime / "regime_features.csv"
        _create_feature_table(table_path, seed=seed, drawdown=drawdown)

    config = trainer.TrainingConfig(
        features_dir=features_dir,
        output_path=tmp_path / "weights.json",
        tickers=(ticker,),
        regimes=("crash", "rally", "consolidation"),
        learning_rate=0.05,
        epochs=80,
        l2_penalty=1e-4,
    )

    dataset = trainer._load_datasets(config)
    assert not dataset.empty
    assert set(dataset["regime_label"].unique()) == {"crash", "rally", "consolidation"}

    calibration = trainer._build_calibration(config, dataset)
    assert calibration.classes == ("crash", "rally", "consolidation")
    assert calibration.sample_count == len(dataset)
    assert calibration.feature_order == trainer.FEATURE_COLUMNS
    assert calibration.accuracy > 0.0

    trainer._write_output(calibration, config.output_path)
    payload = json.loads(config.output_path.read_text())
    assert set(payload["classes"]) == {"crash", "rally", "consolidation"}
    assert payload["feature_order"] == list(trainer.FEATURE_COLUMNS)
    assert payload["metadata"]["sample_count"] == len(dataset)
