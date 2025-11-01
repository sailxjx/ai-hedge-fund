from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DATA_ROOT = Path("log/ml_diagnostics")
DEFAULT_OUTPUT = Path("configs/probability_calibration.json")


def _iter_window_dirs(ticker_dir: Path) -> Iterable[Path]:
    for child in sorted(ticker_dir.iterdir()):
        if child.is_dir():
            yield child


def _gather_growth_samples(ticker_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for window_dir in _iter_window_dirs(ticker_dir):
        dataset_path = window_dir / "growth_momentum_dataset.csv"
        diagnostics_path = window_dir / "growth_momentum_diagnostics.json"
        if not dataset_path.exists() or not diagnostics_path.exists():
            continue

        df = pd.read_csv(dataset_path)
        if df.empty or "label" not in df.columns:
            continue

        try:
            diagnostics = json.loads(diagnostics_path.read_text())
        except json.JSONDecodeError:
            continue

        weights = diagnostics.get("weights")
        if not isinstance(weights, list) or not weights:
            continue

        feature_cols = [col for col in df.columns if col not in {"date", "label"}]
        if not feature_cols:
            continue

        features = df[feature_cols].to_numpy(dtype=float)
        intercept = np.ones((features.shape[0], 1))
        X = np.hstack([intercept, features])
        weight_vec = np.asarray(weights, dtype=float)
        if weight_vec.shape[0] != X.shape[1]:
            continue

        logits = X @ weight_vec
        label_array = df["label"].to_numpy(dtype=float)

        mask = np.isfinite(logits) & np.isfinite(label_array)
        if not np.any(mask):
            continue

        scores.append(logits[mask])
        labels.append(label_array[mask])

    if not scores:
        return np.array([], dtype=float), np.array([], dtype=float)

    return np.concatenate(scores), np.concatenate(labels)


def _gather_mean_reversion_samples(ticker_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for window_dir in _iter_window_dirs(ticker_dir):
        dataset_path = window_dir / "mean_reversion_dataset.csv"
        if not dataset_path.exists():
            continue

        df = pd.read_csv(dataset_path)
        if df.empty:
            continue
        if not {"z_score", "up_move"}.issubset(df.columns):
            continue

        z = df["z_score"].to_numpy(dtype=float)
        y = df["up_move"].to_numpy(dtype=float)

        mask = np.isfinite(z) & np.isfinite(y)
        if not np.any(mask):
            continue

        scores.append(z[mask])
        labels.append(y[mask])

    if not scores:
        return np.array([], dtype=float), np.array([], dtype=float)

    return np.concatenate(scores), np.concatenate(labels)


def _fit_platt_scaling(scores: np.ndarray, labels: np.ndarray) -> dict[str, float] | None:
    if scores.size == 0 or labels.size == 0:
        return None
    labels = labels.astype(float)
    scores = scores.astype(float)

    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return None

    mean = float(scores.mean())
    std = float(scores.std(ddof=0))
    if std == 0.0:
        return None

    norm_scores = (scores - mean) / std

    a_norm = 0.0
    b = 0.0
    lr = 0.01
    l2 = 1e-4
    epochs = 6000

    for _ in range(epochs):
        z = np.clip(a_norm * norm_scores + b, -50.0, 50.0)
        preds = 1.0 / (1.0 + np.exp(-z))
        error = preds - labels

        grad_a = float((error * norm_scores).mean() + l2 * a_norm)
        grad_b = float(error.mean() + l2 * b)

        a_norm -= lr * grad_a
        b -= lr * grad_b

    a = a_norm / std
    b_adj = b - (a_norm * mean / std)

    calibrated_logits = np.clip(a * scores + b_adj, -50.0, 50.0)
    calibrated_probs = 1.0 / (1.0 + np.exp(-calibrated_logits))

    eps = 1e-9
    log_loss = -float((labels * np.log(np.clip(calibrated_probs, eps, 1.0 - eps)) + (1.0 - labels) * np.log(np.clip(1.0 - calibrated_probs, eps, 1.0 - eps))).mean())
    brier = float(np.mean((calibrated_probs - labels) ** 2))

    return {
        "a": float(a),
        "b": float(b_adj),
        "sample_count": int(scores.size),
        "positive_rate": float(labels.mean()),
        "brier_score": brier,
        "log_loss": log_loss,
    }


def _round_payload(payload: dict[str, float]) -> dict[str, float]:
    rounded: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, float):
            rounded[key] = round(float(value), 6)
        else:
            rounded[key] = value
    return rounded


def build_calibration_payload(data_root: Path) -> dict[str, object]:
    if not data_root.exists():
        raise FileNotFoundError(f"Diagnostics root not found: {data_root}")

    payload: dict[str, object] = {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "data_root": str(data_root),
            "method": "platt_scaling",
        },
        "growth_momentum": {},
        "stat_mean_reversion": {},
    }

    growth_scores_all: list[np.ndarray] = []
    growth_labels_all: list[np.ndarray] = []

    mean_scores_all: list[np.ndarray] = []
    mean_labels_all: list[np.ndarray] = []

    for ticker_dir in sorted(data_root.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name

        gm_scores, gm_labels = _gather_growth_samples(ticker_dir)
        params = _fit_platt_scaling(gm_scores, gm_labels)
        if params:
            payload["growth_momentum"][ticker] = _round_payload(params)
            growth_scores_all.append(gm_scores)
            growth_labels_all.append(gm_labels)

        mr_scores, mr_labels = _gather_mean_reversion_samples(ticker_dir)
        params = _fit_platt_scaling(mr_scores, mr_labels)
        if params:
            payload["stat_mean_reversion"][ticker] = _round_payload(params)
            mean_scores_all.append(mr_scores)
            mean_labels_all.append(mr_labels)

    if growth_scores_all:
        stacked_scores = np.concatenate(growth_scores_all)
        stacked_labels = np.concatenate(growth_labels_all)
        params = _fit_platt_scaling(stacked_scores, stacked_labels)
        if params:
            payload["growth_momentum"]["default"] = _round_payload(params)

    if mean_scores_all:
        stacked_scores = np.concatenate(mean_scores_all)
        stacked_labels = np.concatenate(mean_labels_all)
        params = _fit_platt_scaling(stacked_scores, stacked_labels)
        if params:
            payload["stat_mean_reversion"]["default"] = _round_payload(params)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Platt calibration parameters for analyst probabilities.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Root directory containing ml_diagnostics exports (default: log/ml_diagnostics)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination path for calibration JSON (default: configs/probability_calibration.json)",
    )
    args = parser.parse_args()

    payload = build_calibration_payload(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
