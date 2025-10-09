"""Train softmax weights for the regime meta-model using persisted features."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


REGIME_LABELS = ("crash", "rally", "consolidation")
FEATURE_COLUMNS = (
    "volatility_slope",
    "drawdown_depth",
    "return5_skew20",
    "momentum_20",
    "momentum_60",
    "vol_30",
    "vol_60",
    "vol_ratio",
)


@dataclass(slots=True)
class TrainingConfig:
    features_dir: Path
    output_path: Path
    tickers: tuple[str, ...]
    regimes: tuple[str, ...]
    learning_rate: float
    epochs: int
    l2_penalty: float


@dataclass(slots=True)
class CalibrationResult:
    classes: tuple[str, ...]
    feature_order: tuple[str, ...]
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    weights: dict[str, list[float]]
    bias: dict[str, float]
    accuracy: float
    validation_accuracy: float | None
    sample_count: int
    tickers: tuple[str, ...]
    regimes: tuple[str, ...]
    epochs: int
    learning_rate: float
    l2_penalty: float


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _prepare_tickers(features_dir: Path, tickers: Sequence[str] | None) -> tuple[str, ...]:
    if tickers:
        return tuple(t.strip().upper() for t in tickers if t.strip())

    candidates: list[str] = []
    for child in features_dir.iterdir():
        if child.is_dir():
            candidates.append(child.name.upper())
    if not candidates:
        raise FileNotFoundError(f"No ticker directories found under {features_dir}.")
    return tuple(sorted(candidates))


def _prepare_regimes(regimes: Sequence[str] | None) -> tuple[str, ...]:
    if regimes:
        prepared = []
        for regime in regimes:
            name = regime.strip().lower()
            if not name:
                continue
            if name not in REGIME_LABELS:
                raise ValueError(
                    f"Unsupported regime label '{regime}'. Expected one of {REGIME_LABELS}."
                )
            prepared.append(name)
        return tuple(prepared)
    return REGIME_LABELS


def _load_feature_table(path: Path, ticker: str, regime: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing regime feature table for {ticker} {regime}: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Feature table {path} is empty.")
    df["ticker"] = ticker
    df["regime_label"] = regime
    df["close"] = df["close"].astype(float)
    df["momentum_20"] = df["close"].pct_change(20)
    df["momentum_60"] = df["close"].pct_change(60)
    df["vol_ratio"] = df["vol_30"] / df["vol_60"].replace(0.0, np.nan) - 1.0
    df = df.dropna(subset=FEATURE_COLUMNS)
    return df


def _load_datasets(config: TrainingConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker in config.tickers:
        for regime in config.regimes:
            table_path = config.features_dir / ticker / regime / "regime_features.csv"
            try:
                frame = _load_feature_table(table_path, ticker, regime)
            except FileNotFoundError:
                continue
            except ValueError:
                continue
            frames.append(frame)

    if not frames:
        raise SystemExit(
            "No regime feature tables found for the requested tickers/regimes."
        )
    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.reset_index(drop=True)
    dataset = dataset.dropna(subset=FEATURE_COLUMNS)
    dataset = dataset[dataset["regime_label"].isin(config.regimes)]

    present = set(dataset["regime_label"].unique())
    missing = [regime for regime in config.regimes if regime not in present]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            "Insufficient coverage for regimes: "
            f"{joined}. Generate feature tables for the missing windows before training."
        )
    return dataset


def _train_validation_split(
    dataset: pd.DataFrame,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(dataset) < 40:
        return dataset, dataset.iloc[0:0]

    dataset = dataset.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_idx = int(len(dataset) * (1.0 - validation_ratio))
    train_df = dataset.iloc[:split_idx].reset_index(drop=True)
    val_df = dataset.iloc[split_idx:].reset_index(drop=True)
    return train_df, val_df


def _standardise(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds = np.where(stds < 1e-6, 1e-6, stds)
    normalised = (matrix - means) / stds
    return normalised, means, stds


def _encode_labels(labels: Iterable[str], classes: Sequence[str]) -> np.ndarray:
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    encoded = np.array([class_to_index[label] for label in labels], dtype=int)
    return encoded


def _train_softmax(
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    epochs: int,
    learning_rate: float,
    l2_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples, n_features = features.shape

    weights = np.zeros((n_classes, n_features), dtype=float)
    bias = np.zeros(n_classes, dtype=float)

    for epoch in range(epochs):
        logits = features @ weights.T + bias
        probabilities = _softmax(logits)

        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(n_samples), labels] = 1.0

        error = probabilities - one_hot
        grad_w = (error.T @ features) / n_samples + l2_penalty * weights
        grad_b = error.mean(axis=0)

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return weights, bias


def _evaluate(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> float:
    if features.size == 0:
        return float("nan")
    logits = features @ weights.T + bias
    probabilities = _softmax(logits)
    predictions = probabilities.argmax(axis=1)
    accuracy = float((predictions == labels).mean())
    return accuracy


def _build_calibration(
    config: TrainingConfig,
    dataset: pd.DataFrame,
) -> CalibrationResult:
    train_df, val_df = _train_validation_split(dataset)

    x_train = train_df.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    x_train, means, stds = _standardise(x_train)
    class_labels = tuple(cls for cls in REGIME_LABELS if cls in config.regimes)
    if len(class_labels) < 2:
        raise SystemExit("At least two regimes are required for calibration.")

    y_train = _encode_labels(train_df["regime_label"].tolist(), class_labels)

    weights, bias = _train_softmax(
        x_train,
        y_train,
        n_classes=len(class_labels),
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        l2_penalty=config.l2_penalty,
    )

    train_accuracy = _evaluate(x_train, y_train, weights, bias)

    val_accuracy: float | None = None
    if not val_df.empty:
        x_val = val_df.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
        x_val = (x_val - means) / stds
        y_val = _encode_labels(val_df["regime_label"].tolist(), class_labels)
        val_accuracy = _evaluate(x_val, y_val, weights, bias)

    feature_means = {feature: float(value) for feature, value in zip(FEATURE_COLUMNS, means)}
    feature_stds = {feature: float(value) for feature, value in zip(FEATURE_COLUMNS, stds)}

    weights_dict = {
        label: [float(weight) for weight in weights[idx]]
        for idx, label in enumerate(class_labels)
    }
    bias_dict = {
        label: float(bias[idx])
        for idx, label in enumerate(class_labels)
    }

    calibration = CalibrationResult(
        classes=class_labels,
        feature_order=FEATURE_COLUMNS,
        feature_means=feature_means,
        feature_stds=feature_stds,
        weights=weights_dict,
        bias=bias_dict,
        accuracy=train_accuracy,
        validation_accuracy=val_accuracy,
        sample_count=len(dataset),
        tickers=config.tickers,
        regimes=config.regimes,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        l2_penalty=config.l2_penalty,
    )
    return calibration


def _write_output(result: CalibrationResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classes": list(result.classes),
        "feature_order": list(result.feature_order),
        "feature_means": result.feature_means,
        "feature_stds": result.feature_stds,
        "weights": result.weights,
        "bias": result.bias,
        "metadata": {
            "sample_count": result.sample_count,
            "tickers": list(result.tickers),
            "regimes": list(result.regimes),
            "epochs": result.epochs,
            "learning_rate": result.learning_rate,
            "l2_penalty": result.l2_penalty,
            "train_accuracy": result.accuracy,
            "validation_accuracy": result.validation_accuracy,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train softmax weights for the regime meta-model using cached features.",
    )
    parser.add_argument(
        "--features-dir",
        default="log/regime_features",
        help="Directory containing per-ticker/per-regime feature tables.",
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated list of tickers to include. Default: all directories discovered.",
    )
    parser.add_argument(
        "--regimes",
        default="crash,rally,consolidation",
        help="Comma-separated regime labels to include (default: crash,rally,consolidation).",
    )
    parser.add_argument(
        "--output",
        default="configs/regime_meta_weights.json",
        help="Destination path for calibrated weights JSON.",
    )
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs (default: 400).")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Gradient descent learning rate (default: 0.05).",
    )
    parser.add_argument(
        "--l2-penalty",
        type=float,
        default=1e-4,
        help="L2 regularisation strength applied to weights (default: 1e-4).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    features_dir = Path(args.features_dir)
    output_path = Path(args.output)

    tickers = _prepare_tickers(features_dir, args.tickers.split(",") if args.tickers else None)
    regimes = _prepare_regimes(args.regimes.split(",") if args.regimes else None)

    config = TrainingConfig(
        features_dir=features_dir,
        output_path=output_path,
        tickers=tickers,
        regimes=regimes,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2_penalty=args.l2_penalty,
    )

    dataset = _load_datasets(config)
    calibration = _build_calibration(config, dataset)
    _write_output(calibration, output_path)

    metadata = calibration
    print(
        "Saved calibrated weights to"
        f" {output_path} (samples={metadata.sample_count}, train_acc={metadata.accuracy:.3f},"
        + (
            f" val_acc={metadata.validation_accuracy:.3f})"
            if metadata.validation_accuracy is not None
            else ")"
        )
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
