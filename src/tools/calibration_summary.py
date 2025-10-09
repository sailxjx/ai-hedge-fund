from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class CalibrationMetrics:
    agent: str
    ticker: str
    window: str
    sample_count: int
    pre_auc: float
    post_auc: float
    pre_brier: float
    post_brier: float
    calibrated: bool


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def _compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(float)
    if labels.size == 0:
        return math.nan
    pos_count = float(labels.sum())
    neg_count = labels.size - pos_count
    if pos_count == 0 or neg_count == 0:
        return math.nan

    ranks = np.argsort(np.argsort(scores)) + 1
    pos_ranks = ranks[labels == 1]
    numerator = float(pos_ranks.sum() - pos_count * (pos_count + 1) / 2.0)
    denominator = pos_count * neg_count
    return numerator / denominator if denominator else math.nan


def _compute_brier(labels: np.ndarray, probs: np.ndarray) -> float:
    if labels.size == 0:
        return math.nan
    return float(np.mean((probs - labels) ** 2))


def _load_calibration(calibration_path: Path) -> dict[str, object]:
    if calibration_path.exists():
        try:
            return json.loads(calibration_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def _get_params(payload: dict[str, object], agent: str, ticker: str) -> dict[str, float] | None:
    agent_payload = payload.get(agent)
    if not isinstance(agent_payload, dict):
        return None
    candidate = agent_payload.get(ticker)
    if isinstance(candidate, dict) and {"a", "b"} <= candidate.keys():
        try:
            return {"a": float(candidate["a"]), "b": float(candidate["b"])}
        except (TypeError, ValueError):
            return None
    default_candidate = agent_payload.get("default")
    if isinstance(default_candidate, dict) and {"a", "b"} <= default_candidate.keys():
        try:
            return {"a": float(default_candidate["a"]), "b": float(default_candidate["b"])}
        except (TypeError, ValueError):
            return None
    return None


def _apply_platt_vector(scores: np.ndarray, params: dict[str, float] | None) -> np.ndarray:
    if not params:
        return _sigmoid(scores)
    a = float(params.get("a", 0.0))
    b = float(params.get("b", 0.0))
    z = np.clip(a * scores + b, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _iter_ticker_dirs(root: Path) -> Iterable[Path]:
    for child in sorted(root.iterdir()):
        if child.is_dir():
            yield child


def _iter_window_dirs(ticker_dir: Path) -> Iterable[Path]:
    for child in sorted(ticker_dir.iterdir()):
        if child.is_dir():
            yield child


def _growth_metrics_for_window(
    ticker: str,
    window_dir: Path,
    params: dict[str, float] | None,
) -> CalibrationMetrics | None:
    dataset_path = window_dir / "growth_momentum_dataset.csv"
    diagnostics_path = window_dir / "growth_momentum_diagnostics.json"
    if not dataset_path.exists() or not diagnostics_path.exists():
        return None

    try:
        diagnostics = json.loads(diagnostics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    weights = diagnostics.get("weights")
    if not isinstance(weights, list) or not weights:
        return None

    df = pd.read_csv(dataset_path)
    if df.empty or "label" not in df.columns:
        return None

    feature_cols = [col for col in df.columns if col not in {"date", "label"}]
    if not feature_cols:
        return None

    features = df[feature_cols].to_numpy(dtype=float)
    intercept = np.ones((features.shape[0], 1))
    X = np.hstack([intercept, features])
    weight_vec = np.asarray(weights, dtype=float)
    if weight_vec.shape[0] != X.shape[1]:
        return None

    logits = X @ weight_vec
    labels = df["label"].to_numpy(dtype=float)

    raw_probs = _sigmoid(logits)
    calibrated_probs = _apply_platt_vector(logits, params)

    pre_auc = _compute_auc(labels, raw_probs)
    post_auc = _compute_auc(labels, calibrated_probs)
    pre_brier = _compute_brier(labels, raw_probs)
    post_brier = _compute_brier(labels, calibrated_probs)

    return CalibrationMetrics(
        agent="growth_momentum",
        ticker=ticker,
        window=window_dir.name,
        sample_count=int(len(labels)),
        pre_auc=float(pre_auc) if not math.isnan(pre_auc) else math.nan,
        post_auc=float(post_auc) if not math.isnan(post_auc) else math.nan,
        pre_brier=float(pre_brier) if not math.isnan(pre_brier) else math.nan,
        post_brier=float(post_brier) if not math.isnan(post_brier) else math.nan,
        calibrated=bool(params),
    )


def _erf(values: np.ndarray) -> np.ndarray:
    return np.vectorize(math.erf)(values)


def _mean_reversion_metrics_for_window(
    ticker: str,
    window_dir: Path,
    params: dict[str, float] | None,
) -> CalibrationMetrics | None:
    dataset_path = window_dir / "mean_reversion_dataset.csv"
    if not dataset_path.exists():
        return None

    df = pd.read_csv(dataset_path)
    if df.empty or not {"z_score", "up_move"}.issubset(df.columns):
        return None

    z_scores = df["z_score"].to_numpy(dtype=float)
    labels = df["up_move"].to_numpy(dtype=float)
    if labels.size == 0:
        return None

    raw_probs = 0.5 * (1.0 + _erf(-z_scores / math.sqrt(2.0)))
    calibrated_probs = _apply_platt_vector(z_scores, params)

    pre_auc = _compute_auc(labels, raw_probs)
    post_auc = _compute_auc(labels, calibrated_probs)
    pre_brier = _compute_brier(labels, raw_probs)
    post_brier = _compute_brier(labels, calibrated_probs)

    return CalibrationMetrics(
        agent="stat_mean_reversion",
        ticker=ticker,
        window=window_dir.name,
        sample_count=int(len(labels)),
        pre_auc=float(pre_auc) if not math.isnan(pre_auc) else math.nan,
        post_auc=float(post_auc) if not math.isnan(post_auc) else math.nan,
        pre_brier=float(pre_brier) if not math.isnan(pre_brier) else math.nan,
        post_brier=float(post_brier) if not math.isnan(post_brier) else math.nan,
        calibrated=bool(params),
    )


def _format_float(value: float, precision: int = 3) -> str:
    if value != value or math.isinf(value):
        return "na"
    return f"{value:.{precision}f}"


def _format_delta(before: float, after: float, precision: int = 3) -> str:
    if any(math.isnan(v) or math.isinf(v) for v in (before, after)):
        return "na"
    delta = after - before
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.{precision}f}"


def _weighted_average(values: list[float], weights: list[int]) -> float:
    numerator = 0.0
    denominator = 0.0
    for value, weight in zip(values, weights):
        if math.isnan(value) or weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    if denominator == 0:
        return math.nan
    return numerator / denominator


def _render_table(agent: str, rows: list[CalibrationMetrics]) -> str:
    header = "| Ticker | Window | Pre AUC | Post AUC | Δ AUC | Pre Brier | Post Brier | Δ Brier |"
    separator = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [f"## {agent.replace('_', ' ').title()}", header, separator]
    for row in sorted(rows, key=lambda r: (r.ticker, r.window)):
        delta_auc = _format_delta(row.pre_auc, row.post_auc)
        delta_brier = _format_delta(row.pre_brier, row.post_brier)
        lines.append(
            "| {ticker} | {window} | {pre_auc} | {post_auc} | {delta_auc} | {pre_brier} | {post_brier} | {delta_brier} |".format(
                ticker=row.ticker,
                window=row.window,
                pre_auc=_format_float(row.pre_auc),
                post_auc=_format_float(row.post_auc),
                delta_auc=delta_auc,
                pre_brier=_format_float(row.pre_brier),
                post_brier=_format_float(row.post_brier),
                delta_brier=delta_brier,
            )
        )
    return "\n".join(lines)


def _render_aggregates(agent: str, rows: list[CalibrationMetrics]) -> str:
    if not rows:
        return ""
    weights = [row.sample_count for row in rows]
    pre_auc_avg = _weighted_average([row.pre_auc for row in rows], weights)
    post_auc_avg = _weighted_average([row.post_auc for row in rows], weights)
    pre_brier_avg = _weighted_average([row.pre_brier for row in rows], weights)
    post_brier_avg = _weighted_average([row.post_brier for row in rows], weights)

    delta_auc = _format_delta(pre_auc_avg, post_auc_avg)
    delta_brier = _format_delta(pre_brier_avg, post_brier_avg)

    lines = ["### Aggregated Impact"]
    lines.append(
        "- Weighted sample count: {count}".format(count=sum(weights))
    )
    lines.append(
        "- AUC: pre {pre} → post {post} ({delta})".format(
            pre=_format_float(pre_auc_avg),
            post=_format_float(post_auc_avg),
            delta=delta_auc,
        )
    )
    lines.append(
        "- Brier: pre {pre} → post {post} ({delta})".format(
            pre=_format_float(pre_brier_avg),
            post=_format_float(post_brier_avg),
            delta=delta_brier,
        )
    )
    return "\n".join(lines)


def build_calibration_metrics(
    diagnostics_root: Path,
    calibration_path: Path,
) -> list[CalibrationMetrics]:
    calibration_payload = _load_calibration(calibration_path)
    metrics: list[CalibrationMetrics] = []

    for ticker_dir in _iter_ticker_dirs(diagnostics_root):
        ticker = ticker_dir.name
        growth_params = _get_params(calibration_payload, "growth_momentum", ticker)
        reversion_params = _get_params(calibration_payload, "stat_mean_reversion", ticker)

        for window_dir in _iter_window_dirs(ticker_dir):
            growth_metrics = _growth_metrics_for_window(ticker, window_dir, growth_params)
            if growth_metrics:
                metrics.append(growth_metrics)

            mean_metrics = _mean_reversion_metrics_for_window(ticker, window_dir, reversion_params)
            if mean_metrics:
                metrics.append(mean_metrics)

    return metrics


def render_markdown(
    metrics: list[CalibrationMetrics],
    *,
    generated_at: datetime,
) -> str:
    lines = ["# Calibration Impact Summary"]
    lines.append(f"Generated at {generated_at.isoformat(timespec='seconds')}")
    if not metrics:
        lines.append("\nNo diagnostics available to summarise.")
        return "\n".join(lines)

    by_agent: dict[str, list[CalibrationMetrics]] = {}
    for entry in metrics:
        by_agent.setdefault(entry.agent, []).append(entry)

    key_findings: list[str] = []
    for agent, rows in by_agent.items():
        weights = [row.sample_count for row in rows]
        pre_brier_avg = _weighted_average([row.pre_brier for row in rows], weights)
        post_brier_avg = _weighted_average([row.post_brier for row in rows], weights)
        pre_auc_avg = _weighted_average([row.pre_auc for row in rows], weights)
        post_auc_avg = _weighted_average([row.post_auc for row in rows], weights)
        delta_brier = post_brier_avg - pre_brier_avg if not any(math.isnan(v) for v in (pre_brier_avg, post_brier_avg)) else math.nan
        delta_auc = post_auc_avg - pre_auc_avg if not any(math.isnan(v) for v in (pre_auc_avg, post_auc_avg)) else math.nan
        key_findings.append(
            "`{agent}` AUC Δ {auc_delta}, Brier Δ {brier_delta}".format(
                agent=agent,
                auc_delta=_format_delta(pre_auc_avg, post_auc_avg),
                brier_delta=_format_delta(pre_brier_avg, post_brier_avg),
            )
        )
    lines.append("\n## Key Findings")
    for finding in key_findings:
        lines.append(f"- {finding}")

    for agent, rows in sorted(by_agent.items()):
        lines.append("")
        lines.append(_render_table(agent, rows))
        lines.append("")
        aggregate_block = _render_aggregates(agent, rows)
        if aggregate_block:
            lines.append(aggregate_block)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise calibration impact on ML diagnostics.")
    parser.add_argument(
        "--diagnostics-root",
        type=Path,
        default=Path("log/ml_diagnostics"),
        help="Root directory containing per-ticker diagnostics exports.",
    )
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=Path("configs/probability_calibration.json"),
        help="Calibration parameter JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("log/ml_diagnostics/calibration_summary.md"),
        help="Destination markdown file for the summary.",
    )
    args = parser.parse_args()

    metrics = build_calibration_metrics(args.diagnostics_root, args.calibration_config)
    generated_at = datetime.now(UTC)
    markdown = render_markdown(metrics, generated_at=generated_at)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
