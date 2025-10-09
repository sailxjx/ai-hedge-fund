"""CLI utility to summarise backtest logs and surface key risk metrics."""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    if sys.path and sys.path[0] == current_dir:
        sys.path.pop(0)
        sys.path.append(current_dir)

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import argparse
import csv
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
PORTFOLIO_RETURN_RE = re.compile(r"Portfolio Return:\s*([+-]?\d+\.\d+)%")
SHARPE_RE = re.compile(r"Sharpe(?: Ratio)?:\s*([+-]?\d+\.\d+)")
SORTINO_RE = re.compile(r"Sortino(?: Ratio)?:\s*([+-]?\d+\.\d+)")
INFORMATION_RATIO_RE = re.compile(r"Information Ratio:\s*([+-]?\d+\.\d+)")
MAX_DRAWDOWN_RE = re.compile(r"Max Drawdown:\s*([+-]?\d+\.\d+)%")
BENCHMARK_RETURN_RE = re.compile(r"Benchmark Return:\s*([+-]?\d+\.\d+)%")
TURNOVER_RATE_RE = re.compile(r"Turnover Rate:\s*([+-]?\d+\.\d+)%")
FORCE_COVER_RE = re.compile(r"force_cover", re.IGNORECASE)
TARGET_LONG_RE = re.compile(r"target_long_shares", re.IGNORECASE)
PROB_UP_RE = re.compile(r"prob_up\s*[:=]\s*([01]?\.\d+)", re.IGNORECASE)
NEG_RETURN_RE = re.compile(r"Total Return:\s*([-+]\d+\.\d+)%", re.IGNORECASE)


@dataclass
class BacktestMetrics:
    label: str
    path: str
    portfolio_return_pct: float | None = None
    turnover_rate_pct: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    information_ratio: float | None = None
    max_drawdown_pct: float | None = None
    benchmark_return_pct: float | None = None
    force_cover_events: int = 0
    target_long_events: int = 0
    prob_up_calls: int = 0

    def to_row(self) -> dict[str, str]:
        payload = asdict(self)
        return {k: ("" if v is None else v) for k, v in payload.items()}


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _load_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:  # pragma: no cover - file errors bubble up to CLI
        raise SystemExit(f"Failed to read log '{path}': {exc}") from exc


def _parse_metric(regex: re.Pattern[str], text: str) -> float | None:
    matches = regex.findall(text)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[-1]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _count_pattern(regex: re.Pattern[str], text: str) -> int:
    return len(regex.findall(text))


def _parse_log(label: str, path: Path, override_path: Path | None = None) -> BacktestMetrics:
    raw = _load_log(path)
    clean = _strip_ansi(raw)

    metrics = BacktestMetrics(label=label, path=str(path))
    metrics.portfolio_return_pct = _parse_metric(PORTFOLIO_RETURN_RE, clean)
    metrics.turnover_rate_pct = _parse_metric(TURNOVER_RATE_RE, clean)
    metrics.sharpe_ratio = _parse_metric(SHARPE_RE, clean)
    metrics.sortino_ratio = _parse_metric(SORTINO_RE, clean)
    metrics.information_ratio = _parse_metric(INFORMATION_RATIO_RE, clean)
    metrics.max_drawdown_pct = _parse_metric(MAX_DRAWDOWN_RE, clean)
    metrics.benchmark_return_pct = _parse_metric(BENCHMARK_RETURN_RE, clean)
    metrics.force_cover_events = _count_pattern(FORCE_COVER_RE, clean)
    metrics.target_long_events = _count_pattern(TARGET_LONG_RE, clean)

    prob_calls = _count_pattern(PROB_UP_RE, clean)
    negative_return_mentions = _count_pattern(NEG_RETURN_RE, clean)
    metrics.prob_up_calls = max(0, prob_calls - negative_return_mentions)

    if override_path is not None:
        force_covers, target_longs = _parse_override_metrics(override_path)
        metrics.force_cover_events = max(metrics.force_cover_events, force_covers)
        metrics.target_long_events = max(metrics.target_long_events, target_longs)
    return metrics


def _parse_override_metrics(path: Path) -> tuple[int, int]:
    raw_lines = _load_override_lines(path)
    force_covers = 0
    target_longs = 0

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue

        overrides = payload.get("risk_snapshot", {}).get("overrides")
        if not isinstance(overrides, dict):
            continue

        force_cover_qty = _coerce_int(overrides.get("force_cover_qty"))
        if force_cover_qty is not None and force_cover_qty > 0:
            force_covers += 1

        target_long_value = overrides.get("target_long_shares")
        if target_long_value is not None and _coerce_int(target_long_value) is not None:
            target_longs += 1

    return force_covers, target_longs


def _load_override_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:  # pragma: no cover - surfaced via CLI
        raise SystemExit(f"Failed to read override log '{path}': {exc}") from exc


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _parse_log_argument(log_arg: str) -> tuple[str, Path]:
    if "=" in log_arg:
        label, raw_path = log_arg.split("=", 1)
        label = label.strip()
    else:
        raw_path = log_arg
        label = Path(raw_path).stem
    if not label:
        label = Path(raw_path).stem or "run"
    return label, Path(raw_path)


def _write_csv(metrics: Iterable[BacktestMetrics], output: Path) -> None:
    rows = [m.to_row() for m in metrics]
    if not rows:
        raise SystemExit("No metrics computed; nothing to write.")
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise backtest log metrics.")
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="List of logs to evaluate; optionally prefix with label=path",
    )
    parser.add_argument(
        "--overrides",
        nargs="+",
        help="Optional risk override logs keyed by label (label=path)",
    )
    parser.add_argument("--output", help="Optional CSV output path.")
    parser.add_argument("--json", help="Optional JSON output path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metrics: list[BacktestMetrics] = []

    override_map: dict[str, Path] = {}
    if args.overrides:
        for override_arg in args.overrides:
            override_label, override_path = _parse_log_argument(override_arg)
            override_map[override_label] = override_path

    for log_arg in args.logs:
        label, path = _parse_log_argument(log_arg)
        metrics.append(_parse_log(label, path, override_map.get(label)))

    if args.output:
        _write_csv(metrics, Path(args.output))
        print(f"Wrote CSV summary to {args.output}")

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_payload = [m.to_row() for m in metrics]
        json_path.write_text(json.dumps(json_payload, indent=2))
        print(f"Wrote JSON summary to {args.json}")

    if not args.output and not args.json:
        for entry in metrics:
            print(json.dumps(entry.to_row(), indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
