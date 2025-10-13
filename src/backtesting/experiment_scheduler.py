"""Batch scheduler for orchestrating multi-regime backtests.

This module coordinates executing the CLI backtester across predefined
windows, captures the resulting metrics, and appends structured entries to
``log/backtest.csv`` including deltas versus an optional baseline run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from src.tools.llm_backtest_digest import BacktestDigest, parse_backtest_digest

DEFAULT_TIMEOUT_SECONDS = 3600
BACKTESTER_ENTRYPOINT = Path("src/backtester.py")
ASYNC_LEDGER_COLUMNS = [
    "timing_summary_path",
    "async_agent_invoke_total_seconds",
    "async_per_agent_total_seconds",
    "async_concurrency_ratio",
    "async_semaphore_utilization",
    "async_slowest_agent",
    "async_slowest_agent_avg_seconds",
]

CSV_HEADER = [
    "timestamp",
    "label",
    "tickers",
    "analysts",
    "start_date",
    "end_date",
    "prompt_revision",
    "portfolio_return_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown_pct",
    "information_ratio",
    "benchmark_return_pct",
    "turnover_rate_pct",
    "hit_rate",
    "log_path",
    "baseline_label",
    "delta_portfolio_return_pct",
    "delta_sharpe_ratio",
    "delta_max_drawdown_pct",
] + ASYNC_LEDGER_COLUMNS


class ExperimentRunError(RuntimeError):
    """Raised when a scheduled experiment fails to execute successfully."""


@dataclass(slots=True)
class Experiment:
    """Configuration for a single backtest experiment."""

    label: str
    tickers: list[str]
    start_date: str
    end_date: str
    log_path: Path
    prompt_revision: str = "baseline"
    baseline_label: str | None = None
    analysts_all: bool = True
    analysts: list[str] | None = None
    extra_args: list[str] | None = None
    note: str | None = None
    timeout_seconds: int | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object], *, now: datetime | None = None) -> "Experiment":
        """Build an experiment from a JSON/YAML mapping."""

        def _as_str_list(value: object | None) -> list[str] | None:
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                result = [str(item).strip() for item in value if str(item).strip()]
                return result or None
            if isinstance(value, str):
                candidates = [part.strip() for part in value.split(",") if part.strip()]
                return candidates or None
            raise ValueError("analysts must be list or comma-separated string")

        def _as_ticker_list(value: object | None) -> list[str]:
            if value is None:
                raise ValueError("tickers are required for each experiment")
            if isinstance(value, (list, tuple)):
                tickers = [str(item).strip().upper() for item in value if str(item).strip()]
            elif isinstance(value, str):
                tickers = [part.strip().upper() for part in value.split(",") if part.strip()]
            else:
                raise ValueError("tickers must be list or comma-separated string")
            if not tickers:
                raise ValueError("tickers cannot be empty")
            return tickers

        required_fields = {"label", "start_date", "end_date"}
        missing = [field for field in required_fields if field not in payload or not str(payload[field]).strip()]
        if missing:
            raise ValueError(f"Experiment configuration missing fields: {', '.join(missing)}")

        label = str(payload["label"]).strip()
        tickers = _as_ticker_list(payload.get("tickers"))
        start_date = str(payload["start_date"]).strip()
        end_date = str(payload["end_date"]).strip()

        log_value = payload.get("log_path") or payload.get("log_file")
        if log_value:
            log_path = Path(str(log_value)).expanduser()
        else:
            log_path = default_log_path(label, start_date, end_date, now=now)

        prompt_revision = str(payload.get("prompt_revision", "baseline")).strip()
        baseline_label = str(payload["baseline_label"]).strip() if payload.get("baseline_label") else None
        analysts_all = bool(payload.get("analysts_all", True))
        analysts = _as_str_list(payload.get("analysts")) if not analysts_all else None
        extra_args = _as_str_list(payload.get("extra_args"))
        note = str(payload["note"]).strip() if payload.get("note") else None
        timeout_seconds = None
        if payload.get("timeout_seconds") is not None:
            timeout_seconds = int(payload["timeout_seconds"])

        return cls(
            label=label,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            log_path=log_path,
            prompt_revision=prompt_revision,
            baseline_label=baseline_label,
            analysts_all=analysts_all,
            analysts=analysts,
            extra_args=extra_args,
            note=note,
            timeout_seconds=timeout_seconds,
        )


@dataclass(slots=True)
class Metrics:
    """Subset of backtest metrics tracked in the experiment ledger."""

    portfolio_return_pct: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    max_drawdown_pct: float | None
    information_ratio: float | None
    benchmark_return_pct: float | None
    turnover_rate_pct: float | None


RunCommand = Callable[[Sequence[str], int], None]
ParseDigest = Callable[[Path], BacktestDigest]


def _timing_slug(experiment: Experiment) -> str:
    tickers_slug = "_".join(experiment.tickers)
    window_slug = f"{experiment.start_date}_to_{experiment.end_date}".replace("-", "")
    return f"backtest_{tickers_slug}_{window_slug}"


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _extract_async_telemetry(summary: Mapping[str, object] | None) -> dict[str, object] | None:
    if not isinstance(summary, Mapping):
        return None
    async_meta = summary.get("async_meta")
    if not isinstance(async_meta, Mapping):
        return None

    def _coerce(value: object | None) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    telemetry = {
        "async_agent_invoke_total_seconds": _coerce(async_meta.get("agent_invoke_total_seconds")),
        "async_per_agent_total_seconds": _coerce(async_meta.get("per_agent_total_seconds")),
        "async_concurrency_ratio": _coerce(async_meta.get("aggregate_concurrency")),
        "async_semaphore_utilization": _coerce(async_meta.get("semaphore_utilization")),
        "async_slowest_agent": async_meta.get("slowest_agent"),
        "async_slowest_agent_avg_seconds": _coerce(async_meta.get("slowest_agent_avg_seconds")),
    }
    return telemetry


def locate_timing_summary(
    experiment: Experiment,
    *,
    timing_dir: Path,
) -> tuple[Path | None, dict[str, object] | None, dict[str, object] | None]:
    """Locate the latest timing summary for an experiment."""

    slug = _timing_slug(experiment)
    if not timing_dir.exists():
        return None, None, None
    pattern = f"{slug}_*.json"
    candidates = sorted(
        timing_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    if not candidates:
        return None, None, None
    summary_path = candidates[0]
    summary_payload = _load_json(summary_path)
    async_telemetry = _extract_async_telemetry(summary_payload)
    return summary_path, summary_payload, async_telemetry


def default_log_path(label: str, start_date: str, end_date: str, *, now: datetime | None = None) -> Path:
    """Generate a unique log path for the current timestamp."""

    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S")
    safe_label = label.lower().replace(" ", "_")
    slug = f"backtest_{safe_label}_{start_date.replace('-', '')}_{end_date.replace('-', '')}_{timestamp}.log"
    return Path("log") / slug


def build_backtest_command(
    experiment: Experiment,
    *,
    model_provider: str,
    entrypoint: Path = BACKTESTER_ENTRYPOINT,
) -> list[str]:
    """Construct the CLI invocation for the given experiment."""

    command: list[str] = [
        "poetry",
        "run",
        "python",
        str(entrypoint),
        "--model-provider",
        model_provider,
    ]

    if experiment.analysts_all:
        command.append("--analysts-all")
    elif experiment.analysts:
        command.extend(["--analysts", ",".join(experiment.analysts)])

    tickers_arg = ",".join(experiment.tickers)
    command.extend(
        [
            "--tickers",
            tickers_arg,
            "--start-date",
            experiment.start_date,
            "--end-date",
            experiment.end_date,
            "--log-file",
            str(experiment.log_path),
        ]
    )

    if experiment.extra_args:
        command.extend(experiment.extra_args)

    return command


def _format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Best-effort kill of the spawned process and any children."""

    if process.poll() is not None:
        return

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - Windows specific branch
            process.terminate()
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()

    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - Windows specific branch
            process.kill()
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - shouldn't happen
        process.kill()


def run_command(command: Sequence[str], timeout_seconds: int) -> None:
    """Execute a subprocess command with strict timeout enforcement."""

    creationflags = 0
    popen_kwargs: dict[str, object] = {}
    if os.name == "nt":  # pragma: no cover - Windows specific branch
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        popen_kwargs["creationflags"] = creationflags
    else:
        popen_kwargs["start_new_session"] = True

    start_time = time.monotonic()
    with subprocess.Popen(command, stdout=None, stderr=None, **popen_kwargs) as process:
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - requires slow external call
            _terminate_process_tree(process)
            elapsed = time.monotonic() - start_time
            raise ExperimentRunError(f"Command timed out after {timeout_seconds}s (elapsed {elapsed:.1f}s): {_format_command(command)}") from exc

        if return_code != 0:
            _terminate_process_tree(process)
            raise ExperimentRunError(f"Command failed with exit code {return_code}: {_format_command(command)}")


def ensure_csv_header(path: Path) -> None:
    """Ensure the ledger CSV exists with the expected header."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_HEADER)
        return

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        existing_fields = reader.fieldnames or []
        rows = list(reader)

    if existing_fields == CSV_HEADER:
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_HEADER})


def parse_metrics(digest: BacktestDigest) -> Metrics:
    """Extract the subset of metrics needed for the scheduler ledger."""

    def _coerce(value: object | None) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    metrics = digest.metrics
    return Metrics(
        portfolio_return_pct=_coerce(metrics.get("portfolio_return_pct")),
        sharpe_ratio=_coerce(metrics.get("sharpe_ratio")),
        sortino_ratio=_coerce(metrics.get("sortino_ratio")),
        max_drawdown_pct=_coerce(metrics.get("max_drawdown_pct")),
        information_ratio=_coerce(metrics.get("information_ratio")),
        benchmark_return_pct=_coerce(metrics.get("benchmark_return_pct")),
        turnover_rate_pct=_coerce(metrics.get("turnover_rate_pct")),
    )


def load_baseline_metrics(label: str, csv_path: Path) -> Metrics | None:
    """Load the most recent baseline metrics for the given label from the ledger."""

    if not csv_path.exists():
        return None

    def _coerce(value: str | None) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    latest_row: dict[str, str] | None = None
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("label") == label:
                latest_row = row

    if latest_row is None:
        return None

    return Metrics(
        portfolio_return_pct=_coerce(latest_row.get("portfolio_return_pct")),
        sharpe_ratio=_coerce(latest_row.get("sharpe_ratio")),
        sortino_ratio=_coerce(latest_row.get("sortino_ratio")),
        max_drawdown_pct=_coerce(latest_row.get("max_drawdown_pct")),
        information_ratio=_coerce(latest_row.get("information_ratio")),
        benchmark_return_pct=_coerce(latest_row.get("benchmark_return_pct")),
        turnover_rate_pct=_coerce(latest_row.get("turnover_rate_pct")),
    )


def _format_optional(value: float | None, precision: int = 2) -> str:
    if value is None:
        return ""
    formatted = f"{value:.{precision}f}"
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def _compute_delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return current - baseline


def append_ledger_row(
    csv_path: Path,
    *,
    timestamp: datetime,
    experiment: Experiment,
    metrics: Metrics,
    baseline_label: str | None,
    deltas: tuple[float | None, float | None, float | None],
    timing_summary_path: Path | None = None,
    async_telemetry: Mapping[str, object] | None = None,
) -> None:
    """Append a single backtest record to the ledger CSV."""

    ensure_csv_header(csv_path)

    row = {
        "timestamp": timestamp.isoformat(),
        "label": experiment.label,
        "tickers": ",".join(experiment.tickers),
        "analysts": "all" if experiment.analysts_all else ",".join(experiment.analysts or []),
        "start_date": experiment.start_date,
        "end_date": experiment.end_date,
        "prompt_revision": experiment.prompt_revision,
        "portfolio_return_pct": _format_optional(metrics.portfolio_return_pct),
        "sharpe_ratio": _format_optional(metrics.sharpe_ratio),
        "sortino_ratio": _format_optional(metrics.sortino_ratio),
        "max_drawdown_pct": _format_optional(metrics.max_drawdown_pct),
        "information_ratio": _format_optional(metrics.information_ratio),
        "benchmark_return_pct": _format_optional(metrics.benchmark_return_pct),
        "turnover_rate_pct": _format_optional(metrics.turnover_rate_pct),
        "hit_rate": experiment.note or "NA",
        "log_path": str(experiment.log_path),
        "baseline_label": baseline_label or "",
        "delta_portfolio_return_pct": _format_optional(deltas[0]),
        "delta_sharpe_ratio": _format_optional(deltas[1]),
        "delta_max_drawdown_pct": _format_optional(deltas[2]),
    }
    if async_telemetry:
        row.update(
            {
                "timing_summary_path": str(timing_summary_path) if timing_summary_path else "",
                "async_agent_invoke_total_seconds": _format_optional(async_telemetry.get("async_agent_invoke_total_seconds")),
                "async_per_agent_total_seconds": _format_optional(async_telemetry.get("async_per_agent_total_seconds")),
                "async_concurrency_ratio": _format_optional(async_telemetry.get("async_concurrency_ratio")),
                "async_semaphore_utilization": _format_optional(async_telemetry.get("async_semaphore_utilization")),
                "async_slowest_agent": str(async_telemetry.get("async_slowest_agent") or ""),
                "async_slowest_agent_avg_seconds": _format_optional(async_telemetry.get("async_slowest_agent_avg_seconds")),
            }
        )
    else:
        for field in ASYNC_LEDGER_COLUMNS:
            row.setdefault(field, "")

    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        writer.writerow(row)


def schedule_experiments(
    experiments: Iterable[Experiment],
    *,
    model_provider: str = "azure",
    ledger_path: Path = Path("log/backtest.csv"),
    command_runner: RunCommand = run_command,
    digest_parser: ParseDigest = parse_backtest_digest,
    entrypoint: Path = BACKTESTER_ENTRYPOINT,
    timeout_override: int | None = None,
    include_async_telemetry: bool = False,
    timing_dir: Path | None = None,
) -> list[tuple[Experiment, Metrics]]:
    """Execute the experiment schedule and append results to the ledger."""

    results: list[tuple[Experiment, Metrics]] = []
    cached_metrics: dict[str, Metrics] = {}
    timing_directory = timing_dir or Path("log") / "backtest_timings"

    for experiment in experiments:
        experiment.log_path.parent.mkdir(parents=True, exist_ok=True)
        command = build_backtest_command(experiment, model_provider=model_provider, entrypoint=entrypoint)
        timeout = experiment.timeout_seconds or timeout_override or DEFAULT_TIMEOUT_SECONDS
        command_runner(command, timeout)

        digest = digest_parser(experiment.log_path)
        metrics = parse_metrics(digest)

        baseline_metrics: Metrics | None = None
        if experiment.baseline_label:
            baseline_metrics = cached_metrics.get(experiment.baseline_label)
            if baseline_metrics is None:
                baseline_metrics = load_baseline_metrics(experiment.baseline_label, ledger_path)

        deltas = (
            _compute_delta(metrics.portfolio_return_pct, baseline_metrics.portfolio_return_pct) if baseline_metrics else None,
            _compute_delta(metrics.sharpe_ratio, baseline_metrics.sharpe_ratio) if baseline_metrics else None,
            _compute_delta(metrics.max_drawdown_pct, baseline_metrics.max_drawdown_pct) if baseline_metrics else None,
        )

        timing_summary_path: Path | None = None
        async_telemetry: Mapping[str, object] | None = None
        if include_async_telemetry:
            timing_summary_path, _, async_telemetry = locate_timing_summary(experiment, timing_dir=timing_directory)

        append_ledger_row(
            ledger_path,
            timestamp=datetime.now(timezone.utc),
            experiment=experiment,
            metrics=metrics,
            baseline_label=experiment.baseline_label,
            deltas=deltas,
            timing_summary_path=timing_summary_path,
            async_telemetry=async_telemetry,
        )

        cached_metrics[experiment.label] = metrics
        results.append((experiment, metrics))

    return results


def _load_schedule(path: Path, *, now: datetime | None = None) -> tuple[list[Experiment], dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Schedule config must be a JSON object")

    experiments_payload = raw.get("experiments")
    if not isinstance(experiments_payload, list):
        raise ValueError("Config missing 'experiments' list")

    experiments = [Experiment.from_mapping(item, now=now) for item in experiments_payload if isinstance(item, Mapping)]

    timeout_raw = raw.get("timeout_seconds")
    timeout_seconds: int | None = None
    if timeout_raw is not None:
        try:
            timeout_seconds = int(timeout_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout_seconds must be an integer") from exc

    meta = {
        "model_provider": str(raw.get("model_provider", "azure")),
        "timeout_seconds": timeout_seconds,
        "ledger_path": Path(str(raw.get("ledger_path", "log/backtest.csv"))).expanduser(),
        "include_async_telemetry": bool(raw.get("include_async_telemetry", False)),
    }

    return experiments, meta


def _default_schedule(now: datetime | None = None) -> tuple[list[Experiment], dict[str, object]]:
    now = now or datetime.now(timezone.utc)
    defaults: list[Experiment] = [
        Experiment(
            label="multiticker_rally_current",
            tickers=["TSLA", "NVDA", "GOOGL"],
            start_date="2024-10-14",
            end_date="2024-11-11",
            log_path=default_log_path("multiticker_rally_current", "2024-10-14", "2024-11-11", now=now),
            prompt_revision="scheduler_default",
            baseline_label="rally_event_integration",
            note="scheduler default",
        ),
        Experiment(
            label="multiticker_consolidation_current",
            tickers=["TSLA", "NVDA", "GOOGL"],
            start_date="2025-07-01",
            end_date="2025-07-22",
            log_path=default_log_path("multiticker_consolidation_current", "2025-07-01", "2025-07-22", now=now),
            prompt_revision="scheduler_default",
            baseline_label="consolidation_event_catalyst",
            note="scheduler default",
        ),
        Experiment(
            label="multiticker_pullback_current",
            tickers=["TSLA", "NVDA", "GOOGL"],
            start_date="2025-02-07",
            end_date="2025-02-21",
            log_path=default_log_path("multiticker_pullback_current", "2025-02-07", "2025-02-21", now=now),
            prompt_revision="scheduler_default",
            baseline_label="crash_event_integration",
            note="scheduler default",
        ),
    ]

    meta = {
        "model_provider": "azure",
        "ledger_path": Path("log/backtest.csv"),
        "timeout_seconds": None,
        "include_async_telemetry": False,
    }
    return defaults, meta


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scheduled backtests across multiple regimes.")
    parser.add_argument("--config", type=Path, help="Optional JSON config describing the experiment schedule.")
    parser.add_argument("--ledger", type=Path, help="Override ledger CSV path (default log/backtest.csv).")
    parser.add_argument("--model-provider", default=None, help="Override model provider (default azure).")
    parser.add_argument("--timeout", type=int, help="Override per-experiment timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--include-async-telemetry",
        action="store_true",
        help="Append async timing telemetry into the ledger (reads log/backtest_timings).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    now = datetime.now(timezone.utc)

    if args.config:
        experiments, meta = _load_schedule(args.config, now=now)
    else:
        experiments, meta = _default_schedule(now=now)

    if not experiments:
        print("No experiments defined; exiting.")
        return 0

    ledger_path = args.ledger or meta.get("ledger_path", Path("log/backtest.csv"))
    ledger_path = Path(ledger_path).expanduser()

    model_provider = args.model_provider or meta.get("model_provider", "azure")
    meta_timeout = meta.get("timeout_seconds") if isinstance(meta, dict) else None
    timeout_override = args.timeout if args.timeout is not None else meta_timeout
    include_async_telemetry = args.include_async_telemetry or bool(meta.get("include_async_telemetry"))

    if args.dry_run:
        for experiment in experiments:
            command = build_backtest_command(experiment, model_provider=model_provider)
            timeout = experiment.timeout_seconds or timeout_override or DEFAULT_TIMEOUT_SECONDS
            print(f"[dry-run] timeout={timeout}s command={' '.join(command)}")
        return 0

    results = schedule_experiments(
        experiments,
        model_provider=model_provider,
        ledger_path=ledger_path,
        timeout_override=timeout_override,
        include_async_telemetry=include_async_telemetry,
    )

    for experiment, metrics in results:
        print(
            f"Recorded {experiment.label}: return={_format_optional(metrics.portfolio_return_pct)}% " f"sharpe={_format_optional(metrics.sharpe_ratio)} drawdown={_format_optional(metrics.max_drawdown_pct)}%",
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution path
    raise SystemExit(main())
