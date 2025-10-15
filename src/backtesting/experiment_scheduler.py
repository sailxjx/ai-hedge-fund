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
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from src.tools.genome_registry import EvolutionGenome, GenomeLineageRegistry
from src.tools.genome_validation import validate_diversity
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
    "async_concurrency_limit",
]

GENOME_LEDGER_COLUMNS = [
    "generation_id",
    "arena_run_id",
    "genome_id",
    "genome_label",
    "genome_path",
    "genome_revision",
    "analyst_weights",
    "patriarch_variant",
    "async_mode",
    "model_name",
    "token_cost_usd",
    "llm_budget_used_usd",
    "metadata_path",
    "extra_tags",
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
] + GENOME_LEDGER_COLUMNS + ASYNC_LEDGER_COLUMNS


def _parse_env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None:
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _coerce_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
    except Exception:  # pragma: no cover - defensive
        return None
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _find_lingering_backtesters() -> list[int]:
    if os.name != "posix":
        return []
    try:
        output = subprocess.check_output(["pgrep", "-f", "src/backtester.py"], text=True)
    except FileNotFoundError:
        return []
    except subprocess.CalledProcessError:
        return []
    pids: list[int] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid <= 0 or pid == os.getpid():
            continue
        pids.append(pid)
    return sorted(set(pids))


def _terminate_pid(pid: int, *, timeout: float = 5.0) -> bool:
    if os.name != "posix":
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return True


def _log_guardrail_event(log_dir: Path, payload: Mapping[str, object]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "events.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _evaluate_governance_signals(digest: BacktestDigest) -> tuple[list[str], list[str]]:
    violations: list[str] = []
    warnings: list[str] = []

    risk = digest.risk
    if risk:
        if risk.remaining_limit is not None and risk.remaining_limit < 0:
            violations.append(f"Remaining risk limit negative ({risk.remaining_limit}).")
        for note in risk.constraint_notes:
            lower = note.lower()
            if any(keyword in lower for keyword in ("violation", "breach", "override failure")):
                violations.append(f"Constraint note indicates violation: {note}")
            elif note:
                warnings.append(f"Constraint note: {note}")
        if risk.overrides:
            overrides = ", ".join(f"{key}={value}" for key, value in sorted(risk.overrides.items()))
            warnings.append(f"Risk overrides active: {overrides}")

    for anomaly in digest.anomalies:
        description = f"{anomaly.type}: {anomaly.message}".strip()
        warnings.append(f"Arena anomaly detected: {description}")

    return violations, warnings


def _enforce_post_run_guardrails(
    *,
    experiment: Experiment,
    digest: BacktestDigest,
    async_telemetry: Mapping[str, object] | None,
    guardrail_log_dir: Path,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    violations: list[str] = []

    limit_env = _parse_env_int("LLM_ASYNC_MAX_CONCURRENCY")
    observed_limit = _coerce_float(async_telemetry.get("async_concurrency_limit")) if async_telemetry else None

    if async_telemetry:
        if limit_env is not None and observed_limit is not None and observed_limit > float(limit_env):
            message = (
                f"Async concurrency limit {observed_limit} exceeded configured cap {limit_env} "
                f"for experiment {experiment.label}."
            )
            violations.append(message)
            _log_guardrail_event(
                guardrail_log_dir,
                {
                    "timestamp": timestamp,
                    "severity": "violation",
                    "category": "async_concurrency_limit",
                    "experiment": experiment.label,
                    "generation_id": experiment.generation_id,
                    "arena_run_id": experiment.arena_run_id,
                    "genome_id": experiment.genome_id,
                    "message": message,
                },
            )

        ratio = _coerce_float(async_telemetry.get("async_concurrency_ratio"))
        if ratio is not None:
            reference_limit = float(limit_env) if limit_env is not None else observed_limit
            warning_threshold: float
            violation_threshold: float
            if reference_limit is not None:
                warning_threshold = reference_limit * 1.05
                violation_threshold = reference_limit * 2.0
            else:
                warning_threshold = 1.05
                violation_threshold = 1.5

            if ratio > violation_threshold:
                message = (
                    f"Async concurrency ratio {ratio:.2f} exceeded violation threshold {violation_threshold:.2f} "
                    f"for experiment {experiment.label}."
                )
                violations.append(message)
                _log_guardrail_event(
                    guardrail_log_dir,
                    {
                        "timestamp": timestamp,
                        "severity": "violation",
                        "category": "async_concurrency_ratio",
                        "experiment": experiment.label,
                        "generation_id": experiment.generation_id,
                        "arena_run_id": experiment.arena_run_id,
                        "genome_id": experiment.genome_id,
                        "message": message,
                    },
                )
            elif ratio > warning_threshold:
                message = (
                    f"Async concurrency ratio {ratio:.2f} exceeded warning threshold {warning_threshold:.2f} "
                    f"for experiment {experiment.label}."
                )
                _log_guardrail_event(
                    guardrail_log_dir,
                    {
                        "timestamp": timestamp,
                        "severity": "warning",
                        "category": "async_concurrency_ratio",
                        "experiment": experiment.label,
                        "generation_id": experiment.generation_id,
                        "arena_run_id": experiment.arena_run_id,
                        "genome_id": experiment.genome_id,
                        "message": message,
                    },
                )
    elif limit_env is not None and observed_limit is None:
        _log_guardrail_event(
            guardrail_log_dir,
            {
                "timestamp": timestamp,
                "severity": "warning",
                "category": "async_concurrency",
                "experiment": experiment.label,
                "generation_id": experiment.generation_id,
                "arena_run_id": experiment.arena_run_id,
                "genome_id": experiment.genome_id,
                "message": "Async telemetry missing concurrency_limit despite guardrail configuration.",
            },
        )

    lingering = _find_lingering_backtesters()
    if lingering:
        terminated: list[int] = []
        for pid in lingering:
            if _terminate_pid(pid):
                terminated.append(pid)
        _log_guardrail_event(
            guardrail_log_dir,
            {
                "timestamp": timestamp,
                "severity": "warning",
                "category": "process_cleanup",
                "experiment": experiment.label,
                "generation_id": experiment.generation_id,
                "arena_run_id": experiment.arena_run_id,
                "genome_id": experiment.genome_id,
                "message": f"Lingering src/backtester.py processes detected: {lingering}",
                "terminated": terminated,
            },
        )

    governance_violations, governance_warnings = _evaluate_governance_signals(digest)
    for warning in governance_warnings:
        _log_guardrail_event(
            guardrail_log_dir,
            {
                "timestamp": timestamp,
                "severity": "warning",
                "category": "governance",
                "experiment": experiment.label,
                "generation_id": experiment.generation_id,
                "arena_run_id": experiment.arena_run_id,
                "genome_id": experiment.genome_id,
                "message": warning,
            },
        )
    for violation in governance_violations:
        violations.append(violation)
        _log_guardrail_event(
            guardrail_log_dir,
            {
                "timestamp": timestamp,
                "severity": "violation",
                "category": "governance",
                "experiment": experiment.label,
                "generation_id": experiment.generation_id,
                "arena_run_id": experiment.arena_run_id,
                "genome_id": experiment.genome_id,
                "message": violation,
            },
        )

    if violations:
        combined = "; ".join(violations)
        raise ExperimentRunError(f"Guardrail violation detected: {combined}")


def _sanitize_slug(text: str, *, default: str = "item") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")
    return slug.lower() or default


def _emit_arena_reports(
    *,
    base_dir: Path,
    experiment: Experiment,
    metrics: Metrics,
    deltas: tuple[float | None, float | None, float | None],
    async_telemetry: Mapping[str, object] | None,
    timing_summary_path: Path | None,
    timestamp: datetime,
) -> None:
    generation = experiment.generation_id or "ungrouped"
    genome = experiment.genome_id or "no_genome"
    generation_slug = _sanitize_slug(generation, default="generation")
    genome_slug = _sanitize_slug(genome, default="genome")
    label_slug = _sanitize_slug(experiment.label, default="experiment")

    target_dir = base_dir / generation_slug / genome_slug
    target_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": timestamp.isoformat(),
        "label": experiment.label,
        "generation_id": experiment.generation_id,
        "arena_run_id": experiment.arena_run_id,
        "genome_id": experiment.genome_id,
        "genome_label": experiment.genome_label,
        "genome_path": str(experiment.genome_path) if experiment.genome_path else None,
        "tickers": experiment.tickers,
        "start_date": experiment.start_date,
        "end_date": experiment.end_date,
        "prompt_revision": experiment.prompt_revision,
        "baseline_label": experiment.baseline_label,
        "note": experiment.note,
        "log_path": str(experiment.log_path),
        "metrics": {
            "portfolio_return_pct": metrics.portfolio_return_pct,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "information_ratio": metrics.information_ratio,
            "benchmark_return_pct": metrics.benchmark_return_pct,
            "turnover_rate_pct": metrics.turnover_rate_pct,
        },
        "deltas": {
            "portfolio_return_pct": deltas[0],
            "sharpe_ratio": deltas[1],
            "max_drawdown_pct": deltas[2],
        },
        "async_telemetry": async_telemetry or None,
        "timing_summary_path": str(timing_summary_path) if timing_summary_path else None,
        "patriarch_variant": experiment.patriarch_variant,
        "analyst_weights": experiment.analyst_weights,
        "metadata_path": str(experiment.metadata_path) if experiment.metadata_path else None,
        "extra_tags": list(experiment.extra_tags or []),
        "token_cost_usd": experiment.token_cost_usd,
        "llm_budget_used_usd": experiment.llm_budget_used_usd,
        "async_mode": experiment.async_mode,
        "model_name": experiment.model_name,
    }

    json_path = target_dir / f"{label_slug}.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    markdown_lines = [
        f"# {experiment.label}",
        "",
        f"- Generation: {experiment.generation_id or 'n/a'}",
        f"- Genome: {experiment.genome_id or 'n/a'}",
        f"- Window: {experiment.start_date} → {experiment.end_date}",
        f"- Tickers: {', '.join(experiment.tickers)}",
        f"- Prompt Revision: {experiment.prompt_revision}",
        f"- Portfolio Return: {summary['metrics']['portfolio_return_pct']}%",
        f"- Sharpe Ratio: {summary['metrics']['sharpe_ratio']}",
        f"- Max Drawdown: {summary['metrics']['max_drawdown_pct']}%",
        f"- Baseline Delta (return): {summary['deltas']['portfolio_return_pct']}",
        f"- Note: {experiment.note or '—'}",
        "",
        f"Log: `{experiment.log_path}`",
    ]
    if timing_summary_path:
        markdown_lines.append(f"Timing Summary: `{timing_summary_path}`")
    markdown_lines.append("")

    md_path = target_dir / f"{label_slug}.md"
    md_path.write_text("\n".join(markdown_lines), encoding="utf-8")


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
    generation_id: str | None = None
    arena_run_id: str | None = None
    genome_id: str | None = None
    genome_label: str | None = None
    genome_revision: str | None = None
    analyst_weights: Mapping[str, float] | None = None
    patriarch_variant: str | None = None
    async_mode: str | None = None
    model_name: str | None = None
    token_cost_usd: float | None = None
    llm_budget_used_usd: float | None = None
    genome_path: Path | None = None
    metadata_path: Path | None = None
    extra_tags: Sequence[str] | None = None

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

        def _as_optional_str(value: object | None) -> str | None:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        def _as_optional_float(value: object | None) -> float | None:
            if value is None:
                return None
            if isinstance(value, (float, int)):
                return float(value)
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return None
                try:
                    return float(stripped)
                except ValueError as exc:
                    raise ValueError(f"Expected float-compatible value, got {value!r}") from exc
            return None

        def _as_weights(value: object | None) -> Mapping[str, float] | None:
            if value is None:
                return None
            if isinstance(value, Mapping):
                weights: dict[str, float] = {}
                for key, raw in value.items():
                    try:
                        weights[str(key)] = float(raw)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        continue
                return weights or None
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return None
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError as exc:  # pragma: no cover - invalid config guarded upstream
                    raise ValueError("analyst_weights must be a mapping or JSON string") from exc
                if not isinstance(parsed, Mapping):
                    raise ValueError("analyst_weights JSON must decode to a mapping")
                return _as_weights(parsed)
            raise ValueError("analyst_weights must be a mapping or JSON string")

        def _as_optional_path(value: object | None) -> Path | None:
            if value is None:
                return None
            text = str(value).strip()
            if not text:
                return None
            return Path(text).expanduser()

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
        note = _as_optional_str(payload.get("note"))
        timeout_seconds = None
        if payload.get("timeout_seconds") is not None:
            timeout_seconds = int(payload["timeout_seconds"])

        generation_id = _as_optional_str(payload.get("generation_id"))
        arena_run_id = _as_optional_str(payload.get("arena_run_id"))
        genome_id = _as_optional_str(payload.get("genome_id"))
        genome_label = _as_optional_str(payload.get("genome_label"))
        genome_revision = _as_optional_str(payload.get("genome_revision"))
        analyst_weights = _as_weights(payload.get("analyst_weights"))
        patriarch_variant = _as_optional_str(payload.get("patriarch_variant"))
        async_mode = _as_optional_str(payload.get("async_mode"))
        model_name = _as_optional_str(payload.get("model_name"))
        token_cost_usd = _as_optional_float(payload.get("token_cost_usd"))
        llm_budget_used_usd = _as_optional_float(payload.get("llm_budget_used_usd"))
        metadata_path_value = payload.get("metadata_path")
        metadata_path = _as_optional_path(metadata_path_value)
        genome_path = _as_optional_path(payload.get("genome_path"))
        extra_tags = _as_str_list(payload.get("extra_tags"))

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
            generation_id=generation_id,
            arena_run_id=arena_run_id,
            genome_id=genome_id,
            genome_label=genome_label,
            genome_revision=genome_revision,
            analyst_weights=analyst_weights,
            patriarch_variant=patriarch_variant,
            async_mode=async_mode,
            model_name=model_name,
            token_cost_usd=token_cost_usd,
            llm_budget_used_usd=llm_budget_used_usd,
            genome_path=genome_path,
            metadata_path=metadata_path,
            extra_tags=extra_tags,
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
        "async_concurrency_limit": _coerce(async_meta.get("concurrency_limit")),
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

    def _weights_to_json(weights: Mapping[str, float] | None) -> str:
        if not weights:
            return ""
        try:
            return json.dumps(dict(sorted(weights.items())), separators=(",", ":"), sort_keys=True)
        except TypeError:
            return ""

    row.update(
        {
            "generation_id": experiment.generation_id or "",
            "arena_run_id": experiment.arena_run_id or "",
            "genome_id": experiment.genome_id or "",
            "genome_label": experiment.genome_label or "",
            "genome_path": str(experiment.genome_path) if experiment.genome_path else "",
            "genome_revision": experiment.genome_revision or "",
            "analyst_weights": _weights_to_json(experiment.analyst_weights),
            "patriarch_variant": experiment.patriarch_variant or "",
            "async_mode": experiment.async_mode or ("async" if async_telemetry else ""),
            "model_name": experiment.model_name or "",
            "token_cost_usd": _format_optional(experiment.token_cost_usd),
            "llm_budget_used_usd": _format_optional(experiment.llm_budget_used_usd),
            "metadata_path": str(experiment.metadata_path) if experiment.metadata_path else "",
            "extra_tags": ",".join(experiment.extra_tags) if experiment.extra_tags else "",
        }
    )

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
                "async_concurrency_limit": _format_optional(async_telemetry.get("async_concurrency_limit"), precision=0),
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
    genome_registry: GenomeLineageRegistry | None = None,
    diversity_guard: bool = False,
    analyst_overlap_threshold: float = 0.8,
    weight_similarity_threshold: float = 0.9,
    arena_report_dir: Path | None = None,
    guardrail_log_dir: Path | None = None,
) -> list[tuple[Experiment, Metrics]]:
    """Execute the experiment schedule and append results to the ledger."""

    results: list[tuple[Experiment, Metrics]] = []
    cached_metrics: dict[str, Metrics] = {}
    timing_directory = timing_dir or Path("log") / "backtest_timings"

    validated_genomes: dict[str, list[EvolutionGenome]] = {}

    guardrail_directory = guardrail_log_dir or (Path("log") / "arena" / "guardrails")

    for experiment in experiments:
        experiment.log_path.parent.mkdir(parents=True, exist_ok=True)
        if diversity_guard and genome_registry and experiment.genome_id:
            try:
                genome = genome_registry.load(experiment.genome_id)
            except FileNotFoundError as exc:  # pragma: no cover - configuration error
                raise ExperimentRunError(f"Genome '{experiment.genome_id}' not found for experiment '{experiment.label}'") from exc
            generation_key = experiment.generation_id or genome.generation_id or "default"
            cohort = validated_genomes.setdefault(generation_key, [])
            is_valid, assessment = validate_diversity(
                genome,
                cohort,
                analyst_overlap_threshold=analyst_overlap_threshold,
                weight_similarity_threshold=weight_similarity_threshold,
            )
            if not is_valid:
                warning_text = "; ".join(assessment.warnings) or "diversity guard triggered"
                raise ExperimentRunError(
                    f"Genome '{experiment.genome_id}' failed diversity guard: {warning_text}"
                )
            cohort.append(genome)
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
        need_async_meta = include_async_telemetry or os.getenv("LLM_ASYNC_MAX_CONCURRENCY")
        if need_async_meta:
            timing_summary_path, _, async_telemetry = locate_timing_summary(experiment, timing_dir=timing_directory)

        recorded_at = datetime.now(timezone.utc)

        _enforce_post_run_guardrails(
            experiment=experiment,
            digest=digest,
            async_telemetry=async_telemetry,
            guardrail_log_dir=guardrail_directory,
        )

        if not include_async_telemetry:
            timing_summary_path = None
            async_telemetry = None

        append_ledger_row(
            ledger_path,
            timestamp=recorded_at,
            experiment=experiment,
            metrics=metrics,
            baseline_label=experiment.baseline_label,
            deltas=deltas,
            timing_summary_path=timing_summary_path,
            async_telemetry=async_telemetry,
        )

        if arena_report_dir is not None:
            _emit_arena_reports(
                base_dir=arena_report_dir,
                experiment=experiment,
                metrics=metrics,
                deltas=deltas,
                async_telemetry=async_telemetry,
                timing_summary_path=timing_summary_path,
                timestamp=recorded_at,
            )

        cached_metrics[experiment.label] = metrics
        results.append((experiment, metrics))

    return results


def _load_schedule(path: Path, *, now: datetime | None = None) -> tuple[list[Experiment], dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Schedule config must be a JSON object")

    now = now or datetime.now(timezone.utc)

    def _with_defaults(payload: Mapping[str, object]) -> dict[str, object]:
        merged = dict(payload)
        if raw.get("generation_id") and not merged.get("generation_id"):
            merged["generation_id"] = raw["generation_id"]
        if raw.get("arena_run_id") and not merged.get("arena_run_id"):
            merged["arena_run_id"] = raw["arena_run_id"]
        if raw.get("prompt_revision") and not merged.get("prompt_revision"):
            merged["prompt_revision"] = raw["prompt_revision"]
        if raw.get("model_name") and not merged.get("model_name"):
            merged["model_name"] = raw["model_name"]
        if raw.get("async_mode") and not merged.get("async_mode"):
            merged["async_mode"] = raw["async_mode"]
        if raw.get("genome_id") and not merged.get("genome_id"):
            merged["genome_id"] = raw["genome_id"]
        if raw.get("genome_path") and not merged.get("genome_path"):
            merged["genome_path"] = raw["genome_path"]
        return merged

    experiments: list[Experiment] = []
    experiments_payload = raw.get("experiments")
    if isinstance(experiments_payload, list):
        for item in experiments_payload:
            if not isinstance(item, Mapping):
                continue
            experiments.append(Experiment.from_mapping(_with_defaults(item), now=now))

    ticker_bundles = raw.get("ticker_bundles")
    windows = raw.get("windows")
    log_dir_value = raw.get("log_dir")
    log_dir: Path | None = Path(str(log_dir_value)).expanduser() if log_dir_value else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(ticker_bundles, list) and isinstance(windows, list):
        for window in windows:
            if not isinstance(window, Mapping):
                continue
            start_date = str(window.get("start_date", "")).strip()
            end_date = str(window.get("end_date", "")).strip()
            if not start_date or not end_date:
                raise ValueError("Window entries require start_date and end_date")
            window_label = str(window.get("label", f"{start_date}_to_{end_date}")).strip()
            window_generation = window.get("generation_id") or raw.get("generation_id")
            window_prompt = window.get("prompt_revision") or raw.get("prompt_revision")
            window_note = window.get("note")
            window_baseline = window.get("baseline_label")
            window_arena = window.get("arena_run_id") or raw.get("arena_run_id")

            for bundle in ticker_bundles:
                if not isinstance(bundle, Mapping):
                    continue
                tickers = bundle.get("tickers")
                if not tickers:
                    raise ValueError("ticker_bundles entries require tickers")
                bundle_label = str(bundle.get("label", "_".join(str(t).upper() for t in tickers))).strip()
                label = f"{window_label}_{bundle_label}" if bundle_label else window_label
                genome_id = bundle.get("genome_id") or window.get("genome_id") or raw.get("genome_id")
                genome_path = bundle.get("genome_path") or window.get("genome_path") or raw.get("genome_path")
                prompt_revision = bundle.get("prompt_revision") or window_prompt or raw.get("prompt_revision", "baseline")
                analysts_all = bundle.get("analysts_all", True)
                analysts = bundle.get("analysts") if not analysts_all else None
                baseline_label = bundle.get("baseline_label") or window_baseline
                note = bundle.get("note") or window_note
                log_path_value = bundle.get("log_path") or window.get("log_path")
                if log_path_value:
                    log_path = Path(str(log_path_value)).expanduser()
                elif log_dir is not None:
                    slug = f"{label}_{start_date.replace('-', '')}_{end_date.replace('-', '')}.log"
                    log_path = log_dir / slug
                else:
                    log_path = default_log_path(label, start_date, end_date, now=now)

                payload: dict[str, object] = {
                    "label": label,
                    "tickers": tickers,
                    "start_date": start_date,
                    "end_date": end_date,
                    "log_path": str(log_path),
                    "prompt_revision": prompt_revision,
                    "baseline_label": baseline_label,
                    "analysts_all": analysts_all,
                }
                if analysts is not None:
                    payload["analysts"] = analysts
                if note:
                    payload["note"] = note
                if genome_id:
                    payload["genome_id"] = genome_id
                if genome_path:
                    payload["genome_path"] = genome_path
                if window_generation:
                    payload.setdefault("generation_id", window_generation)
                if window_arena:
                    payload.setdefault("arena_run_id", window_arena)
                if bundle.get("extra_args"):
                    payload["extra_args"] = bundle["extra_args"]
                if bundle.get("timeout_seconds"):
                    payload["timeout_seconds"] = bundle["timeout_seconds"]
                if bundle.get("note"):
                    payload["note"] = bundle["note"]
                experiments.append(Experiment.from_mapping(_with_defaults(payload), now=now))

    if not experiments:
        raise ValueError("Schedule configuration produced no experiments")

    timeout_raw = raw.get("timeout_seconds")
    timeout_seconds: int | None = None
    if timeout_raw is not None:
        try:
            timeout_seconds = int(timeout_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout_seconds must be an integer") from exc

    def _as_float(value: object | None, default: float) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    genome_registry_path_value = raw.get("genome_registry_path")
    genome_registry_path = Path(str(genome_registry_path_value)).expanduser() if genome_registry_path_value else None
    arena_report_dir_value = raw.get("arena_report_dir")
    arena_report_dir = Path(str(arena_report_dir_value)).expanduser() if arena_report_dir_value else None
    if log_dir is not None:
        arena_report_dir = log_dir
    guardrail_log_dir_value = raw.get("guardrail_log_dir")
    guardrail_log_dir = Path(str(guardrail_log_dir_value)).expanduser() if guardrail_log_dir_value else None

    meta = {
        "model_provider": str(raw.get("model_provider", "azure")),
        "timeout_seconds": timeout_seconds,
        "ledger_path": Path(str(raw.get("ledger_path", "log/backtest.csv"))).expanduser(),
        "include_async_telemetry": bool(raw.get("include_async_telemetry", False)),
        "generation_id": raw.get("generation_id"),
        "arena_run_id": raw.get("arena_run_id"),
        "diversity_guard": bool(raw.get("diversity_guard", False)),
        "analyst_overlap_threshold": _as_float(raw.get("analyst_overlap_threshold"), 0.8),
        "weight_similarity_threshold": _as_float(raw.get("weight_similarity_threshold"), 0.9),
    }
    if genome_registry_path:
        meta["genome_registry_path"] = genome_registry_path
    if arena_report_dir is not None:
        meta["arena_report_dir"] = arena_report_dir
    if guardrail_log_dir is not None:
        meta["guardrail_log_dir"] = guardrail_log_dir

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
        "generation_id": None,
        "arena_run_id": None,
        "diversity_guard": False,
        "analyst_overlap_threshold": 0.8,
        "weight_similarity_threshold": 0.9,
        "genome_registry_path": None,
        "arena_report_dir": Path("log") / "arena",
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

    genome_registry: GenomeLineageRegistry | None = None
    registry_path = meta.get("genome_registry_path")
    if registry_path:
        genome_registry = GenomeLineageRegistry(root=Path(str(registry_path)).expanduser())

    arena_report_dir_meta = meta.get("arena_report_dir")
    if arena_report_dir_meta is None:
        arena_report_dir = Path("log") / "arena"
    else:
        arena_report_dir = Path(str(arena_report_dir_meta)).expanduser()

    def _extract_float(value: object | None, default: float) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    diversity_guard = bool(meta.get("diversity_guard", False))
    analyst_overlap_threshold = _extract_float(meta.get("analyst_overlap_threshold"), 0.8)
    weight_similarity_threshold = _extract_float(meta.get("weight_similarity_threshold"), 0.9)

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
        genome_registry=genome_registry,
        diversity_guard=diversity_guard,
        analyst_overlap_threshold=analyst_overlap_threshold,
        weight_similarity_threshold=weight_similarity_threshold,
        arena_report_dir=arena_report_dir,
    )

    for experiment, metrics in results:
        print(
            f"Recorded {experiment.label}: return={_format_optional(metrics.portfolio_return_pct)}% " f"sharpe={_format_optional(metrics.sharpe_ratio)} drawdown={_format_optional(metrics.max_drawdown_pct)}%",
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution path
    raise SystemExit(main())
