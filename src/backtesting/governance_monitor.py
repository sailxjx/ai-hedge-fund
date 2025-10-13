"""Monitor backtest ledger metrics and surface guardrail breaches.

Stage 5 automation wires continuous governance around the experiment ledger.
The CLI in this module inspects ``log/backtest.csv`` (or a custom ledger),
reports regressions against configurable guardrails, mines previously emitted
LLM diagnostics for recurring failure motifs, persists an aggregated report,
and, when guardrails are breached, appends TODO entries so the next iteration
loop is pre-seeded with investigation tasks.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the standard library ``types`` module retains precedence when this
# script is executed directly (``python src/backtesting/governance_monitor.py``).
# Otherwise, the sibling ``types.py`` file in this package shadows stdlib
# imports and breaks argparse/typing initialisation before we can patch paths.
if __package__ in (None, ""):
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parents[1]
    sys.path.insert(0, str(project_root))
    try:
        sys.path.remove(str(current_dir))
    except ValueError:
        pass

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Iterator, Sequence

DEFAULT_LEDGER = Path("log/backtest.csv")
DEFAULT_ANALYSIS_DIR = Path("log/analysis")
DEFAULT_TODO_PATH = Path("TODO.md")
DEFAULT_REPORT_PATH = Path("log/analysis/governance_report.json")
DEFAULT_ASYNC_REPORT_PATH = Path("log/analysis/governance_async.json")

DEFAULT_MIN_SHARPE = 0.5
DEFAULT_MAX_DRAWDOWN_MAGNITUDE = 5.0  # percent units (absolute)
DEFAULT_MIN_RETURN = 0.0
DEFAULT_REGRESSION_TOLERANCE = 0.1
DEFAULT_ASYNC_UTILIZATION_THRESHOLD = 0.2


@dataclass(slots=True)
class LedgerEntry:
    """Subset of ledger metrics tracked for guardrails."""

    timestamp: datetime
    label: str
    tickers: str | None
    analysts: str | None
    start_date: str | None
    end_date: str | None
    prompt_revision: str | None
    portfolio_return_pct: float | None
    sharpe_ratio: float | None
    max_drawdown_pct: float | None
    information_ratio: float | None
    benchmark_return_pct: float | None
    hit_rate: str | None
    log_path: str | None
    timing_summary_path: str | None = None
    async_agent_invoke_total: float | None = None
    async_per_agent_total: float | None = None
    async_concurrency_ratio: float | None = None
    async_semaphore_utilization: float | None = None
    async_slowest_agent: str | None = None
    async_slowest_agent_avg_seconds: float | None = None


@dataclass(slots=True)
class GuardrailBreach:
    """Triggered guardrail breach for a ledger entry."""

    label: str
    metric: str
    value: float
    threshold: float
    comparison: str
    timestamp: datetime
    log_path: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "timestamp": self.timestamp.isoformat(),
            "log_path": self.log_path,
        }


@dataclass(slots=True)
class AsyncLatencyAlert:
    """Async persona semaphore utilization alert derived from telemetry."""

    label: str
    semaphore_utilization: float
    concurrency_ratio: float | None
    agent_invoke_total: float | None
    per_agent_total: float | None
    slowest_agent: str | None
    slowest_agent_avg_seconds: float | None
    timestamp: datetime
    log_path: str | None
    timing_summary_path: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "semaphore_utilization": self.semaphore_utilization,
            "concurrency_ratio": self.concurrency_ratio,
            "agent_invoke_total": self.agent_invoke_total,
            "per_agent_total": self.per_agent_total,
            "slowest_agent": self.slowest_agent,
            "slowest_agent_avg_seconds": self.slowest_agent_avg_seconds,
            "timestamp": self.timestamp.isoformat(),
            "log_path": self.log_path,
            "timing_summary_path": self.timing_summary_path,
        }


@dataclass(slots=True)
class RegressionRecord:
    """Represents a regression relative to the previous run of the same label."""

    label: str
    metric: str
    previous_value: float
    current_value: float
    delta: float
    previous_timestamp: datetime
    current_timestamp: datetime
    log_path: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "metric": self.metric,
            "previous_value": self.previous_value,
            "current_value": self.current_value,
            "delta": self.delta,
            "previous_timestamp": self.previous_timestamp.isoformat(),
            "current_timestamp": self.current_timestamp.isoformat(),
            "log_path": self.log_path,
        }


@dataclass(slots=True)
class MotifSummary:
    """Aggregate recurring motif pulled from diagnostics logs."""

    issue_type: str
    description: str
    count: int
    recommendations: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "issue_type": self.issue_type,
            "description": self.description,
            "count": self.count,
            "recommendations": sorted(set(self.recommendations)),
            "sources": sorted(set(self.sources)),
        }


@dataclass(slots=True)
class GovernanceReport:
    """Top-level report returned by the governance monitor."""

    generated_at: datetime
    guardrail_breaches: list[GuardrailBreach]
    regressions: list[RegressionRecord]
    motifs: list[MotifSummary]
    async_alerts: list[AsyncLatencyAlert]

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "guardrail_breaches": [item.to_dict() for item in self.guardrail_breaches],
            "regressions": [item.to_dict() for item in self.regressions],
            "motifs": [item.to_dict() for item in self.motifs],
            "async_alerts": [item.to_dict() for item in self.async_alerts],
        }


def parse_float(value: str | None) -> float | None:
    """Convert CSV values into floats while tolerating blanks and sentinels."""

    if value is None:
        return None
    text = value.strip()
    if not text or text.upper() in {"NA", "N/A", "NULL"}:
        return None
    text = text.replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def parse_ledger_row(row: dict[str, str]) -> LedgerEntry | None:
    """Convert a CSV row into a :class:`LedgerEntry`."""

    timestamp_raw = row.get("timestamp")
    if not timestamp_raw:
        return None
    try:
        normalized = timestamp_raw.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        timestamp = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)

    return LedgerEntry(
        timestamp=timestamp,
        label=row.get("label", ""),
        tickers=row.get("tickers"),
        analysts=row.get("analysts"),
        start_date=row.get("start_date"),
        end_date=row.get("end_date"),
        prompt_revision=row.get("prompt_revision"),
        portfolio_return_pct=parse_float(row.get("portfolio_return_pct")),
        sharpe_ratio=parse_float(row.get("sharpe_ratio")),
        max_drawdown_pct=parse_float(row.get("max_drawdown_pct")),
        information_ratio=parse_float(row.get("information_ratio")),
        benchmark_return_pct=parse_float(row.get("benchmark_return_pct")),
        hit_rate=row.get("hit_rate"),
        log_path=row.get("log_path"),
        timing_summary_path=row.get("timing_summary_path"),
        async_agent_invoke_total=parse_float(row.get("async_agent_invoke_total_seconds")),
        async_per_agent_total=parse_float(row.get("async_per_agent_total_seconds")),
        async_concurrency_ratio=parse_float(row.get("async_concurrency_ratio")),
        async_semaphore_utilization=parse_float(row.get("async_semaphore_utilization")),
        async_slowest_agent=row.get("async_slowest_agent"),
        async_slowest_agent_avg_seconds=parse_float(row.get("async_slowest_agent_avg_seconds")),
    )


def load_ledger(path: Path) -> list[LedgerEntry]:
    """Load ledger entries sorted by timestamp."""

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            entries = [entry for row in reader if (entry := parse_ledger_row(row))]
    except FileNotFoundError:
        return []

    return sorted(entries, key=lambda entry: entry.timestamp)


def iter_latest_per_label(entries: Iterable[LedgerEntry]) -> Iterator[list[LedgerEntry]]:
    """Yield chronological runs grouped by label."""

    grouped: dict[str, list[LedgerEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.label, []).append(entry)
    for label_entries in grouped.values():
        label_entries.sort(key=lambda row: row.timestamp)
        yield label_entries


def detect_guardrail_breaches(
    entries: Iterable[LedgerEntry],
    *,
    min_sharpe: float,
    max_drawdown_magnitude: float,
    min_return: float,
) -> list[GuardrailBreach]:
    """Identify guardrail violations on the latest entry for each label."""

    breaches: list[GuardrailBreach] = []
    for label_entries in iter_latest_per_label(entries):
        latest = label_entries[-1]
        if latest.sharpe_ratio is not None and latest.sharpe_ratio < min_sharpe:
            breaches.append(
                GuardrailBreach(
                    label=latest.label,
                    metric="sharpe_ratio",
                    value=latest.sharpe_ratio,
                    threshold=min_sharpe,
                    comparison="<",
                    timestamp=latest.timestamp,
                    log_path=latest.log_path,
                )
            )
        if latest.max_drawdown_pct is not None and abs(latest.max_drawdown_pct) > max_drawdown_magnitude:
            breaches.append(
                GuardrailBreach(
                    label=latest.label,
                    metric="max_drawdown_pct",
                    value=latest.max_drawdown_pct,
                    threshold=max_drawdown_magnitude,
                    comparison=">",
                    timestamp=latest.timestamp,
                    log_path=latest.log_path,
                )
            )
        if latest.portfolio_return_pct is not None and latest.portfolio_return_pct < min_return:
            breaches.append(
                GuardrailBreach(
                    label=latest.label,
                    metric="portfolio_return_pct",
                    value=latest.portfolio_return_pct,
                    threshold=min_return,
                    comparison="<",
                    timestamp=latest.timestamp,
                    log_path=latest.log_path,
                )
            )
    return breaches


def detect_regressions(
    entries: Iterable[LedgerEntry],
    *,
    tolerance: float,
) -> list[RegressionRecord]:
    """Compare the latest run to the previous run for each label."""

    regressions: list[RegressionRecord] = []
    for label_entries in iter_latest_per_label(entries):
        if len(label_entries) < 2:
            continue
        previous, current = label_entries[-2], label_entries[-1]
        for metric in ("portfolio_return_pct", "sharpe_ratio"):
            previous_value = getattr(previous, metric)
            current_value = getattr(current, metric)
            if previous_value is None or current_value is None:
                continue
            delta = current_value - previous_value
            if delta < -tolerance:
                regressions.append(
                    RegressionRecord(
                        label=current.label,
                        metric=metric,
                        previous_value=previous_value,
                        current_value=current_value,
                        delta=delta,
                        previous_timestamp=previous.timestamp,
                        current_timestamp=current.timestamp,
                        log_path=current.log_path,
                    )
                )
    return regressions


def detect_async_latency_alerts(
    entries: Iterable[LedgerEntry],
    *,
    utilization_threshold: float,
) -> list[AsyncLatencyAlert]:
    """Identify async latency alerts based on semaphore utilization."""

    alerts: list[AsyncLatencyAlert] = []
    for label_entries in iter_latest_per_label(entries):
        latest = label_entries[-1]
        utilization = latest.async_semaphore_utilization
        if utilization is None or utilization <= utilization_threshold:
            continue
        alerts.append(
            AsyncLatencyAlert(
                label=latest.label,
                semaphore_utilization=utilization,
                concurrency_ratio=latest.async_concurrency_ratio,
                agent_invoke_total=latest.async_agent_invoke_total,
                per_agent_total=latest.async_per_agent_total,
                slowest_agent=latest.async_slowest_agent,
                slowest_agent_avg_seconds=latest.async_slowest_agent_avg_seconds,
                timestamp=latest.timestamp,
                log_path=latest.log_path,
                timing_summary_path=latest.timing_summary_path,
            )
        )
    return alerts


def mine_recurring_motifs(analysis_dir: Path) -> list[MotifSummary]:
    """Aggregate recurring diagnostic motifs from analysis artifacts."""

    aggregator: dict[tuple[str, str], MotifSummary] = {}

    def _add(issue_type: str, description: str, *, recommendation: str | None, source: str) -> None:
        key = (issue_type, description)
        if key not in aggregator:
            aggregator[key] = MotifSummary(issue_type=issue_type, description=description, count=0)
        entry = aggregator[key]
        entry.count += 1
        entry.sources.append(source)
        if recommendation:
            entry.recommendations.append(recommendation)

    combo_path = analysis_dir / "latest_llm_combo_diagnostics.json"
    if combo_path.exists():
        try:
            combo_payload = json.loads(combo_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            combo_payload = []
        if isinstance(combo_payload, list):
            for item in combo_payload:
                issues = item.get("issues", {}) if isinstance(item, dict) else {}
                if isinstance(issues, dict):
                    for issue_type, details in issues.items():
                        if not isinstance(details, list):
                            continue
                        for detail in details:
                            description = ""
                            recommendation = None
                            if isinstance(detail, dict):
                                description = str(detail.get("description") or detail.get("summary") or issue_type)
                            else:
                                description = str(detail)
                            if description:
                                _add(issue_type, description, recommendation=recommendation, source="llm_combo_diagnostics")

    iteration_path = analysis_dir / "agent_iteration_log.jsonl"
    if iteration_path.exists():
        try:
            with iteration_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    hypotheses = payload.get("hypotheses", [])
                    for hypothesis in hypotheses if isinstance(hypotheses, list) else []:
                        if not isinstance(hypothesis, dict):
                            continue
                        issue_type = str(hypothesis.get("issue") or "agent_hypothesis")
                        description = str(hypothesis.get("description") or issue_type)
                        recommendation = hypothesis.get("recommendation")
                        _add(issue_type, description, recommendation=recommendation, source="agent_iteration_log")
        except OSError:
            pass

    motifs = [entry for entry in aggregator.values() if entry.count > 0]
    motifs.sort(key=lambda item: (-item.count, item.issue_type, item.description))
    return motifs


def build_todo_updates(
    breaches: Sequence[GuardrailBreach],
    regeneration_time: datetime,
) -> list[str]:
    """Render TODO bullet lines for guardrail breaches."""

    if not breaches:
        return []
    date_slug = regeneration_time.strftime("%Y-%m-%d")
    updates: list[str] = []
    for breach in breaches:
        log_hint = f" (log {breach.log_path})" if breach.log_path else ""
        updates.append(f"  - [ ] {date_slug}: Guardrail {breach.metric} {breach.comparison} {breach.threshold} hit by '{breach.label}' ({breach.value:.2f}){log_hint}")
    return updates


def apply_todo_updates(todo_path: Path, updates: Sequence[str]) -> bool:
    """Insert updates under the Stage 5 section if they are not already present."""

    if not updates:
        return False
    try:
        lines = todo_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return False

    stage_header = "- [ ] Stage 5"
    try:
        index = next(i for i, line in enumerate(lines) if line.startswith(stage_header))
    except StopIteration:
        return False

    existing = set(line.strip() for line in lines)
    filtered_updates = [update for update in updates if update.strip() not in existing]
    if not filtered_updates:
        return False

    insert_at = index + 1
    while insert_at < len(lines) and lines[insert_at].startswith("  "):
        insert_at += 1

    new_lines = lines[:insert_at] + filtered_updates + lines[insert_at:]
    todo_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return True


def persist_report(report: GovernanceReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def persist_async_alerts(alerts: Sequence[AsyncLatencyAlert], path: Path) -> None:
    payload = [alert.to_dict() for alert in alerts]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_governance_monitor(
    *,
    ledger_path: Path,
    analysis_dir: Path,
    todo_path: Path | None,
    report_path: Path,
    min_sharpe: float,
    max_drawdown_magnitude: float,
    min_return: float,
    regression_tolerance: float,
    async_utilization_threshold: float,
    async_report_path: Path,
    dry_run: bool,
    now: datetime | None = None,
) -> GovernanceReport:
    entries = load_ledger(ledger_path)
    evaluation_time = now or datetime.now(timezone.utc)

    breaches = detect_guardrail_breaches(
        entries,
        min_sharpe=min_sharpe,
        max_drawdown_magnitude=max_drawdown_magnitude,
        min_return=min_return,
    )
    regressions = detect_regressions(entries, tolerance=regression_tolerance)
    motifs = mine_recurring_motifs(analysis_dir)
    async_alerts = detect_async_latency_alerts(entries, utilization_threshold=async_utilization_threshold)

    report = GovernanceReport(
        generated_at=evaluation_time,
        guardrail_breaches=breaches,
        regressions=regressions,
        motifs=motifs,
        async_alerts=async_alerts,
    )

    persist_report(report, report_path)
    persist_async_alerts(async_alerts, async_report_path)

    if not dry_run and todo_path is not None:
        updates = build_todo_updates(breaches, evaluation_time)
        apply_todo_updates(todo_path, updates)

    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor experiment ledger guardrails.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER, help="Path to backtest ledger CSV")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR, help="Directory containing diagnostics")
    parser.add_argument("--todo", type=Path, default=DEFAULT_TODO_PATH, help="Path to TODO.md for auto-queuing tasks")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH, help="Output JSON report path")
    parser.add_argument(
        "--async-report",
        type=Path,
        default=DEFAULT_ASYNC_REPORT_PATH,
        help="Output JSON path for async telemetry alerts.",
    )
    parser.add_argument("--min-sharpe", type=float, default=DEFAULT_MIN_SHARPE, help="Sharpe threshold guardrail")
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=DEFAULT_MAX_DRAWDOWN_MAGNITUDE,
        help="Maximum allowed drawdown magnitude (percent units)",
    )
    parser.add_argument(
        "--min-return",
        type=float,
        default=DEFAULT_MIN_RETURN,
        help="Minimum allowed portfolio return percent for guardrail",
    )
    parser.add_argument(
        "--regression-tolerance",
        type=float,
        default=DEFAULT_REGRESSION_TOLERANCE,
        help="Minimum drop required to flag a regression",
    )
    parser.add_argument(
        "--async-utilization-threshold",
        type=float,
        default=DEFAULT_ASYNC_UTILIZATION_THRESHOLD,
        help="Upper bound for async semaphore utilization before alerting (0-1 scale).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not modify TODO.md")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_governance_monitor(
        ledger_path=args.ledger,
        analysis_dir=args.analysis_dir,
        todo_path=None if args.dry_run else args.todo,
        report_path=args.report,
        min_sharpe=args.min_sharpe,
        max_drawdown_magnitude=args.max_drawdown,
        min_return=args.min_return,
        regression_tolerance=args.regression_tolerance,
        async_utilization_threshold=args.async_utilization_threshold,
        async_report_path=args.async_report,
        dry_run=args.dry_run,
    )

    summary_lines = [
        f"Generated governance report with {len(report.guardrail_breaches)} guardrail breaches, " f"{len(report.regressions)} regressions, and {len(report.motifs)} motifs.",
    ]
    if report.guardrail_breaches:
        for breach in report.guardrail_breaches:
            summary_lines.append(f"- {breach.label}: {breach.metric} {breach.comparison} {breach.threshold} (value={breach.value:.2f})")
    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
