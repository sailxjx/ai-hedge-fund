from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.backtesting.experiment_scheduler import CSV_HEADER
from src.backtesting.governance_monitor import (
    apply_todo_updates,
    AsyncLatencyAlert,
    build_todo_updates,
    detect_async_latency_alerts,
    detect_guardrail_breaches,
    detect_regressions,
    GuardrailBreach,
    LedgerEntry,
    mine_recurring_motifs,
    run_governance_monitor,
)


def make_entry(
    *,
    timestamp: datetime,
    label: str,
    portfolio_return: float,
    sharpe: float,
    drawdown: float,
    log_path: str | None = None,
    async_utilization: float | None = None,
) -> LedgerEntry:
    return LedgerEntry(
        timestamp=timestamp,
        label=label,
        tickers="TSLA",
        analysts="all",
        start_date="2025-01-01",
        end_date="2025-01-15",
        prompt_revision="baseline",
        portfolio_return_pct=portfolio_return,
        sharpe_ratio=sharpe,
        max_drawdown_pct=drawdown,
        information_ratio=None,
        benchmark_return_pct=None,
        hit_rate="NA",
        log_path=log_path,
        timing_summary_path="log/backtest_timings/sample.json" if async_utilization is not None else None,
        async_agent_invoke_total=12.0 if async_utilization is not None else None,
        async_per_agent_total=18.0 if async_utilization is not None else None,
        async_concurrency_ratio=1.5 if async_utilization is not None else None,
        async_semaphore_utilization=async_utilization,
        async_slowest_agent="technical_analyst_agent" if async_utilization is not None else None,
        async_slowest_agent_avg_seconds=3.5 if async_utilization is not None else None,
    )


def test_detect_guardrail_breaches_uses_latest_entry() -> None:
    t0 = datetime(2025, 9, 27, tzinfo=timezone.utc)
    t1 = datetime(2025, 9, 28, tzinfo=timezone.utc)
    good = make_entry(timestamp=t0, label="alpha", portfolio_return=1.2, sharpe=1.5, drawdown=-2.0)
    bad = make_entry(timestamp=t1, label="alpha", portfolio_return=-0.3, sharpe=-0.4, drawdown=-7.5)

    breaches = detect_guardrail_breaches(
        [good, bad],
        min_sharpe=0.5,
        max_drawdown_magnitude=5.0,
        min_return=0.0,
    )

    assert {(b.metric, b.comparison) for b in breaches} == {
        ("sharpe_ratio", "<"),
        ("max_drawdown_pct", ">"),
        ("portfolio_return_pct", "<"),
    }
    assert all(b.label == "alpha" for b in breaches)
    assert all(b.timestamp == t1 for b in breaches)


def test_detect_regressions_requires_drop_beyond_tolerance() -> None:
    t0 = datetime(2025, 9, 27, tzinfo=timezone.utc)
    t1 = datetime(2025, 9, 28, tzinfo=timezone.utc)
    previous = make_entry(timestamp=t0, label="beta", portfolio_return=1.0, sharpe=0.8, drawdown=-3.0)
    current = make_entry(timestamp=t1, label="beta", portfolio_return=0.95, sharpe=0.72, drawdown=-3.5)
    regressions = detect_regressions([previous, current], tolerance=0.1)
    assert len(regressions) == 0

    sharper_drop = make_entry(timestamp=t1, label="beta", portfolio_return=0.4, sharpe=-0.1, drawdown=-4.0)
    regressions = detect_regressions([previous, sharper_drop], tolerance=0.1)
    metrics = {(r.metric, round(r.delta, 2)) for r in regressions}
    assert metrics == {("portfolio_return_pct", -0.6), ("sharpe_ratio", -0.9)}


def test_detect_async_latency_alerts_flags_threshold() -> None:
    t0 = datetime(2025, 9, 27, tzinfo=timezone.utc)
    entry = make_entry(timestamp=t0, label="async_alpha", portfolio_return=0.3, sharpe=0.4, drawdown=-2.0, async_utilization=0.35)
    alerts = detect_async_latency_alerts([entry], utilization_threshold=0.2)
    assert alerts
    alert = alerts[0]
    assert isinstance(alert, AsyncLatencyAlert)
    assert alert.label == "async_alpha"
    assert alert.semaphore_utilization == pytest.approx(0.35)


def test_build_and_apply_todo_updates(tmp_path: Path) -> None:
    todo_path = tmp_path / "TODO.md"
    todo_path.write_text(
        """# TODO\n\n- [x] Stage 4 – Complete scheduler\n  - 2025-09-30: Scheduler logging validated\n- [ ] Stage 5 – Wire continuous governance\n""",
        encoding="utf-8",
    )

    now = datetime(2025, 10, 3, tzinfo=timezone.utc)
    breach_updates = build_todo_updates(
        [
            GuardrailBreach(
                label="alpha",
                metric="sharpe_ratio",
                value=-0.4,
                threshold=0.5,
                comparison="<",
                timestamp=now,
                log_path="log/backtest_alpha.log",
            ),
            GuardrailBreach(
                label="alpha",
                metric="portfolio_return_pct",
                value=-0.3,
                threshold=0.0,
                comparison="<",
                timestamp=now,
                log_path="log/backtest_alpha.log",
            ),
        ],
        now,
    )

    written = apply_todo_updates(todo_path, breach_updates)
    assert written is True

    contents = todo_path.read_text(encoding="utf-8")
    assert "Guardrail sharpe_ratio" in contents
    second_write = apply_todo_updates(todo_path, breach_updates)
    assert second_write is False


def test_mine_recurring_motifs_aggregates_sources(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)

    combo_payload = [{"issues": {"sentinel_disagreements": [{"description": "Risk overrides blocking shorts"}]}}]
    (analysis_dir / "latest_llm_combo_diagnostics.json").write_text(
        json.dumps(combo_payload),
        encoding="utf-8",
    )

    iteration_line = {
        "hypotheses": [
            {
                "issue": "sentinel_disagreements",
                "description": "Risk overrides blocking shorts",
                "recommendation": "Relax Range Recovery Sentinel guardrail",
            }
        ]
    }
    (analysis_dir / "agent_iteration_log.jsonl").write_text(json.dumps(iteration_line) + "\n", encoding="utf-8")

    motifs = mine_recurring_motifs(analysis_dir)
    assert len(motifs) == 1
    motif = motifs[0]
    assert motif.issue_type == "sentinel_disagreements"
    assert motif.count == 2
    assert "agent_iteration_log" in motif.sources


def test_run_governance_monitor_writes_report_and_async_alerts(tmp_path: Path) -> None:
    ledger_path = tmp_path / "log" / "backtest.csv"
    ledger_path.parent.mkdir(parents=True)

    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "2025-09-28T00:00:00+00:00",
                "label": "alpha",
                "tickers": "TSLA",
                "analysts": "all",
                "start_date": "2025-08-01",
                "end_date": "2025-08-15",
                "prompt_revision": "baseline_v1",
                "portfolio_return_pct": "1.20",
                "sharpe_ratio": "0.80",
                "max_drawdown_pct": "-3.0",
                "information_ratio": "0.10",
                "benchmark_return_pct": "0.90",
                "hit_rate": "NA",
                "log_path": "log/backtest_alpha_baseline.log",
                "baseline_label": "",
                "delta_portfolio_return_pct": "",
                "delta_sharpe_ratio": "",
                "delta_max_drawdown_pct": "",
                "timing_summary_path": "",
                "async_agent_invoke_total_seconds": "",
                "async_per_agent_total_seconds": "",
                "async_concurrency_ratio": "",
                "async_semaphore_utilization": "",
                "async_slowest_agent": "",
                "async_slowest_agent_avg_seconds": "",
            }
        )
        writer.writerow(
            {
                "timestamp": "2025-09-29T00:00:00+00:00",
                "label": "alpha",
                "tickers": "TSLA",
                "analysts": "all",
                "start_date": "2025-08-15",
                "end_date": "2025-08-30",
                "prompt_revision": "variant_v1",
                "portfolio_return_pct": "-1.50",
                "sharpe_ratio": "-0.20",
                "max_drawdown_pct": "-7.0",
                "information_ratio": "-0.40",
                "benchmark_return_pct": "0.50",
                "hit_rate": "NA",
                "log_path": "log/backtest_alpha_variant.log",
                "baseline_label": "alpha",
                "delta_portfolio_return_pct": "-2.70",
                "delta_sharpe_ratio": "-1.00",
                "delta_max_drawdown_pct": "-4.0",
                "timing_summary_path": "log/backtest_timings/alpha.json",
                "async_agent_invoke_total_seconds": "9.5",
                "async_per_agent_total_seconds": "14.5",
                "async_concurrency_ratio": "1.5",
                "async_semaphore_utilization": "0.3",
                "async_slowest_agent": "technical_analyst_agent",
                "async_slowest_agent_avg_seconds": "3.5",
            }
        )

    todo_path = tmp_path / "TODO.md"
    todo_path.write_text(
        """# TODO\n\n- [ ] Stage 5 – Wire continuous governance\n""",
        encoding="utf-8",
    )

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "latest_llm_combo_diagnostics.json").write_text("[]", encoding="utf-8")

    report_path = tmp_path / "analysis" / "governance_report.json"
    async_report_path = tmp_path / "analysis" / "governance_async.json"

    report = run_governance_monitor(
        ledger_path=ledger_path,
        analysis_dir=analysis_dir,
        todo_path=todo_path,
        report_path=report_path,
        min_sharpe=0.5,
        max_drawdown_magnitude=5.0,
        min_return=0.0,
        regression_tolerance=0.1,
        async_utilization_threshold=0.2,
        async_report_path=async_report_path,
        dry_run=False,
        now=datetime(2025, 10, 3, tzinfo=timezone.utc),
    )

    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["guardrail_breaches"]
    assert payload["regressions"]
    assert payload["async_alerts"]

    todo_contents = todo_path.read_text(encoding="utf-8")
    assert "Guardrail" in todo_contents

    assert async_report_path.exists()
    async_payload = json.loads(async_report_path.read_text(encoding="utf-8"))
    assert async_payload
    assert async_payload[0]["semaphore_utilization"] == pytest.approx(0.3)
    assert len(report.guardrail_breaches) == len(payload["guardrail_breaches"])
