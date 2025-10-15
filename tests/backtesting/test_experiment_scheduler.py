from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pytest

from src.backtesting.experiment_scheduler import (
    build_backtest_command,
    CSV_HEADER,
    ensure_csv_header,
    Experiment,
    ExperimentRunError,
    run_command,
    schedule_experiments,
    _load_schedule,
)
from src.tools.genome_registry import EvolutionGenome, GenomeLineageRegistry
from src.tools.llm_backtest_digest import BacktestDigest, RiskTelemetry, parse_backtest_digest


def test_run_command_times_out() -> None:
    command = [sys.executable, "-c", "import time; time.sleep(5)"]

    start = time.monotonic()
    with pytest.raises(ExperimentRunError) as excinfo:
        run_command(command, timeout_seconds=1)

    elapsed = time.monotonic() - start
    assert elapsed < 4, "run_command did not enforce timeout promptly"
    assert "timed out" in str(excinfo.value)


def test_run_command_raises_on_non_zero_exit() -> None:
    command = [sys.executable, "-c", "import sys; sys.exit(3)"]

    with pytest.raises(ExperimentRunError) as excinfo:
        run_command(command, timeout_seconds=5)

    assert "exit code 3" in str(excinfo.value)


def test_build_backtest_command_with_custom_analysts(tmp_path: Path) -> None:
    experiment = Experiment(
        label="custom_run",
        tickers=["TSLA", "NVDA"],
        start_date="2024-09-01",
        end_date="2024-09-15",
        log_path=tmp_path / "logs" / "custom.log",
        analysts_all=False,
        analysts=["michael_burry", "momentum_guardian"],
        extra_args=["--initial-cash", "150000"],
    )

    command = build_backtest_command(experiment, model_provider="azure")

    assert command[:5] == ["poetry", "run", "python", "src/backtester.py", "--model-provider"]
    assert "--analysts-all" not in command
    assert "--analysts" in command
    assert "michael_burry,momentum_guardian" in command
    assert command[-2:] == ["--initial-cash", "150000"]


def test_schedule_experiments_appends_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger_path = tmp_path / "log" / "backtest.csv"
    ensure_csv_header(ledger_path)

    # Seed baseline metrics to compute deltas against
    baseline_row = {
        "timestamp": "2024-09-15T00:00:00+00:00",
        "label": "baseline_run",
        "tickers": "TSLA,NVDA,GOOGL",
        "analysts": "all",
        "start_date": "2024-08-15",
        "end_date": "2024-09-15",
        "prompt_revision": "baseline_v1",
        "portfolio_return_pct": "1.00",
        "sharpe_ratio": "0.50",
        "max_drawdown_pct": "-4.00",
        "information_ratio": "0.10",
        "benchmark_return_pct": "0.80",
        "hit_rate": "NA",
        "log_path": "log/baseline.log",
        "baseline_label": "",
        "delta_portfolio_return_pct": "",
        "delta_sharpe_ratio": "",
        "delta_max_drawdown_pct": "",
        "generation_id": "",
        "arena_run_id": "",
        "genome_id": "",
        "genome_label": "",
        "genome_path": "",
        "genome_revision": "",
        "analyst_weights": "",
        "patriarch_variant": "",
        "async_mode": "",
        "model_name": "",
        "token_cost_usd": "",
        "llm_budget_used_usd": "",
        "metadata_path": "",
        "extra_tags": "",
        "timing_summary_path": "",
        "async_agent_invoke_total_seconds": "",
        "async_per_agent_total_seconds": "",
        "async_concurrency_ratio": "",
        "async_semaphore_utilization": "",
        "async_slowest_agent": "",
        "async_slowest_agent_avg_seconds": "",
        "async_concurrency_limit": "",
    }
    with ledger_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADER)
        writer.writerow(baseline_row)

    log_path = tmp_path / "logs" / "variant.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        """
Portfolio Return: 3.00%
Sharpe Ratio: 1.20
Max Drawdown: -2.50%
Information Ratio: 0.60
Benchmark Return: 0.90%
        """.strip(),
        encoding="utf-8",
    )

    experiment = Experiment(
        label="variant_run",
        tickers=["TSLA", "NVDA", "GOOGL"],
        start_date="2024-08-15",
        end_date="2024-09-15",
        log_path=log_path,
        prompt_revision="test_variant",
        baseline_label="baseline_run",
        note="unit-test",
    )

    executed: list[tuple[Sequence[str], int]] = []

    def fake_runner(command: Sequence[str], timeout: int) -> None:
        executed.append((list(command), timeout))

    monkeypatch.setattr("src.backtesting.experiment_scheduler._find_lingering_backtesters", lambda: [])

    results = schedule_experiments(
        [experiment],
        model_provider="azure",
        ledger_path=ledger_path,
        command_runner=fake_runner,
        digest_parser=parse_backtest_digest,
        timeout_override=120,
    )

    assert executed, "Command runner was not invoked"
    command, timeout = executed[0]
    assert command[:3] == ["poetry", "run", "python"]
    assert timeout == 120

    assert len(results) == 1
    result_experiment, metrics = results[0]
    assert result_experiment.label == "variant_run"
    assert metrics.portfolio_return_pct == pytest.approx(3.0)

    with ledger_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["label"] == "variant_run"
    assert rows[-1]["baseline_label"] == "baseline_run"
    assert rows[-1]["delta_portfolio_return_pct"] == "2"
    assert rows[-1]["delta_sharpe_ratio"] == "0.7"
    assert rows[-1]["delta_max_drawdown_pct"] == "1.5"
    assert rows[-1]["hit_rate"] == "unit-test"
    assert rows[-1]["timing_summary_path"] == ""
    assert rows[-1]["async_semaphore_utilization"] == ""
    assert rows[-1]["genome_path"] == ""
    assert rows[-1]["generation_id"] == ""
    assert rows[-1]["analyst_weights"] == ""
    assert rows[-1]["token_cost_usd"] == ""
    assert rows[-1]["async_mode"] == ""


def test_schedule_experiments_includes_async_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger_path = tmp_path / "log" / "backtest.csv"
    ensure_csv_header(ledger_path)

    log_path = tmp_path / "logs" / "async.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        """
Portfolio Return: 1.50%
Sharpe Ratio: 0.90
Max Drawdown: -1.25%
Information Ratio: 0.30
Benchmark Return: 0.80%
        """.strip(),
        encoding="utf-8",
    )

    experiment = Experiment(
        label="async_run",
        tickers=["TSLA", "NVDA"],
        start_date="2024-10-01",
        end_date="2024-10-04",
        log_path=log_path,
        prompt_revision="async_variant",
        analysts_all=True,
    )

    timing_dir = tmp_path / "log" / "backtest_timings"
    timing_dir.mkdir(parents=True, exist_ok=True)
    timing_path = timing_dir / "backtest_TSLA_NVDA_20241001_to_20241004_20250101T000000Z.json"
    timing_summary = {
        "agent_invoke": {"total_seconds": 10.0, "count": 2, "avg_seconds": 5.0},
        "per_agent": {
            "technical_analyst_agent": {"total_seconds": 7.0, "count": 2, "avg_seconds": 3.5, "updates": 4},
            "risk_management_agent": {"total_seconds": 1.0, "count": 2, "avg_seconds": 0.5, "updates": 3},
        },
        "async_meta": {
            "agent_invoke_total_seconds": 10.0,
            "per_agent_total_seconds": 8.0,
            "aggregate_concurrency": 0.8,
            "slowest_agent": "technical_analyst_agent",
            "slowest_agent_avg_seconds": 3.5,
            "concurrency_limit": 8,
            "semaphore_utilization": 0.1,
        },
    }
    timing_path.write_text(json.dumps(timing_summary), encoding="utf-8")

    def fake_runner(command: Sequence[str], timeout: int) -> None:  # pragma: no cover - simple stub
        pass

    monkeypatch.setattr("src.backtesting.experiment_scheduler._find_lingering_backtesters", lambda: [])

    schedule_experiments(
        [experiment],
        model_provider="azure",
        ledger_path=ledger_path,
        command_runner=fake_runner,
        digest_parser=parse_backtest_digest,
        include_async_telemetry=True,
        timing_dir=timing_dir,
    )

    with ledger_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["async_agent_invoke_total_seconds"] == "10"
    assert rows[-1]["async_per_agent_total_seconds"] == "8"
    assert rows[-1]["async_concurrency_ratio"] == "0.8"
    assert rows[-1]["async_semaphore_utilization"] == "0.1"
    assert rows[-1]["async_slowest_agent"] == "technical_analyst_agent"
    assert rows[-1]["async_slowest_agent_avg_seconds"] == "3.5"
    assert rows[-1]["async_concurrency_limit"] == "8"
    assert rows[-1]["async_mode"] == "async"
    assert rows[-1]["generation_id"] == ""
    assert rows[-1]["genome_path"] == ""

    with ledger_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    latest = rows[-1]
    assert latest["timing_summary_path"].endswith(timing_path.name)
    assert latest["async_semaphore_utilization"] == "0.1"
    assert latest["async_concurrency_ratio"] == "0.8"
    assert latest["async_slowest_agent"] == "technical_analyst_agent"


def test_schedule_experiments_enforces_diversity_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger_path = tmp_path / "log" / "backtest.csv"
    ensure_csv_header(ledger_path)

    registry = GenomeLineageRegistry(root=tmp_path / "genomes")
    genome_a = EvolutionGenome(
        genome_id="g_seed_a",
        label="seed_a",
        generation_id="gen-1",
        analysts=["ben_graham", "warren_buffett"],
        analyst_weights={"ben_graham": 0.6, "warren_buffett": 0.4},
    )
    registry.save(genome_a)

    genome_b = EvolutionGenome(
        genome_id="g_seed_b",
        label="seed_b",
        generation_id="gen-1",
        analysts=["ben_graham", "warren_buffett"],
        analyst_weights={"ben_graham": 0.61, "warren_buffett": 0.39},
    )
    registry.save(genome_b)

    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def _write_log(path: Path) -> None:
        path.write_text(
            """
Portfolio Return: 1.00%
Sharpe Ratio: 0.50
Max Drawdown: -1.00%
Information Ratio: 0.20
Benchmark Return: 0.30%
            """.strip(),
            encoding="utf-8",
        )

    log_a = log_dir / "a.log"
    log_b = log_dir / "b.log"
    _write_log(log_a)
    _write_log(log_b)

    experiments = [
        Experiment(
            label="seed_a_run",
            tickers=["TSLA"],
            start_date="2024-01-01",
            end_date="2024-01-08",
            log_path=log_a,
            prompt_revision="v1",
            genome_id="g_seed_a",
            generation_id="gen-1",
        ),
        Experiment(
            label="seed_b_run",
            tickers=["TSLA"],
            start_date="2024-01-01",
            end_date="2024-01-08",
            log_path=log_b,
            prompt_revision="v1",
            genome_id="g_seed_b",
            generation_id="gen-1",
        ),
    ]

    executed: list[tuple[Sequence[str], int]] = []

    def fake_runner(command: Sequence[str], timeout: int) -> None:
        executed.append((list(command), timeout))

    monkeypatch.setattr("src.backtesting.experiment_scheduler._find_lingering_backtesters", lambda: [])

    with pytest.raises(ExperimentRunError) as excinfo:
        schedule_experiments(
            experiments,
            model_provider="azure",
            ledger_path=ledger_path,
            command_runner=fake_runner,
            digest_parser=parse_backtest_digest,
            timeout_override=120,
            genome_registry=registry,
            diversity_guard=True,
        )

    assert "failed diversity guard" in str(excinfo.value)
    assert len(executed) == 1, "second command should not have executed"


def test_load_schedule_expands_windows(tmp_path: Path) -> None:
    log_dir = tmp_path / "arena_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    genome_growth_path = tmp_path / "growth_genome.json"
    genome_growth_path.write_text("{}", encoding="utf-8")

    config = {
        "generation_id": "gen-1",
        "ticker_bundles": [
            {"label": "value", "tickers": ["TSLA", "AAPL"], "genome_id": "g_value"},
            {
                "label": "growth",
                "tickers": ["NVDA", "GOOGL"],
                "analysts_all": False,
                "analysts": ["cathie_wood"],
                "genome_path": str(genome_growth_path),
            },
        ],
        "windows": [
            {
                "label": "week1",
                "start_date": "2024-10-01",
                "end_date": "2024-10-07",
                "note": "window-note",
            }
        ],
        "log_dir": str(log_dir),
        "diversity_guard": True,
        "analyst_overlap_threshold": 0.75,
        "weight_similarity_threshold": 0.85,
    }

    config_path = tmp_path / "schedule.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    experiments, meta = _load_schedule(config_path, now=datetime(2024, 10, 1, tzinfo=timezone.utc))

    assert len(experiments) == 2
    labels = {experiment.label for experiment in experiments}
    assert "week1_value" in labels
    assert "week1_growth" in labels
    for experiment in experiments:
        assert experiment.generation_id == "gen-1"
        assert experiment.log_path.parent == log_dir

    growth = next(exp for exp in experiments if exp.label == "week1_growth")
    assert growth.analysts_all is False
    assert growth.analysts == ["cathie_wood"]
    assert growth.genome_path == genome_growth_path

    assert meta["diversity_guard"] is True
    assert meta["analyst_overlap_threshold"] == 0.75
    assert meta["weight_similarity_threshold"] == 0.85


def test_guardrail_concurrency_violation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger_path = tmp_path / "log" / "backtest.csv"
    ensure_csv_header(ledger_path)

    experiment = Experiment(
        label="guardrail_concurrency",
        tickers=["TSLA"],
        start_date="2024-07-01",
        end_date="2024-07-08",
        log_path=tmp_path / "logs" / "guardrail.log",
        prompt_revision="v1",
    )

    def fake_runner(command: Sequence[str], timeout: int) -> None:  # pragma: no cover - simple stub
        pass

    digest = BacktestDigest(
        path=str(experiment.log_path),
        ticker="TSLA",
        metrics={
            "portfolio_return_pct": 1.0,
            "sharpe_ratio": 1.0,
            "sortino_ratio": 0.8,
            "max_drawdown_pct": -1.0,
            "information_ratio": 0.2,
            "benchmark_return_pct": 0.5,
            "turnover_rate_pct": 5.0,
        },
        analyst_votes=[],
        anomalies=[],
        trading_decision=None,
        risk=None,
    )

    def fake_digest_parser(path: Path) -> BacktestDigest:  # pragma: no cover - simple stub
        return digest

    timing_dir = tmp_path / "log" / "backtest_timings"
    timing_dir.mkdir(parents=True, exist_ok=True)
    timing_path = timing_dir / "backtest_TSLA_20240701_to_20240708_20250101T000000Z.json"
    timing_summary = {
        "async_meta": {
            "agent_invoke_total_seconds": 5,
            "per_agent_total_seconds": 4,
            "aggregate_concurrency": 0.9,
            "concurrency_limit": 8,
            "semaphore_utilization": 0.2,
            "slowest_agent": "risk_management_agent",
            "slowest_agent_avg_seconds": 2.0,
        }
    }
    timing_path.write_text(json.dumps(timing_summary), encoding="utf-8")

    monkeypatch.setenv("LLM_ASYNC_MAX_CONCURRENCY", "4")
    monkeypatch.setattr("src.backtesting.experiment_scheduler._find_lingering_backtesters", lambda: [])

    guardrail_dir = tmp_path / "guardrails"

    with pytest.raises(ExperimentRunError) as excinfo:
        schedule_experiments(
            [experiment],
            model_provider="azure",
            ledger_path=ledger_path,
            command_runner=fake_runner,
            digest_parser=fake_digest_parser,
            include_async_telemetry=True,
            timing_dir=timing_dir,
            guardrail_log_dir=guardrail_dir,
        )

    assert "Guardrail violation" in str(excinfo.value)
    events_path = guardrail_dir / "events.jsonl"
    assert events_path.exists()
    content = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert any("async_concurrency_limit" in line for line in content)

    with ledger_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 0, "ledger should remain empty when guardrail fails"


def test_guardrail_governance_violation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger_path = tmp_path / "log" / "backtest.csv"
    ensure_csv_header(ledger_path)

    log_path = tmp_path / "logs" / "governance.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("Portfolio Return: 1.0%\nSharpe Ratio: 1.1\nMax Drawdown: -1.2%", encoding="utf-8")

    experiment = Experiment(
        label="guardrail_governance",
        tickers=["TSLA"],
        start_date="2024-07-01",
        end_date="2024-07-08",
        log_path=log_path,
        prompt_revision="v1",
    )

    risk = RiskTelemetry(
        remaining_limit=-1000.0,
        constraint_notes=["Exposure violation on TSLA position"],
        overrides={"block_new_shorts": "True"},
    )
    digest = BacktestDigest(
        path=str(log_path),
        ticker="TSLA",
        metrics={
            "portfolio_return_pct": 1.0,
            "sharpe_ratio": 1.1,
            "sortino_ratio": 0.9,
            "max_drawdown_pct": -1.2,
            "information_ratio": 0.3,
            "benchmark_return_pct": 0.4,
            "turnover_rate_pct": 4.5,
        },
        analyst_votes=[],
        anomalies=[],
        trading_decision=None,
        risk=risk,
    )

    def fake_runner(command: Sequence[str], timeout: int) -> None:  # pragma: no cover - simple stub
        pass

    def fake_digest_parser(path: Path) -> BacktestDigest:  # pragma: no cover - simple stub
        return digest

    guardrail_dir = tmp_path / "guardrails"
    monkeypatch.delenv("LLM_ASYNC_MAX_CONCURRENCY", raising=False)
    monkeypatch.setattr("src.backtesting.experiment_scheduler._find_lingering_backtesters", lambda: [])

    with pytest.raises(ExperimentRunError) as excinfo:
        schedule_experiments(
            [experiment],
            model_provider="azure",
            ledger_path=ledger_path,
            command_runner=fake_runner,
            digest_parser=fake_digest_parser,
            guardrail_log_dir=guardrail_dir,
        )

    assert "Guardrail violation" in str(excinfo.value)
    events_path = guardrail_dir / "events.jsonl"
    assert events_path.exists()
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert any("governance" in line for line in lines)
    assert any("violation" in line.lower() for line in lines)

    with ledger_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 0


def test_guardrail_concurrency_ratio_violation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ledger_path = tmp_path / "log" / "backtest.csv"
    ensure_csv_header(ledger_path)

    experiment = Experiment(
        label="ratio_violation",
        tickers=["TSLA"],
        start_date="2024-07-01",
        end_date="2024-07-08",
        log_path=tmp_path / "logs" / "ratio.log",
        prompt_revision="v1",
    )

    digest = BacktestDigest(
        path=str(experiment.log_path),
        ticker="TSLA",
        metrics={
            "portfolio_return_pct": 0.1,
            "sharpe_ratio": 0.2,
            "sortino_ratio": 0.2,
            "max_drawdown_pct": -0.1,
            "information_ratio": 0.0,
            "benchmark_return_pct": 0.05,
            "turnover_rate_pct": 1.0,
        },
        analyst_votes=[],
        anomalies=[],
        trading_decision=None,
        risk=None,
    )

    def fake_runner(command: Sequence[str], timeout: int) -> None:  # pragma: no cover - simple stub
        pass

    def fake_digest_parser(path: Path) -> BacktestDigest:  # pragma: no cover - simple stub
        return digest

    timing_dir = tmp_path / "log" / "backtest_timings"
    timing_dir.mkdir(parents=True, exist_ok=True)
    timing_payload = {
        "async_meta": {
            "agent_invoke_total_seconds": 10.0,
            "per_agent_total_seconds": 25.0,
            "aggregate_concurrency": "17",
            "concurrency_limit": "8",
            "semaphore_utilization": 0.5,
            "slowest_agent": "risk_management_agent",
            "slowest_agent_avg_seconds": 3.0,
        }
    }
    summary_path = timing_dir / "backtest_TSLA_20240701_to_20240708_20250101T000000Z.json"
    summary_path.write_text(json.dumps(timing_payload), encoding="utf-8")

    guardrail_dir = tmp_path / "guardrails"
    monkeypatch.delenv("LLM_ASYNC_MAX_CONCURRENCY", raising=False)
    monkeypatch.setattr("src.backtesting.experiment_scheduler._find_lingering_backtesters", lambda: [])

    with pytest.raises(ExperimentRunError) as excinfo:
        schedule_experiments(
            [experiment],
            model_provider="azure",
            ledger_path=ledger_path,
            command_runner=fake_runner,
            digest_parser=fake_digest_parser,
            include_async_telemetry=True,
            timing_dir=timing_dir,
            guardrail_log_dir=guardrail_dir,
        )

    assert "Guardrail violation" in str(excinfo.value)
    events_path = guardrail_dir / "events.jsonl"
    assert events_path.exists()
    content = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert any("async_concurrency_ratio" in line for line in content)

    with ledger_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 0


def test_schedule_experiments_emits_reports(tmp_path: Path) -> None:
    ledger_path = tmp_path / "log" / "backtest.csv"
    ensure_csv_header(ledger_path)

    log_path = tmp_path / "logs" / "report.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        """
Portfolio Return: 2.00%
Sharpe Ratio: 0.80
Max Drawdown: -1.00%
Information Ratio: 0.40
Benchmark Return: 0.50%
        """.strip(),
        encoding="utf-8",
    )

    experiment = Experiment(
        label="gen1_value",
        tickers=["TSLA", "AAPL"],
        start_date="2024-10-01",
        end_date="2024-10-07",
        log_path=log_path,
        prompt_revision="arena_v1",
        generation_id="gen-1",
        genome_id="genome_value",
    )

    arena_dir = tmp_path / "arena_reports"

    def fake_runner(command: Sequence[str], timeout: int) -> None:
        pass

    schedule_experiments(
        [experiment],
        model_provider="azure",
        ledger_path=ledger_path,
        command_runner=fake_runner,
        digest_parser=parse_backtest_digest,
        timeout_override=120,
        arena_report_dir=arena_dir,
    )

    generation_dir = arena_dir / "gen_1" / "genome_value"
    json_path = generation_dir / "gen1_value.json"
    md_path = generation_dir / "gen1_value.md"
    assert json_path.exists()
    assert md_path.exists()

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["label"] == "gen1_value"
    assert summary["metrics"]["portfolio_return_pct"] == 2.0
    assert summary["deltas"]["portfolio_return_pct"] is None

    markdown = md_path.read_text(encoding="utf-8")
    assert "# gen1_value" in markdown
