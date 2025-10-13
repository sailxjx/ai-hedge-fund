from __future__ import annotations

import csv
import json
import sys
import time
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
)
from src.tools.llm_backtest_digest import parse_backtest_digest


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


def test_schedule_experiments_appends_ledger(tmp_path: Path) -> None:
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
        "timing_summary_path": "",
        "async_agent_invoke_total_seconds": "",
        "async_per_agent_total_seconds": "",
        "async_concurrency_ratio": "",
        "async_semaphore_utilization": "",
        "async_slowest_agent": "",
        "async_slowest_agent_avg_seconds": "",
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


def test_schedule_experiments_includes_async_telemetry(tmp_path: Path) -> None:
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
    latest = rows[-1]
    assert latest["timing_summary_path"].endswith(timing_path.name)
    assert latest["async_semaphore_utilization"] == "0.1"
    assert latest["async_concurrency_ratio"] == "0.8"
    assert latest["async_slowest_agent"] == "technical_analyst_agent"
