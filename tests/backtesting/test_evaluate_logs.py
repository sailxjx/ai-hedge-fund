import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.backtesting import evaluate_logs

SAMPLE_LOG_TEXT = """Portfolio Return: +5.00%
Sharpe Ratio: 1.20
Sortino Ratio: 1.80
Information Ratio: 0.75
Max Drawdown: -3.50%
Benchmark Return: +2.40%
Turnover Rate: 12.50%
force_cover triggered
target_long_shares -> 125
prob_up: 0.60
prob_up: 0.30
Total Return: -1.00%
"""


@pytest.fixture
def sample_log(tmp_path: Path) -> Path:
    path = tmp_path / "sample_backtest.log"
    path.write_text(SAMPLE_LOG_TEXT, encoding="utf-8")
    return path


def test_parse_log_extracts_metrics(sample_log: Path):
    metrics = evaluate_logs._parse_log("sample", sample_log)
    assert metrics.label == "sample"
    assert metrics.portfolio_return_pct == 5.0
    assert metrics.turnover_rate_pct == 12.5
    assert metrics.sharpe_ratio == 1.2
    assert metrics.sortino_ratio == 1.8
    assert metrics.information_ratio == 0.75
    assert metrics.max_drawdown_pct == -3.5
    assert metrics.benchmark_return_pct == 2.4
    assert metrics.force_cover_events == 1
    assert metrics.target_long_events == 1
    assert metrics.prob_up_calls == 1  # two prob_up entries minus one negative return mention


def test_cli_generates_outputs(tmp_path, monkeypatch, sample_log):
    csv_path = tmp_path / "summary.csv"
    json_path = tmp_path / "summary.json"
    evaluate_logs.main(
        [
            "--logs",
            f"sample={sample_log}",
            "--output",
            str(csv_path),
            "--json",
            str(json_path),
        ]
    )
    assert csv_path.exists()
    assert json_path.exists()
    csv_content = csv_path.read_text().strip().splitlines()
    assert len(csv_content) == 2  # header + row
    json_payload = json.loads(json_path.read_text())
    assert json_payload[0]["label"] == "sample"
    assert json_payload[0]["information_ratio"] == 0.75
    assert json_payload[0]["benchmark_return_pct"] == 2.4
    assert json_payload[0]["turnover_rate_pct"] == 12.5


def test_parse_log_includes_override_counts(tmp_path):
    log_path = tmp_path / "plain.log"
    log_path.write_text("Portfolio Return: +1.00%\n", encoding="utf-8")

    override_entries = [
        {"risk_snapshot": {"overrides": {"force_cover_qty": 3, "target_long_shares": 5}}},
        {"risk_snapshot": {"overrides": {"force_cover_qty": 0}}},
        {"risk_snapshot": {"overrides": {"target_long_shares": 0}}},
        "not-json",
    ]
    override_path = tmp_path / "risk_overrides.jsonl"
    override_path.write_text(
        "\n".join(entry if isinstance(entry, str) else json.dumps(entry) for entry in override_entries) + "\n",
        encoding="utf-8",
    )

    metrics = evaluate_logs._parse_log("run", log_path, override_path)
    assert metrics.force_cover_events == 1
    assert metrics.target_long_events == 2


def test_cli_with_override_logs(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        "\n".join(
            [
                "Portfolio Return: +2.50%",
                "Sharpe Ratio: 0.85",
                "Sortino Ratio: 1.20",
                "Max Drawdown: -1.10%",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    override_path = tmp_path / "run_overrides.jsonl"
    override_payload = {"risk_snapshot": {"overrides": {"force_cover_qty": 4, "target_long_shares": 12}}}
    override_path.write_text(json.dumps(override_payload) + "\n", encoding="utf-8")

    csv_path = tmp_path / "summary.csv"
    evaluate_logs.main(
        [
            "--logs",
            f"run={log_path}",
            "--overrides",
            f"run={override_path}",
            "--output",
            str(csv_path),
        ]
    )

    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["label"] == "run"
    assert rows[0]["force_cover_events"] == "1"
    assert rows[0]["target_long_events"] == "1"
    assert rows[0]["information_ratio"] == ""
    assert rows[0]["benchmark_return_pct"] == ""


def test_standalone_execution():
    script_path = Path("src/backtesting/evaluate_logs.py")
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Summarise backtest log metrics." in result.stdout
