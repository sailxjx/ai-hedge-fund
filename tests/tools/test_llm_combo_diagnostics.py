from __future__ import annotations

import json
from pathlib import Path

from src.tools.llm_backtest_digest import parse_backtest_digest
from src.tools.llm_combo_diagnostics import analyse_digest, main

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_LOG = REPO_ROOT / "tests" / "fixtures" / "logs" / "sample_backtest_agents.txt"


def test_analyse_digest_flags_conflicts() -> None:
    digest = parse_backtest_digest(FIXTURE_LOG)
    payload = analyse_digest(digest)

    issues = payload["issues"]
    assert "bias_conflicts" in issues
    assert "sentinel_disagreements" in issues
    assert "risk_gating_bottlenecks" in issues
    if issues.get("hedge_conflicts"):
        assert any("Decision" in item["description"] for item in issues["hedge_conflicts"])

    summary = payload["summary"].lower()
    assert "bullish cohort" in summary
    assert "risk guardrails" in summary


def test_main_writes_default_output(tmp_path) -> None:
    output_path = tmp_path / "diagnostics.json"

    # Force CLI to use temporary output path
    args = [
        "--logs",
        str(FIXTURE_LOG),
        "--output",
        str(output_path),
    ]
    main(args)

    diagnostics = json.loads(output_path.read_text())
    assert diagnostics[0]["issues"]["bias_conflicts"]
    assert diagnostics[0]["summary"]
