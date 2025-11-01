from __future__ import annotations

import sys
from pathlib import Path

from src.tools.agent_iteration import (
    build_agent_file_map,
    determine_guard_paths,
    generate_hypotheses,
    GuardedFiles,
    IterationHypothesis,
    run_command,
)


def test_generate_hypotheses_extracts_agents_and_recommendations() -> None:
    diagnostics = [
        {
            "issues": {
                "sentinel_disagreements": [
                    {
                        "guardrails": ["Range Recovery Sentinel"],
                        "other_direction": ["Momentum Guardian"],
                        "description": "Guardrails conflicting with directionals",
                    }
                ],
                "risk_gating_bottlenecks": [
                    {
                        "overrides": {"block_new_shorts": "True"},
                        "constraint_notes": ["RangeRecovery keeps shorts blocked"],
                        "description": "Risk override blocking shorts",
                    }
                ],
            }
        }
    ]

    hypotheses = generate_hypotheses(diagnostics)
    assert len(hypotheses) == 2
    sentinel = next(item for item in hypotheses if item.issue == "sentinel_disagreements")
    assert "Range Recovery Sentinel" in sentinel.agents
    assert "Momentum Guardian" in sentinel.agents
    assert "guardrail" in sentinel.recommendation.lower()

    risk = next(item for item in hypotheses if item.issue == "risk_gating_bottlenecks")
    assert "Risk Management" in risk.agents
    assert "short" in risk.recommendation.lower()


def test_guarded_files_reverts_changes(tmp_path: Path) -> None:
    target = tmp_path / "agent.py"
    target.write_text("original")
    guard = GuardedFiles([target])
    guard.snapshot()
    target.write_text("mutated")
    guard.revert()
    assert target.read_text() == "original"

    new_file = tmp_path / "new_agent.py"
    guard_new = GuardedFiles([new_file])
    guard_new.snapshot()
    new_file.write_text("created")
    guard_new.revert()
    assert not new_file.exists()


def test_build_agent_file_map_contains_known_agent() -> None:
    mapping = build_agent_file_map()
    key = "rangerecoverysentinel"
    assert key in mapping
    assert mapping[key].name.endswith(".py")


def test_determine_guard_paths_includes_extra(tmp_path: Path) -> None:
    mapping = build_agent_file_map()
    hypotheses = [
        IterationHypothesis(
            issue="sentinel_disagreements",
            agents=["Range Recovery Sentinel"],
            description="conflict",
            recommendation="adjust guard",
        )
    ]
    extra_file = tmp_path / "custom.py"
    extra_file.write_text("pass\n")
    guard_paths = determine_guard_paths(hypotheses, mapping, [extra_file])
    assert mapping["rangerecoverysentinel"].resolve() in guard_paths
    assert extra_file.resolve() in guard_paths


def test_run_command_reports_timeout() -> None:
    result = run_command(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(0.2)",
        ],
        timeout=0.05,
    )
    assert result.status == "timeout"
