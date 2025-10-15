from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools.arena_selection import (
    SelectionCandidate,
    SelectionRules,
    apply_recommendations,
    discover_candidates,
    load_selection_rules,
    persist_report,
)


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.fixture()
def selection_rules(tmp_path: Path) -> Path:
    rules = """
version: 1
gates:
  min_sharpe: 0.4
  min_return_pct: 0.5
  max_drawdown_pct: -5.0
  max_turnover_pct: 60.0
  max_token_cost_usd: 30.0
  max_async_concurrency_ratio: 1.0
  min_weight_entropy: 0.5
fitness:
  weights:
    sharpe_ratio: 1.0
    portfolio_return_pct: 0.4
    sortino_ratio: 0.3
  penalties:
    max_drawdown_pct: 0.1
    turnover_rate_pct: 0.02
    token_cost_usd: 0.02
    async_concurrency_ratio: 0.05
  bonus_tags:
    novelty: 0.2
promotion:
  promote_threshold: 0.2
  hold_threshold: 0.0
  top_n: 1
"""
    path = tmp_path / "rules.yaml"
    path.write_text(rules.strip(), encoding="utf-8")
    return path


def test_arena_selection_promotes_best_candidate(tmp_path: Path, selection_rules: Path) -> None:
    arena_dir = tmp_path / "arena"
    summary_good = {
        "timestamp": "2025-10-13T12:00:00Z",
        "label": "gen1_value",
        "generation_id": "gen-1",
        "arena_run_id": "arena-001",
        "genome_id": "genome_value",
        "genome_label": "Value Stack",
        "tickers": ["TSLA", "NVDA"],
        "start_date": "2025-09-01",
        "end_date": "2025-09-08",
        "prompt_revision": "v1",
        "baseline_label": None,
        "note": None,
        "log_path": "log/backtest_gen1_value.log",
        "metrics": {
            "portfolio_return_pct": 2.4,
            "sharpe_ratio": 0.8,
            "sortino_ratio": 0.6,
            "max_drawdown_pct": -3.2,
            "information_ratio": 0.3,
            "benchmark_return_pct": 1.1,
            "turnover_rate_pct": 35.0,
        },
        "deltas": {"portfolio_return_pct": 1.2, "sharpe_ratio": 0.3, "max_drawdown_pct": 1.1},
        "async_telemetry": {"async_concurrency_ratio": 0.8},
        "timing_summary_path": None,
        "patriarch_variant": "weighted_majority",
        "analyst_weights": {"ben_graham": 0.6, "warren_buffett": 0.4},
        "metadata_path": None,
        "extra_tags": ["novelty", "diversified"],
        "token_cost_usd": 18.5,
        "llm_budget_used_usd": 25.0,
        "async_mode": "async",
        "model_name": "gpt-5",
    }
    summary_bad = {
        "timestamp": "2025-10-13T12:05:00Z",
        "label": "gen1_mean",
        "generation_id": "gen-1",
        "arena_run_id": "arena-001",
        "genome_id": "genome_mean",
        "genome_label": "Mean Reversion",
        "tickers": ["TSLA", "AAPL"],
        "start_date": "2025-09-01",
        "end_date": "2025-09-08",
        "prompt_revision": "v1",
        "baseline_label": None,
        "note": None,
        "log_path": "log/backtest_gen1_mean.log",
        "metrics": {
            "portfolio_return_pct": 0.3,
            "sharpe_ratio": 0.2,
            "sortino_ratio": 0.1,
            "max_drawdown_pct": -6.4,
            "information_ratio": 0.1,
            "benchmark_return_pct": 0.2,
            "turnover_rate_pct": 45.0,
        },
        "deltas": {"portfolio_return_pct": -0.2, "sharpe_ratio": -0.1, "max_drawdown_pct": -1.0},
        "async_telemetry": {"async_concurrency_ratio": 0.9},
        "timing_summary_path": None,
        "patriarch_variant": "weighted_majority",
        "analyst_weights": {"stat_mean": 1.0},
        "metadata_path": None,
        "extra_tags": [],
        "token_cost_usd": 22.0,
        "llm_budget_used_usd": 30.0,
        "async_mode": "async",
        "model_name": "gpt-5",
    }

    _write_summary(arena_dir / "gen-1" / "genome_value" / "gen1_value.json", summary_good)
    _write_summary(arena_dir / "gen-1" / "genome_mean" / "gen1_mean.json", summary_bad)

    rules = load_selection_rules(selection_rules)
    candidates = discover_candidates(arena_dir, "gen-1", rules)
    candidates.sort(key=lambda c: c.label)

    assert len(candidates) == 2
    apply_recommendations(candidates, rules)

    promoted = next(c for c in candidates if c.label == "gen1_value")
    rejected = next(c for c in candidates if c.label == "gen1_mean")

    assert promoted.recommendation == "promote"
    assert promoted.fitness is not None and promoted.fitness > 0
    assert not promoted.gating_violations
    assert rejected.recommendation == "reject"
    assert rejected.gating_violations, "Expected gating violations for low Sharpe candidate"

    output_dir = tmp_path / "selection"
    report_path = persist_report(
        output_dir=output_dir,
        candidates=candidates,
        rules_path=selection_rules,
        rules=rules,
        generation="gen-1",
    )

    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["generation_id"] == "gen-1"
    assert len(report["candidates"]) == 2
    recs = {item["label"]: item["recommendation"] for item in report["candidates"]}
    assert recs["gen1_value"] == "promote"
    assert recs["gen1_mean"] == "reject"

    md_matches = list(output_dir.glob("selection_gen-1_*.md"))
    assert md_matches, "Markdown selection report not generated"
