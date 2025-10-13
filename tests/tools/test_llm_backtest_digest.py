from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools.llm_backtest_digest import main, parse_backtest_digest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_LOG = REPO_ROOT / "tests" / "fixtures" / "logs" / "sample_backtest_agents.txt"


def test_parse_backtest_digest_extracts_metrics_votes_and_anomalies() -> None:
    digest = parse_backtest_digest(FIXTURE_LOG)

    assert digest.metrics["portfolio_return_pct"] == -3.5
    assert digest.metrics["sharpe_ratio"] == -2.38
    assert digest.metrics["max_drawdown_pct"] == -5.01
    assert digest.metrics["benchmark_return_pct"] == 1.25
    assert digest.metrics["total_return_pct"] == -3.1

    assert digest.ticker == "TSLA"
    assert len(digest.analyst_votes) == 5

    ben_graham = digest.analyst_votes[0]
    assert ben_graham.agent == "Ben Graham"
    assert ben_graham.signal == "BEARISH"
    assert ben_graham.confidence_pct == 85.0
    assert "valuation downside" in ben_graham.reasoning

    cathie_wood = digest.analyst_votes[1]
    assert cathie_wood.agent == "Cathie Wood"
    assert cathie_wood.signal == "BULLISH"
    assert cathie_wood.confidence_pct == 60.0

    momentum_guardian = next(vote for vote in digest.analyst_votes if vote.agent == "Momentum Guardian")
    assert momentum_guardian.signal == "BULLISH_MOMENTUM"
    assert momentum_guardian.confidence_pct == 92.0

    assert digest.risk is not None
    assert digest.risk.remaining_limit == 13430.0
    assert digest.risk.overrides["block_new_shorts"] == "True"
    assert "RangeRecovery forcing covers" in digest.risk.constraint_notes

    assert digest.trading_decision is not None
    assert digest.trading_decision.action == "HOLD"
    assert digest.trading_decision.quantity == 0
    assert digest.trading_decision.confidence_pct == 55.0
    assert "stay flat" in (digest.trading_decision.reasoning or "")

    anomaly_types = {anomaly.type for anomaly in digest.anomalies}
    assert "data_fetch" in anomaly_types
    assert "short_cover_threshold" in anomaly_types

    short_cover = next(anomaly for anomaly in digest.anomalies if anomaly.type == "short_cover_threshold")
    assert short_cover.metadata is not None
    assert short_cover.metadata["ticker"] == "TSLA"
    assert short_cover.metadata["decision"] == "NEUTRAL"


def test_main_writes_json(tmp_path) -> None:
    output_path = tmp_path / "digest.json"
    main(["--logs", str(FIXTURE_LOG), "--output", str(output_path)])

    payload = json.loads(output_path.read_text())
    assert isinstance(payload, list)
    first = payload[0]
    assert first["metrics"]["portfolio_return_pct"] == -3.5
    assert first["analyst_votes"][0]["agent"] == "Ben Graham"
    assert first["risk"]["overrides"]["block_new_shorts"] == "True"
    assert first["trading_decision"]["action"] == "HOLD"


def test_parse_backtest_digest_handles_persona_short_cover(tmp_path) -> None:
    log_text = "⋯ Short Cover Classifier[TSLA] Persona squeeze view 23.5% (thr 24.6%) →        \n" "BIAS_LONG\n"
    log_path = tmp_path / "persona_short_cover.log"
    log_path.write_text(log_text)

    digest = parse_backtest_digest(log_path)
    anomalies = [a for a in digest.anomalies if a.type == "short_cover_threshold"]

    assert anomalies, "Expected persona short-cover event to be captured as an anomaly"
    metadata = anomalies[0].metadata or {}
    assert metadata.get("ticker") == "TSLA"
    assert pytest.approx(metadata.get("prob_squeeze_pct"), rel=1e-6) == 23.5
    assert pytest.approx(metadata.get("threshold_pct"), rel=1e-6) == 24.6
    assert metadata.get("decision") == "BIAS_LONG"
