"""Derive failure taxonomies from LLM backtest digests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

from src.tools.llm_backtest_digest import AnalystVote, BacktestDigest, parse_backtest_digest

CORE_ANALYSTS = {
    "Aswath Damodaran",
    "Ben Graham",
    "Bill Ackman",
    "Cathie Wood",
    "Charlie Munger",
    "Michael Burry",
    "Mohnish Pabrai",
    "Peter Lynch",
    "Phil Fisher",
    "Rakesh Jhunjhunwala",
    "Stanley Druckenmiller",
    "Warren Buffett",
    "Fundamentals Analyst",
    "Sentiment Analyst",
    "Valuation Analyst",
    "Growth Momentum",
    "Stat Mean Reversion",
}

SENTINEL_AGENTS = {
    "Momentum Guardian",
    "Trend Regime",
    "Short Squeeze Guardian",
    "Range Recovery Sentinel",
    "Breakout Cover Sentinel",
    "Downside Flow Sentinel",
    "Macro Volatility Sentinel",
    "Crash Short Allocator",
    "Risk Override Amplifier",
    "Short Cover Classifier",
    "Stop Loss Guardian",
    "Event Catalyst",
    "Risk Management",
}

BULLISH_KEYWORDS = {"bull", "long", "buy", "accumulate", "add"}
BEARISH_KEYWORDS = {"bear", "short", "sell", "trim", "reduce", "crash"}
GUARDRAIL_KEYWORDS = {"guardrail", "avoid_short", "avoid shorts", "block", "cover", "stand-down", "force cover"}
NEUTRAL_KEYWORDS = {"neutral", "hold", "flat", "calm"}

DEFAULT_OUTPUT_PATH = Path("log/analysis/latest_llm_combo_diagnostics.json")


def _normalise_signal(signal: str | None) -> str:
    if not signal:
        return "unknown"
    return signal.strip().lower()


def _signal_polarity(signal: str | None) -> str:
    sig = _normalise_signal(signal)
    if sig == "unknown":
        return "unknown"
    if any(keyword in sig for keyword in GUARDRAIL_KEYWORDS):
        return "guardrail"
    if any(keyword in sig for keyword in BULLISH_KEYWORDS):
        return "bullish"
    if any(keyword in sig for keyword in BEARISH_KEYWORDS):
        return "bearish"
    if any(keyword in sig for keyword in NEUTRAL_KEYWORDS):
        return "neutral"
    return "unknown"


def _vote_confident(vote: AnalystVote, threshold: float) -> bool:
    if vote.confidence_pct is None:
        return True
    return vote.confidence_pct >= threshold


def _extract_votes(votes: Iterable[AnalystVote], agents: set[str], threshold: float) -> dict[str, list[AnalystVote]]:
    grouped: dict[str, list[AnalystVote]] = {"bullish": [], "bearish": [], "guardrail": [], "neutral": [], "unknown": []}
    for vote in votes:
        if vote.agent not in agents:
            continue
        polarity = _signal_polarity(vote.signal)
        if not _vote_confident(vote, threshold) and polarity not in {"guardrail", "unknown"}:
            continue
        grouped.setdefault(polarity, []).append(vote)
    return grouped


def _decision_direction(digest: BacktestDigest) -> str:
    if digest.trading_decision is None or digest.trading_decision.action is None:
        return "unknown"
    action = digest.trading_decision.action.strip().lower()
    if action in {"buy", "long"}:
        return "bullish"
    if action in {"sell", "short"}:
        return "bearish"
    if action in {"hold", "flat", "none"}:
        return "neutral"
    return "unknown"


def _format_agent_list(votes: Iterable[AnalystVote]) -> str:
    agents = [vote.agent for vote in votes]
    if not agents:
        return ""
    if len(agents) == 1:
        return agents[0]
    return ", ".join(agents[:2]) + ("…" if len(agents) > 2 else "")


def _build_bias_conflicts(core_votes: dict[str, list[AnalystVote]]) -> list[dict[str, object]]:
    bullish = core_votes.get("bullish", [])
    bearish = core_votes.get("bearish", [])
    if not bullish or not bearish:
        return []
    return [
        {
            "bullish_agents": [vote.agent for vote in bullish],
            "bearish_agents": [vote.agent for vote in bearish],
            "bullish_confidence": [vote.confidence_pct for vote in bullish],
            "bearish_confidence": [vote.confidence_pct for vote in bearish],
            "description": f"Bullish cohort ({_format_agent_list(bullish)}) vs bearish cohort ({_format_agent_list(bearish)})",
        }
    ]


def _build_sentinel_disagreements(sentinel_votes: dict[str, list[AnalystVote]]) -> list[dict[str, object]]:
    bullish = sentinel_votes.get("bullish", [])
    bearish = sentinel_votes.get("bearish", [])
    guardrail = sentinel_votes.get("guardrail", [])
    findings: list[dict[str, object]] = []
    if bullish and bearish:
        findings.append(
            {
                "bullish_agents": [vote.agent for vote in bullish],
                "bearish_agents": [vote.agent for vote in bearish],
                "description": f"Sentinel split between {_format_agent_list(bullish)} and {_format_agent_list(bearish)}",
            }
        )
    if guardrail and (bullish or bearish):
        findings.append(
            {
                "guardrails": [vote.agent for vote in guardrail],
                "other_direction": [vote.agent for vote in bullish or bearish],
                "description": f"Guardrails ({_format_agent_list(guardrail)}) conflicting with directional sentinel votes",
            }
        )
    return findings


def _build_hedge_conflicts(sentinel_votes: dict[str, list[AnalystVote]], decision_direction: str) -> list[dict[str, object]]:
    if decision_direction not in {"bullish", "bearish"}:
        return []
    bullish = sentinel_votes.get("bullish", [])
    bearish = sentinel_votes.get("bearish", [])
    polarity_counts = Counter({"bullish": len(bullish), "bearish": len(bearish)})
    if polarity_counts["bullish"] == polarity_counts["bearish"] == 0:
        return []
    majority = "bullish" if polarity_counts["bullish"] >= polarity_counts["bearish"] else "bearish"
    if majority == decision_direction:
        return []
    conflicted_agents = bullish if decision_direction == "bearish" else bearish
    return [
        {
            "decision_direction": decision_direction,
            "sentinel_majority": majority,
            "conflicted_agents": [vote.agent for vote in conflicted_agents],
            "description": f"Decision {decision_direction} while sentinel majority leans {majority} ({_format_agent_list(conflicted_agents)})",
        }
    ]


def _risk_blocks_shorts(digest: BacktestDigest) -> bool:
    if digest.risk is None:
        return False
    overrides = {key.lower(): value.lower() for key, value in digest.risk.overrides.items()}
    if overrides.get("block_new_shorts") in {"true", "yes", "1"}:
        return True
    target = overrides.get("target_short_shares")
    if target and target.startswith("0"):
        return True
    preferred = overrides.get("preferred_direction")
    if preferred and "avoid_short" in preferred:
        return True
    notes = digest.risk.constraint_notes
    return any("block" in note.lower() and "short" in note.lower() for note in notes)


def _build_risk_bottlenecks(digest: BacktestDigest, core_votes: dict[str, list[AnalystVote]], sentinel_votes: dict[str, list[AnalystVote]]) -> list[dict[str, object]]:
    if not _risk_blocks_shorts(digest):
        return []
    bearish_pressure = bool(core_votes.get("bearish")) or bool(sentinel_votes.get("bearish"))
    if not bearish_pressure:
        return []
    overrides = digest.risk.overrides if digest.risk else {}
    return [
        {
            "overrides": overrides,
            "constraint_notes": digest.risk.constraint_notes if digest.risk else [],
            "description": "Risk overrides blocking shorts despite bearish signals",
        }
    ]


def analyse_digest(digest: BacktestDigest) -> dict[str, object]:
    core_votes = _extract_votes(digest.analyst_votes, CORE_ANALYSTS, threshold=60.0)
    sentinel_votes = _extract_votes(digest.analyst_votes, SENTINEL_AGENTS, threshold=55.0)

    issues: dict[str, list[dict[str, object]]] = {}

    bias_conflicts = _build_bias_conflicts(core_votes)
    if bias_conflicts:
        issues["bias_conflicts"] = bias_conflicts

    sentinel_disagreements = _build_sentinel_disagreements(sentinel_votes)
    if sentinel_disagreements:
        issues["sentinel_disagreements"] = sentinel_disagreements

    decision_direction = _decision_direction(digest)
    hedge_conflicts = _build_hedge_conflicts(sentinel_votes, decision_direction)
    if hedge_conflicts:
        issues["hedge_conflicts"] = hedge_conflicts

    risk_bottlenecks = _build_risk_bottlenecks(digest, core_votes, sentinel_votes)
    if risk_bottlenecks:
        issues["risk_gating_bottlenecks"] = risk_bottlenecks

    summary_parts: list[str] = []
    if bias_conflicts:
        summary_parts.append(bias_conflicts[0]["description"])
    if sentinel_disagreements:
        summary_parts.append(sentinel_disagreements[0]["description"])
    if hedge_conflicts:
        summary_parts.append(hedge_conflicts[0]["description"])
    if risk_bottlenecks:
        summary_parts.append("Risk guardrails blocking bearish execution")

    summary = "; ".join(summary_parts) if summary_parts else "No critical conflicts detected"

    payload = {
        "path": digest.path,
        "ticker": digest.ticker,
        "metrics": digest.metrics,
        "issues": issues,
        "trading_decision": asdict(digest.trading_decision) if digest.trading_decision else None,
        "risk": asdict(digest.risk) if digest.risk else None,
        "summary": summary,
    }
    return payload


def analyse_logs(paths: Sequence[Path]) -> list[dict[str, object]]:
    return [analyse_digest(parse_backtest_digest(path)) for path in paths]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate failure taxonomy diagnostics from LLM backtest logs.")
    parser.add_argument("--logs", nargs="+", required=True, help="Backtest log files to analyse.")
    parser.add_argument("--output", help="Path to write diagnostics JSON (default: log/analysis/latest_llm_combo_diagnostics.json)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    log_paths = [Path(path) for path in args.logs]
    diagnostics = analyse_logs(log_paths)

    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    for entry in diagnostics:
        print(f"{entry['path']}: {entry['summary']}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
