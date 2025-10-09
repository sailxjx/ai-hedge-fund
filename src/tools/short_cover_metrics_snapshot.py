"""Generate day-by-day metrics for the short-cover classifier agent."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.agents.short_cover_classifier import short_cover_classifier_agent
from src.graph.state import AgentState


@dataclass
class Snapshot:
    date: str
    signal: str
    confidence: int
    reasoning: str
    metrics: dict[str, Any]

    def to_flat_dict(self) -> dict[str, Any]:
        flat = {
            "date": self.date,
            "signal": self.signal,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }
        flat.update(self.metrics)
        return flat


def _build_state(
    ticker: str,
    start_date: str,
    end_date: str,
    short_shares: int,
) -> AgentState:
    return {
        "messages": [],
        "data": {
            "tickers": [ticker],
            "portfolio": {
                "cash": 100_000.0,
                "margin_requirement": 0.5,
                "margin_used": 0.0,
                "equity": 100_000.0,
                "positions": {
                    ticker: {
                        "long": 0,
                        "short": short_shares,
                        "long_cost_basis": 0.0,
                        "short_cost_basis": 0.0,
                    }
                },
            },
            "start_date": start_date,
            "end_date": end_date,
            "analyst_signals": {},
        },
        "metadata": {"show_reasoning": False},
    }


def _capture_snapshot(state: AgentState, ticker: str) -> Snapshot | None:
    result = short_cover_classifier_agent(state)
    analysis = result["data"].get("analyst_signals", {}).get("short_cover_classifier_agent", {})
    payload = analysis.get(ticker)
    if payload is None:
        return None

    metrics = payload.get("metrics", {})
    return Snapshot(
        date=state["data"]["end_date"],
        signal=str(payload.get("signal", "neutral")),
        confidence=int(payload.get("confidence", 0)),
        reasoning=str(payload.get("reasoning", "")),
        metrics={k: metrics.get(k) for k in sorted(metrics.keys())},
    )


def _generate_snapshots(
    ticker: str,
    start_date: str,
    end_date: str,
    short_shares: int,
) -> list[Snapshot]:
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")
    snapshots: list[Snapshot] = []

    for current_date in date_range:
        state = _build_state(
            ticker=ticker,
            start_date=start_date,
            end_date=current_date.strftime("%Y-%m-%d"),
            short_shares=short_shares,
        )
        snapshot = _capture_snapshot(state, ticker)
        if snapshot:
            snapshots.append(snapshot)

    return snapshots


def _to_dataframe(snapshots: list[Snapshot]) -> pd.DataFrame:
    if not snapshots:
        return pd.DataFrame()

    rows = [snapshot.to_flat_dict() for snapshot in snapshots]
    df = pd.DataFrame(rows).set_index("date").sort_index()
    numeric_cols = [col for col in df.columns if df[col].dtype != "object"]
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].apply(lambda col: pd.to_numeric(col, errors="coerce"))

    if {"probability", "threshold"}.issubset(df.columns):
        df["probability_gap"] = df["probability"] - df["threshold"]
    if {"probability", "raw_probability"}.issubset(df.columns):
        df["boost_delta_check"] = df["probability"] - df["raw_probability"]
    if {"improvement_threshold", "improvement_threshold_effective"}.issubset(df.columns):
        df["improvement_threshold_reduction"] = (
            df["improvement_threshold"] - df["improvement_threshold_effective"]
        )
    if {"improvement", "improvement_threshold_effective"}.issubset(df.columns):
        df["improvement_margin"] = df["improvement"] - df["improvement_threshold_effective"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--short-shares", type=int, default=0)
    parser.add_argument(
        "--output-prefix",
        default="log/analysis/short_cover_metrics",
        help="Prefix (without extension) for output files.",
    )
    parser.add_argument("--json", action="store_true", help="Write a JSON file alongside CSV output.")
    args = parser.parse_args()

    snapshots = _generate_snapshots(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        short_shares=args.short_shares,
    )

    df = _to_dataframe(snapshots)
    if df.empty:
        print("No snapshots generated; verify parameters and data availability.")
        return

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output_prefix.with_suffix(".csv")
    df.to_csv(csv_path)
    print(f"Saved CSV to {csv_path}")

    if args.json:
        json_path = output_prefix.with_suffix(".json")
        json_payload = [snapshot.to_flat_dict() for snapshot in snapshots]
        json_path.write_text(json.dumps(json_payload, indent=2))
        print(f"Saved JSON to {json_path}")


if __name__ == "__main__":
    main()
