"""Utility for summarising risk override compliance issues in backtest audits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

ISSUE_ALIASES: dict[str, str] = {
    "preferred short": "direction_conflict",
    "preferred long": "direction_conflict",
    "expected": "position_target_violation",
    "missing": "missing_trade",
    "force_cover": "force_cover_gap",
}


@dataclass(slots=True)
class Summary:
    inputs: list[str]
    rows_analyzed: int
    compliant_count: int
    non_compliant_count: int
    issue_counts: dict[str, int]
    issue_dates: dict[str, list[str]]
    max_target_short_excess: float | None
    most_recent_violation: str | None

    def to_json(self) -> dict[str, object]:
        return {
            "inputs": self.inputs,
            "rows_analyzed": self.rows_analyzed,
            "compliant_count": self.compliant_count,
            "non_compliant_count": self.non_compliant_count,
            "non_compliant_ratio": round(self.non_compliant_count / self.rows_analyzed, 4) if self.rows_analyzed else 0.0,
            "issue_counts": self.issue_counts,
            "issue_dates": self.issue_dates,
            "max_target_short_excess": self.max_target_short_excess,
            "most_recent_violation": self.most_recent_violation,
        }

    def to_markdown(self) -> str:
        header = "# Risk Override Conflict Summary\n"
        lines = [
            header,
            f"- Files analysed: {', '.join(self.inputs)}",
            f"- Rows analysed: {self.rows_analyzed}",
            f"- Non-compliant rows: {self.non_compliant_count} ({self._ratio_pct():.2f}%)",
        ]
        if self.max_target_short_excess is not None:
            lines.append(f"- Max excess vs target_short_shares: {self.max_target_short_excess:.0f} shares")
        if self.most_recent_violation:
            lines.append(f"- Most recent violation: {self.most_recent_violation}")

        if self.issue_counts:
            lines.append("\n## Issue Breakdown")
            for issue, count in sorted(self.issue_counts.items(), key=lambda item: item[1], reverse=True):
                lines.append(f"- {issue.replace('_', ' ').title()}: {count}")

        if self.issue_dates:
            lines.append("\n## Dates by Issue")
            for issue, dates in sorted(self.issue_dates.items()):
                uniq_dates = ", ".join(sorted(set(dates)))
                lines.append(f"- {issue.replace('_', ' ').title()}: {uniq_dates}")

        return "\n".join(lines) + "\n"

    def _ratio_pct(self) -> float:
        if not self.rows_analyzed:
            return 0.0
        return (self.non_compliant_count / self.rows_analyzed) * 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise compliance issues from risk override audit CSV files.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more CSV files produced by risk_override_audit.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path to write a JSON summary.",
    )
    parser.add_argument(
        "--markdown-output",
        help="Optional path to write a Markdown narrative.",
    )
    return parser.parse_args()


def load_frames(paths: Iterable[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Audit file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        df["source"] = csv_path.name
        frames.append(df)
    if not frames:
        raise ValueError("No audit data provided")
    return pd.concat(frames, ignore_index=True)


def normalise_boolean(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def extract_issue_labels(issue_text: str) -> set[str]:
    labels: set[str] = set()
    lower = issue_text.lower()
    for needle, label in ISSUE_ALIASES.items():
        if needle in lower:
            labels.add(label)
    if "short_shares" in lower and "expected" in lower:
        labels.add("short_allocation_mismatch")
    return labels or {"unspecified"}


def summarise(df: pd.DataFrame, inputs: list[str]) -> Summary:
    df = df.copy()
    df["compliant"] = normalise_boolean(df["compliant"])
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    numeric_cols = [
        "short_shares",
        "target_short_shares",
        "force_cover_qty",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    non_compliant = df[~df["compliant"]]
    issue_counter: Counter[str] = Counter()
    issue_dates: dict[str, list[str]] = defaultdict(list)

    max_excess: float | None = None
    most_recent_violation: str | None = None

    for _, row in non_compliant.iterrows():
        issues_raw = str(row.get("issues", "")).strip()
        labels = extract_issue_labels(issues_raw)
        date_val = row.get("date")
        date_str = date_val.strftime("%Y-%m-%d") if isinstance(date_val, pd.Timestamp) and not pd.isna(date_val) else "unknown"
        for label in labels:
            issue_counter[label] += 1
            issue_dates[label].append(date_str)

        # Track overshoot vs target short shares if both values exist
        short_shares = row.get("short_shares")
        target_short = row.get("target_short_shares")
        if pd.notna(short_shares) and pd.notna(target_short):
            excess = float(short_shares) - float(target_short)
            if excess > 0:
                max_excess = excess if max_excess is None else max(max_excess, excess)

        if date_str != "unknown":
            if most_recent_violation is None:
                most_recent_violation = date_str
            else:
                try:
                    existing = datetime.strptime(most_recent_violation, "%Y-%m-%d")
                    candidate = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    continue
                if candidate > existing:
                    most_recent_violation = date_str

    return Summary(
        inputs=inputs,
        rows_analyzed=len(df),
        compliant_count=int(df["compliant"].sum()),
        non_compliant_count=int(len(non_compliant)),
        issue_counts=dict(issue_counter),
        issue_dates={k: v for k, v in issue_dates.items()},
        max_target_short_excess=max_excess,
        most_recent_violation=most_recent_violation,
    )


def write_optional(path: str | None, content: str | dict[str, object]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, dict):
        target.write_text(json.dumps(content, indent=2))
    else:
        target.write_text(content)


def main() -> None:
    args = parse_args()
    df = load_frames(args.input)
    summary = summarise(df, inputs=[Path(p).name for p in args.input])
    write_optional(args.json_output, summary.to_json())
    write_optional(args.markdown_output, summary.to_markdown())
    print(json.dumps(summary.to_json(), indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
