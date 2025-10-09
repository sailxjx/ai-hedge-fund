"""Audit risk manager overrides against executed trades in backtest logs."""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    if sys.path and sys.path[0] == current_dir:
        sys.path.pop(0)
        sys.path.append(current_dir)

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    from .short_exposure_audit import SummaryRecord, parse_summary_records
except ImportError:  # pragma: no cover - triggered when executed as script
    from src.backtesting.short_exposure_audit import SummaryRecord, parse_summary_records


@dataclass
class OverrideEntry:
    """Single override instruction emitted by the risk manager."""

    date: str
    target_long_shares: int | None = None
    target_short_shares: int | None = None
    preferred_direction: str | None = None
    block_new_shorts: bool | None = None
    max_additional_short_shares: int | None = None
    force_cover_qty: int | None = None


@dataclass
class OverrideCheck:
    """Comparison result between an override and the executed trade."""

    date: str
    action: str | None = None
    quantity: int | None = None
    long_shares: int | None = None
    short_shares: int | None = None
    entry: OverrideEntry | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def compliant(self) -> bool:
        return not self.issues

    def to_row(self) -> dict[str, object]:
        entry = self.entry or OverrideEntry(date=self.date)
        return {
            "date": self.date,
            "action": self.action,
            "quantity": self.quantity,
            "long_shares": self.long_shares,
            "short_shares": self.short_shares,
            "target_long_shares": entry.target_long_shares,
            "target_short_shares": entry.target_short_shares,
            "preferred_direction": entry.preferred_direction,
            "block_new_shorts": entry.block_new_shorts,
            "max_additional_short_shares": entry.max_additional_short_shares,
            "force_cover_qty": entry.force_cover_qty,
            "compliant": self.compliant,
            "issues": "; ".join(self.issues),
        }


def _load_overrides(path: Path) -> list[OverrideEntry]:
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:  # pragma: no cover - surfaced via CLI
        raise SystemExit(f"Failed to read overrides file '{path}': {exc}") from exc

    entries: list[OverrideEntry] = []
    for line in raw_lines:
        if not line.strip():
            continue
        data = json.loads(line)
        overrides = data.get("risk_snapshot", {}).get("overrides", {})
        entry = OverrideEntry(
            date=data.get("end_date"),
            target_long_shares=_safe_int(overrides.get("target_long_shares")),
            target_short_shares=_safe_int(overrides.get("target_short_shares")),
            preferred_direction=_safe_str(overrides.get("preferred_direction")),
            block_new_shorts=_safe_bool(overrides.get("block_new_shorts")),
            max_additional_short_shares=_safe_int(overrides.get("max_additional_short_shares")),
            force_cover_qty=_safe_int(overrides.get("force_cover_qty")),
        )
        entries.append(entry)
    return entries


def _records_by_date(records: Sequence[SummaryRecord]) -> dict[str, SummaryRecord]:
    """Collapse repeated log snapshots down to the latest entry per date."""

    by_date: dict[str, SummaryRecord] = {}
    for record in records:
        by_date[record.date] = record
    return by_date


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):  # guard bools masquerading as ints
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _safe_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value)


def audit_overrides(
    overrides: Sequence[OverrideEntry],
    trade_records: Dict[str, SummaryRecord],
) -> list[OverrideCheck]:
    results: list[OverrideCheck] = []
    for entry in overrides:
        record = trade_records.get(entry.date)
        check = OverrideCheck(
            date=entry.date,
            action=record.action if record else None,
            quantity=record.quantity if record else None,
            long_shares=record.long_shares if record else None,
            short_shares=record.short_shares if record else None,
            entry=entry,
        )
        if record is None:
            check.issues.append("trade record missing in log")
            results.append(check)
            continue

        # Target long exposure alignment
        if entry.target_long_shares is not None and record.long_shares != entry.target_long_shares:
            check.issues.append(
                f"long_shares={record.long_shares} expected {entry.target_long_shares}"
            )

        # Target short exposure alignment (target represents final desired exposure)
        if entry.target_short_shares is not None:
            expected_short = entry.target_short_shares
            if record.short_shares != expected_short:
                check.issues.append(
                    f"short_shares={record.short_shares} expected {expected_short}"
                )

        # Block new shorts directive
        if entry.block_new_shorts and record.action.upper() == "SHORT":
            check.issues.append("block_new_shorts violated by SHORT action")

        # Maximum incremental short sizing guard
        if (
            entry.max_additional_short_shares is not None
            and record.action.upper() == "SHORT"
            and record.quantity is not None
            and record.quantity > entry.max_additional_short_shares
        ):
            check.issues.append(
                f"short quantity {record.quantity} exceeds max {entry.max_additional_short_shares}"
            )

        # Force cover instructions
        if entry.force_cover_qty is not None:
            if record.action.upper() != "COVER":
                check.issues.append("force_cover_qty issued but action was not COVER")
            elif record.quantity is not None and record.quantity < entry.force_cover_qty:
                check.issues.append(
                    f"cover quantity {record.quantity} below required {entry.force_cover_qty}"
                )

        # Preferred direction sanity check (best-effort)
        if entry.preferred_direction:
            preferred = entry.preferred_direction.lower()
            action = record.action.upper()
            if preferred == "long" and action == "SHORT":
                check.issues.append("preferred long but executed SHORT")
            if preferred == "short" and action == "BUY":
                check.issues.append("preferred short but executed BUY")

        results.append(check)
    return results


def _write_csv(path: Path, checks: Iterable[OverrideCheck]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "action",
                "quantity",
                "long_shares",
                "short_shares",
                "target_long_shares",
                "target_short_shares",
                "preferred_direction",
                "block_new_shorts",
                "max_additional_short_shares",
                "force_cover_qty",
                "compliant",
                "issues",
            ],
        )
        writer.writeheader()
        for check in checks:
            writer.writerow(check.to_row())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit risk manager overrides against backtest trade logs",
    )
    parser.add_argument("--log", required=True, type=Path, help="Path to backtest log")
    parser.add_argument(
        "--overrides",
        required=True,
        type=Path,
        help="Path to risk override JSONL emitted during the run",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to write CSV output summarising the audit",
    )
    args = parser.parse_args(argv)

    records = parse_summary_records(label="audit", path=args.log)
    trade_records = _records_by_date(records)
    overrides = _load_overrides(args.overrides)
    checks = audit_overrides(overrides, trade_records)

    compliant = sum(1 for c in checks if c.compliant)
    failures = len(checks) - compliant

    print(f"Audited {len(checks)} override entries: {compliant} aligned, {failures} issues.")
    if failures:
        for check in checks:
            if check.issues:
                issues = "; ".join(check.issues)
                print(f" - {check.date}: {issues}")

    if args.csv is not None:
        _write_csv(args.csv, checks)
        print(f"Wrote audit summary to {args.csv}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
