"""Detect lapses where short exposure persists into rising prices."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
PORTFOLIO_SUMMARY_MARKER = "PORTFOLIO SUMMARY:"
ROW_START_RE = re.compile(r"^\|\s*(20\d{2}-\d{2}-\d{2})\s*\|")
CASH_RE = re.compile(r"Cash Balance:\s*\$([\d,\.\-]+)")
TOTAL_POSITION_RE = re.compile(r"Total Position Value:\s*\$([\d,\.\-]+)")
TOTAL_VALUE_RE = re.compile(r"Total Value:\s*\$([\d,\.\-]+)")
RETURN_RE = re.compile(r"Portfolio Return:\s*([+-]?[\d\.]+)%")
SHARPE_RE = re.compile(r"Sharpe(?: Ratio)?:\s*([+-]?[\d\.]+)")
SORTINO_RE = re.compile(r"Sortino(?: Ratio)?:\s*([+-]?[\d\.]+)")
INFO_RATIO_RE = re.compile(r"Information Ratio:\s*([+-]?[\d\.]+)")
DRAWDOWN_RE = re.compile(r"Max Drawdown:\s*([+-]?[\d\.]+)%")
BENCHMARK_RETURN_RE = re.compile(r"Benchmark Return:\s*([+-]?[\d\.]+)%")


@dataclass
class SummaryRecord:
    """Single `PORTFOLIO SUMMARY` snapshot tied to the leading log row."""

    label: str
    date: str
    ticker: str
    action: str
    quantity: int
    price: float
    long_shares: int
    short_shares: int
    position_value: float | None
    cash: float | None
    total_position_value: float | None
    total_value: float | None
    portfolio_return: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    max_drawdown: float | None
    information_ratio: float | None
    benchmark_return_pct: float | None


@dataclass
class ExposureFailure:
    """A detected short exposure that persisted while price advanced."""

    label: str
    date: str
    ticker: str
    action: str
    prev_action: str
    short_shares: int
    price: float
    prev_price: float
    price_change: float
    price_change_pct: float
    nav: float | None
    prev_nav: float | None
    nav_change: float | None
    nav_change_pct: float | None
    exposure_value: float | None
    exposure_pct: float | None
    information_ratio: float | None
    benchmark_return_pct: float | None

    def to_row(self) -> dict[str, str | float | int | None]:
        return asdict(self)


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _parse_header_row(line: str) -> list[str] | None:
    if not ROW_START_RE.match(line):
        return None
    cells = [cell.strip() for cell in line.split("|")[1:-1]]
    return cells if len(cells) == 8 else None


def _to_int(value: str) -> int:
    return int(value.replace(",", ""))


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return float(value.replace(",", ""))


def parse_summary_records(label: str, path: Path) -> list[SummaryRecord]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:  # pragma: no cover - surfaced in CLI
        raise SystemExit(f"Failed to read log '{path}': {exc}") from exc

    clean = _strip_ansi(raw)
    lines = clean.splitlines()
    records: list[SummaryRecord] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith(PORTFOLIO_SUMMARY_MARKER):
            i += 1
            cash = total_position = total_value = portfolio_return = None
            sharpe = sortino = drawdown = None
            info_ratio = benchmark_return = None
            # collect summary block
            while i < len(lines) and lines[i].strip():
                line = lines[i]
                if (match := CASH_RE.search(line)) is not None:
                    cash = _safe_float(match.group(1))
                if (match := TOTAL_POSITION_RE.search(line)) is not None:
                    total_position = _safe_float(match.group(1))
                if (match := TOTAL_VALUE_RE.search(line)) is not None:
                    total_value = _safe_float(match.group(1))
                if (match := RETURN_RE.search(line)) is not None:
                    portfolio_return = float(match.group(1)) / 100.0
                if (match := SHARPE_RE.search(line)) is not None:
                    sharpe = float(match.group(1))
                if (match := SORTINO_RE.search(line)) is not None:
                    sortino = float(match.group(1))
                if (match := INFO_RATIO_RE.search(line)) is not None:
                    info_ratio = float(match.group(1))
                if (match := DRAWDOWN_RE.search(line)) is not None:
                    drawdown = float(match.group(1)) / 100.0
                if (match := BENCHMARK_RETURN_RE.search(line)) is not None:
                    benchmark_return = float(match.group(1)) / 100.0
                i += 1
            # advance to first data row
            while i < len(lines) and _parse_header_row(lines[i]) is None:
                i += 1
            if i >= len(lines):
                break
            cells = _parse_header_row(lines[i])
            if cells is None:
                continue
            date, ticker, action, quantity, price, long_shares, short_shares, position_value = cells
            try:
                record = SummaryRecord(
                    label=label,
                    date=date,
                    ticker=ticker,
                    action=action.replace(" ", ""),
                    quantity=_to_int(quantity),
                    price=float(price),
                    long_shares=_to_int(long_shares),
                    short_shares=_to_int(short_shares),
                    position_value=_safe_float(position_value),
                    cash=cash,
                    total_position_value=total_position,
                    total_value=total_value,
                    portfolio_return=portfolio_return,
                    sharpe_ratio=sharpe,
                    sortino_ratio=sortino,
                    max_drawdown=drawdown,
                    information_ratio=info_ratio,
                    benchmark_return_pct=benchmark_return,
                )
            except ValueError:
                i += 1
                continue
            records.append(record)
        else:
            i += 1
    return records


def _group_records(records: Sequence[SummaryRecord]) -> dict[str, List[SummaryRecord]]:
    groups: dict[str, List[SummaryRecord]] = {}
    for record in records:
        groups.setdefault(record.label, []).append(record)
    for recs in groups.values():
        recs.sort(key=lambda rec: rec.date)
    return groups


def detect_short_failures(
    records: Sequence[SummaryRecord],
    *,
    price_tolerance: float = 0.0,
    min_exposure_pct: float = 0.02,
) -> list[ExposureFailure]:
    grouped = _group_records(records)
    failures: list[ExposureFailure] = []

    for label, recs in grouped.items():
        previous: SummaryRecord | None = None
        for record in recs:
            if previous is None:
                previous = record
                continue
            price_change = record.price - previous.price
            if price_change <= price_tolerance:
                previous = record
                continue
            if record.short_shares <= 0:
                previous = record
                continue

            price_change_pct = price_change / previous.price if previous.price else 0.0
            nav = record.total_value
            prev_nav = previous.total_value
            nav_change = nav - prev_nav if nav is not None and prev_nav is not None else None
            nav_change_pct = None
            if nav_change is not None and prev_nav:
                nav_change_pct = nav_change / prev_nav

            exposure_value = None
            exposure_pct = None
            if record.position_value is not None:
                exposure_value = abs(record.position_value)
                if nav and nav != 0:
                    exposure_pct = exposure_value / nav

            if exposure_pct is not None and exposure_pct < min_exposure_pct:
                previous = record
                continue

            failures.append(
                ExposureFailure(
                    label=label,
                    date=record.date,
                    ticker=record.ticker,
                    action=record.action,
                    prev_action=previous.action,
                    short_shares=record.short_shares,
                    price=record.price,
                    prev_price=previous.price,
                    price_change=price_change,
                    price_change_pct=price_change_pct,
                    nav=nav,
                    prev_nav=prev_nav,
                    nav_change=nav_change,
                    nav_change_pct=nav_change_pct,
                    exposure_value=exposure_value,
                    exposure_pct=exposure_pct,
                    information_ratio=record.information_ratio,
                    benchmark_return_pct=record.benchmark_return_pct,
                )
            )
            previous = record
    return failures


def _parse_log_argument(log_arg: str) -> tuple[str, Path]:
    if "=" in log_arg:
        label, raw_path = log_arg.split("=", 1)
        label = label.strip() or Path(raw_path).stem
    else:
        raw_path = log_arg
        label = Path(raw_path).stem
    return label, Path(raw_path)


def _write_csv(rows: Iterable[dict[str, object]], output_path: Path) -> None:
    rows = list(rows)
    if not rows:
        raise SystemExit("No exposure failures detected; nothing to write.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect short exposure failures where prices rise while shorts persist.",
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="List of logs to audit; optionally prefix with label=path",
    )
    parser.add_argument(
        "--output",
        help="Optional CSV output path for detected failures.",
    )
    parser.add_argument(
        "--json",
        help="Optional JSON output path for detected failures.",
    )
    parser.add_argument(
        "--price-tolerance",
        type=float,
        default=0.0,
        help="Ignore price increases at or below this threshold (absolute dollars).",
    )
    parser.add_argument(
        "--min-exposure-pct",
        type=float,
        default=0.02,
        help="Minimum |position|/NAV required to flag a short exposure (default 0.02 = 2%%).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    all_records: list[SummaryRecord] = []
    for log_arg in args.logs:
        label, path = _parse_log_argument(log_arg)
        all_records.extend(parse_summary_records(label, path))

    failures = detect_short_failures(
        all_records,
        price_tolerance=args.price_tolerance,
        min_exposure_pct=args.min_exposure_pct,
    )

    if args.output:
        _write_csv((failure.to_row() for failure in failures), Path(args.output))
        print(f"Wrote CSV report to {args.output}")

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_payload = [failure.to_row() for failure in failures]
        json_path.write_text(json.dumps(json_payload, indent=2))
        print(f"Wrote JSON report to {args.json}")

    if not args.output and not args.json:
        print(json.dumps([failure.to_row() for failure in failures], indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
