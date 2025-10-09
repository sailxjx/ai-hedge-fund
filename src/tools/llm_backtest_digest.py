"""Parse backtest logs into structured telemetry for downstream analysis."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
PORTFOLIO_RETURN_RE = re.compile(r"Portfolio Return:\s*([+-]?\d+(?:\.\d+)?)%")
SHARPE_RE = re.compile(r"Sharpe(?: Ratio)?:\s*([+-]?\d+(?:\.\d+)?)")
SORTINO_RE = re.compile(r"Sortino(?: Ratio)?:\s*([+-]?\d+(?:\.\d+)?)")
INFORMATION_RATIO_RE = re.compile(r"Information Ratio:\s*([+-]?\d+(?:\.\d+)?)")
MAX_DRAWDOWN_RE = re.compile(r"Max Drawdown:\s*([+-]?\d+(?:\.\d+)?)%")
BENCHMARK_RETURN_RE = re.compile(r"Benchmark Return:\s*([+-]?\d+(?:\.\d+)?)%")
TOTAL_RETURN_RE = re.compile(r"Total Return:\s*([+-]?\d+(?:\.\d+)?)%", re.IGNORECASE)
AGENT_ANALYSIS_RE = re.compile(r"AGENT ANALYSIS:\s*\[(?P<ticker>[^\]]+)\]")
RISK_CONTROLS_RE = re.compile(r"RISK CONTROLS:\s*\[(?P<ticker>[^\]]+)\]", re.IGNORECASE)
TRADING_DECISION_RE = re.compile(r"TRADING DECISION:\s*\[(?P<ticker>[^\]]+)\]", re.IGNORECASE)
CONFIDENCE_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)")
DATA_FETCH_RE = re.compile(r"Stopping .*? because .*", re.IGNORECASE)
TIMEOUT_RE = re.compile(r"timed out", re.IGNORECASE)
SHORT_COVER_RE = re.compile(
    r"Short Cover Classifier\[(?P<ticker>[^\]]+)\]\s*(?:Prob squeeze|Persona squeeze view)\s*"
    r"(?P<prob>\d+(?:\.\d+)?)%\s*\(thr\s*(?P<threshold>\d+(?:\.\d+)?)%\)\s*(?:\u2192|->)\s*"
    r"(?P<decision>[A-Z_]+)",
    re.IGNORECASE,
)
TABLE_LINE_PREFIXES = ("|", "+", "=")
NEAR_THRESHOLD_DELTA = 2.5


@dataclass
class AnalystVote:
    agent: str
    signal: str | None
    confidence_pct: float | None
    reasoning: str
    ticker: str | None


@dataclass
class Anomaly:
    type: str
    message: str
    metadata: dict[str, float | str] | None = None


@dataclass
class BacktestDigest:
    path: str
    ticker: str | None
    metrics: dict[str, float | None]
    analyst_votes: list[AnalystVote]
    anomalies: list[Anomaly]
    trading_decision: TradingDecision | None = None
    risk: RiskTelemetry | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "ticker": self.ticker,
            "metrics": self.metrics,
            "analyst_votes": [asdict(v) for v in self.analyst_votes],
            "anomalies": [asdict(a) for a in self.anomalies],
            "trading_decision": asdict(self.trading_decision) if self.trading_decision else None,
            "risk": asdict(self.risk) if self.risk else None,
        }


@dataclass
class RiskTelemetry:
    remaining_limit: float | None
    constraint_notes: list[str]
    overrides: dict[str, str]


@dataclass
class TradingDecision:
    action: str | None
    quantity: int | None
    confidence_pct: float | None
    reasoning: str | None


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _extract_float(regex: re.Pattern[str], text: str) -> float | None:
    matches = regex.findall(text)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[-1]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _parse_metrics(text: str) -> dict[str, float | None]:
    return {
        "portfolio_return_pct": _extract_float(PORTFOLIO_RETURN_RE, text),
        "sharpe_ratio": _extract_float(SHARPE_RE, text),
        "sortino_ratio": _extract_float(SORTINO_RE, text),
        "information_ratio": _extract_float(INFORMATION_RATIO_RE, text),
        "max_drawdown_pct": _extract_float(MAX_DRAWDOWN_RE, text),
        "benchmark_return_pct": _extract_float(BENCHMARK_RETURN_RE, text),
        "total_return_pct": _extract_float(TOTAL_RETURN_RE, text),
    }


def _parse_confidence(raw: str) -> float | None:
    raw = raw.strip()
    if not raw:
        return None
    if raw.endswith("%"):
        raw = raw[:-1]
    match = CONFIDENCE_RE.search(raw)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _collect_table_lines(lines: Iterable[str]) -> list[str]:
    collected: list[str] = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if started:
                break
            continue
        if not stripped.startswith(TABLE_LINE_PREFIXES):
            if started:
                break
            continue
        started = True
        collected.append(line)
    return collected


def _parse_two_column_table(lines: Sequence[str]) -> dict[str, list[str]]:
    table: dict[str, list[str]] = {}
    current_key: str | None = None

    for line in lines:
        if not line.lstrip().startswith("|"):
            continue
        columns = line.rstrip("\n").split("|")[1:-1]
        if len(columns) < 2:
            continue
        key_candidate = _clean_whitespace(columns[0])
        value_candidate = _clean_whitespace(columns[1])

        if key_candidate:
            current_key = key_candidate
            table.setdefault(current_key, [])
            if value_candidate:
                table[current_key].append(value_candidate)
            continue

        if current_key is None or not value_candidate:
            continue

        if (
            current_key == "Overrides" and ":" in value_candidate
        ) or re.match(r"\d+\.\s", value_candidate):
            table[current_key].append(value_candidate)
            continue

        if not table[current_key]:
            table[current_key].append(value_candidate)
            continue

        table[current_key][-1] = f"{table[current_key][-1]} {value_candidate}".strip()

    return table


def _parse_currency(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = value.replace("$", "").replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalise_notes(entries: list[str]) -> list[str]:
    notes: list[str] = []
    for entry in entries:
        cleaned = entry.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
        if cleaned:
            notes.append(cleaned)
    return notes


def _parse_overrides(entries: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for entry in entries:
        if ":" not in entry:
            continue
        key, value = entry.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key:
            overrides[key] = value
    return overrides


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _parse_agent_table(lines: Sequence[str], ticker: str | None) -> list[AnalystVote]:
    votes: list[AnalystVote] = []
    current: AnalystVote | None = None

    for line in lines:
        if not line.lstrip().startswith("|"):
            continue
        parts = line.rstrip("\n").split("|")[1:-1]
        if len(parts) != 4:
            continue
        columns = [part.strip() for part in parts]
        agent_raw, signal_raw, confidence_raw, reasoning_raw = columns

        agent_value = _clean_whitespace(agent_raw)
        signal_value = _clean_whitespace(signal_raw)
        confidence_value = _clean_whitespace(confidence_raw)
        reasoning_value = _clean_whitespace(reasoning_raw)

        # Skip header rows inside the table
        if agent_value.lower() == 'agent' and signal_value.lower() == 'signal':
            current = None
            continue

        if agent_value:
            current = AnalystVote(
                agent=agent_value,
                signal=signal_value.upper() if signal_value else None,
                confidence_pct=_parse_confidence(confidence_value),
                reasoning=reasoning_value,
                ticker=ticker,
            )
            votes.append(current)
            continue

        if current is None:
            continue

        if signal_value and not current.signal:
            current.signal = signal_value.upper()
        if confidence_value and current.confidence_pct is None:
            current.confidence_pct = _parse_confidence(confidence_value)
        extra_reasoning = reasoning_value
        if extra_reasoning:
            if current.reasoning:
                current.reasoning = f"{current.reasoning} {extra_reasoning}"
            else:
                current.reasoning = extra_reasoning

    return votes


def _extract_analyst_votes(text: str) -> list[AnalystVote]:
    votes: list[AnalystVote] = []
    for match in AGENT_ANALYSIS_RE.finditer(text):
        ticker = _clean_whitespace(match.group("ticker")) or None
        remainder = text[match.end() :]
        lines = remainder.splitlines()
        table_lines = _collect_table_lines(lines)
        votes.extend(_parse_agent_table(table_lines, ticker))
    return votes


def _extract_anomalies(text: str) -> list[Anomaly]:
    anomalies: list[Anomaly] = []
    seen: set[tuple[str, str]] = set()

    for line in text.splitlines():
        if DATA_FETCH_RE.search(line) or TIMEOUT_RE.search(line):
            message = _clean_whitespace(line)
            key = ("data_fetch", message)
            if message and key not in seen:
                anomalies.append(Anomaly(type="data_fetch", message=message))
                seen.add(key)

    for match in SHORT_COVER_RE.finditer(text):
        try:
            prob = float(match.group("prob"))
            threshold = float(match.group("threshold"))
        except (TypeError, ValueError):
            continue
        delta = prob - threshold
        if abs(delta) <= NEAR_THRESHOLD_DELTA:
            message = _clean_whitespace(match.group(0))
            key = ("short_cover_threshold", message)
            if key in seen:
                continue
            metadata = {
                "ticker": match.group("ticker"),
                "prob_squeeze_pct": prob,
                "threshold_pct": threshold,
                "delta_pct": delta,
                "decision": match.group("decision").upper(),
            }
            anomalies.append(Anomaly(type="short_cover_threshold", message=message, metadata=metadata))
            seen.add(key)

    return anomalies


def _extract_risk_controls(text: str) -> RiskTelemetry | None:
    match = RISK_CONTROLS_RE.search(text)
    if not match:
        return None
    remainder = text[match.end() :]
    lines = remainder.splitlines()
    table_lines = _collect_table_lines(lines)
    if not table_lines:
        return None
    table = _parse_two_column_table(table_lines)

    remaining_entries = table.get("Remaining Limit") or []
    remaining_limit = _parse_currency(remaining_entries[0]) if remaining_entries else None
    constraint_notes = _normalise_notes(table.get("Constraint Notes") or [])
    overrides = _parse_overrides(table.get("Overrides") or [])

    return RiskTelemetry(
        remaining_limit=remaining_limit,
        constraint_notes=constraint_notes,
        overrides=overrides,
    )


def _extract_trading_decision(text: str) -> TradingDecision | None:
    match = TRADING_DECISION_RE.search(text)
    if not match:
        return None
    remainder = text[match.end() :]
    lines = remainder.splitlines()
    table_lines = _collect_table_lines(lines)
    if not table_lines:
        return None
    table = _parse_two_column_table(table_lines)

    action_entries = table.get("Action") or []
    quantity_entries = table.get("Quantity") or []
    confidence_entries = table.get("Confidence") or []
    reasoning_entries = table.get("Reasoning") or []

    action = action_entries[0] if action_entries else None
    quantity = _parse_int(quantity_entries[0] if quantity_entries else None)
    confidence = _parse_confidence(confidence_entries[0] if confidence_entries else "")
    reasoning = " ".join(reasoning_entries).strip() if reasoning_entries else None

    return TradingDecision(
        action=action or None,
        quantity=quantity,
        confidence_pct=confidence,
        reasoning=reasoning or None,
    )


def parse_backtest_digest(path: Path | str) -> BacktestDigest:
    log_path = Path(path)
    raw_text = log_path.read_text(encoding="utf-8", errors="ignore")
    clean_text = _strip_ansi(raw_text)

    metrics = _parse_metrics(clean_text)
    votes = _extract_analyst_votes(clean_text)
    anomalies = _extract_anomalies(clean_text)
    risk = _extract_risk_controls(clean_text)
    decision = _extract_trading_decision(clean_text)

    unique_tickers = sorted({vote.ticker for vote in votes if vote.ticker})
    ticker = None
    if unique_tickers:
        ticker = unique_tickers[0] if len(unique_tickers) == 1 else None

    return BacktestDigest(
        path=str(log_path),
        ticker=ticker,
        metrics=metrics,
        analyst_votes=votes,
        anomalies=anomalies,
        trading_decision=decision,
        risk=risk,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise LLM backtest logs into structured digests.")
    parser.add_argument("--logs", nargs="+", required=True, help="Paths to log files to parse.")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    digests = [parse_backtest_digest(Path(path)) for path in args.logs]
    payload = [digest.to_dict() for digest in digests]

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote digest to {output_path}")
        return

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
