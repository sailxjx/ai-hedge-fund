"""Arena selection engine for ranking genomes and logging promotion decisions."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalise_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    if not weights:
        return {}
    total = sum(abs(float(value)) for value in weights.values())
    if total == 0:
        return dict((key, float(value)) for key, value in weights.items())
    return {key: float(value) / total for key, value in weights.items()}


def _weight_entropy(weights: Mapping[str, float] | None) -> float | None:
    if not weights:
        return None
    normalised = _normalise_weights(weights)
    entropy = 0.0
    for value in normalised.values():
        proportion = abs(value)
        if proportion > 0:
            entropy -= proportion * math.log(proportion, 2)
    return entropy


@dataclass
class SelectionRules:
    version: int
    gates: dict[str, float]
    weights: dict[str, float]
    penalties: dict[str, float]
    bonus_tags: dict[str, float]
    promote_threshold: float
    hold_threshold: float
    top_n: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SelectionRules":
        version = int(payload.get("version", 1))
        gates = dict(payload.get("gates", {}))
        fitness = dict(payload.get("fitness", {}))
        weights = dict(fitness.get("weights", {}))
        penalties = dict(fitness.get("penalties", {}))
        bonus_tags = dict(fitness.get("bonus_tags", {}))
        promotion = dict(payload.get("promotion", {}))
        promote_threshold = float(promotion.get("promote_threshold", 0.5))
        hold_threshold = float(promotion.get("hold_threshold", 0.0))
        top_n = int(promotion.get("top_n", 3))
        metadata_raw = dict(payload.get("metadata", {}))
        metadata: dict[str, Any] = {}
        for key, value in metadata_raw.items():
            if isinstance(value, datetime):
                metadata[str(key)] = value.isoformat()
            else:
                metadata[str(key)] = value
        return cls(
            version=version,
            gates=gates,
            weights=weights,
            penalties=penalties,
            bonus_tags=bonus_tags,
            promote_threshold=promote_threshold,
            hold_threshold=hold_threshold,
            top_n=top_n,
            metadata=metadata,
        )


@dataclass
class SelectionCandidate:
    label: str
    generation_id: str | None
    genome_id: str | None
    genome_label: str | None
    summary_path: Path
    metrics: dict[str, float | None]
    token_cost_usd: float | None
    async_concurrency_ratio: float | None
    weight_entropy: float | None
    extra_tags: list[str]
    gating_violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fitness: float | None = None
    recommendation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "generation_id": self.generation_id,
            "genome_id": self.genome_id,
            "genome_label": self.genome_label,
            "summary_path": str(self.summary_path),
            "metrics": self.metrics,
            "token_cost_usd": self.token_cost_usd,
            "async_concurrency_ratio": self.async_concurrency_ratio,
            "weight_entropy": self.weight_entropy,
            "extra_tags": self.extra_tags,
            "gating_violations": self.gating_violations,
            "warnings": self.warnings,
            "fitness": self.fitness,
            "recommendation": self.recommendation,
        }


def load_selection_rules(path: Path) -> SelectionRules:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Selection rules file {path} must contain a mapping.")
    return SelectionRules.from_mapping(payload)


def _load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Arena summary {path} must contain a JSON object.")
    return dict(payload)


def _extract_async_ratio(summary: Mapping[str, Any]) -> float | None:
    async_meta = summary.get("async_telemetry")
    if not isinstance(async_meta, Mapping):
        return None
    return _coerce_float(async_meta.get("async_concurrency_ratio"))


def _extract_metrics(summary: Mapping[str, Any]) -> dict[str, float | None]:
    metrics_payload = summary.get("metrics")
    metrics: dict[str, float | None] = {}
    if isinstance(metrics_payload, Mapping):
        for key, value in metrics_payload.items():
            metrics[key] = _coerce_float(value)
    return metrics


def _extract_weights(summary: Mapping[str, Any]) -> dict[str, float]:
    weights_payload = summary.get("analyst_weights")
    if isinstance(weights_payload, Mapping):
        try:
            return {str(key): float(value) for key, value in weights_payload.items()}
        except (TypeError, ValueError):
            return {}
    return {}


def evaluate_candidate(summary_path: Path, rules: SelectionRules) -> SelectionCandidate:
    summary = _load_summary(summary_path)
    metrics = _extract_metrics(summary)
    token_cost = _coerce_float(summary.get("token_cost_usd"))
    async_ratio = _extract_async_ratio(summary)
    weights = _extract_weights(summary)
    entropy = _weight_entropy(weights) if weights else None
    extra_tags = [str(tag) for tag in (summary.get("extra_tags") or [])]

    candidate = SelectionCandidate(
        label=str(summary.get("label")),
        generation_id=(str(summary.get("generation_id")) if summary.get("generation_id") else None),
        genome_id=(str(summary.get("genome_id")) if summary.get("genome_id") else None),
        genome_label=(str(summary.get("genome_label")) if summary.get("genome_label") else None),
        summary_path=summary_path,
        metrics=metrics,
        token_cost_usd=token_cost,
        async_concurrency_ratio=async_ratio,
        weight_entropy=entropy,
        extra_tags=extra_tags,
    )

    gates = rules.gates

    def _require(metric_key: str, threshold: float, *, comparator: str) -> None:
        value = None
        if metric_key in ("token_cost_usd", "async_concurrency_ratio"):
            value = token_cost if metric_key == "token_cost_usd" else async_ratio
        elif metric_key == "weight_entropy":
            value = entropy
        else:
            value = metrics.get(metric_key)

        if value is None:
            candidate.warnings.append(f"Missing metric {metric_key} for gating assessment.")
            return

        if comparator == "min" and value < threshold:
            candidate.gating_violations.append(f"{metric_key}={value:.3f} < min {threshold}")
        if comparator == "max" and value > threshold:
            candidate.gating_violations.append(f"{metric_key}={value:.3f} > max {threshold}")

    if "min_sharpe" in gates:
        _require("sharpe_ratio", float(gates["min_sharpe"]), comparator="min")
    if "min_sortino" in gates:
        _require("sortino_ratio", float(gates["min_sortino"]), comparator="min")
    if "min_return_pct" in gates:
        _require("portfolio_return_pct", float(gates["min_return_pct"]), comparator="min")
    if "max_drawdown_pct" in gates:
        threshold = float(gates["max_drawdown_pct"])
        drawdown = metrics.get("max_drawdown_pct")
        if drawdown is not None and drawdown < threshold:
            candidate.gating_violations.append(f"max_drawdown_pct={drawdown:.3f} < allowed {threshold}")
    if "max_turnover_pct" in gates:
        _require("turnover_rate_pct", float(gates["max_turnover_pct"]), comparator="max")
    if "max_token_cost_usd" in gates:
        _require("token_cost_usd", float(gates["max_token_cost_usd"]), comparator="max")
    if "max_async_concurrency_ratio" in gates:
        _require("async_concurrency_ratio", float(gates["max_async_concurrency_ratio"]), comparator="max")
    if "min_weight_entropy" in gates:
        _require("weight_entropy", float(gates["min_weight_entropy"]), comparator="min")

    if not candidate.gating_violations:
        fitness = 0.0
        for key, weight in rules.weights.items():
            value = None
            if key in ("token_cost_usd", "async_concurrency_ratio", "weight_entropy"):
                value = token_cost if key == "token_cost_usd" else async_ratio if key == "async_concurrency_ratio" else entropy
            else:
                value = metrics.get(key)
            if value is not None:
                fitness += weight * value

        for key, penalty in rules.penalties.items():
            value = None
            if key == "token_cost_usd":
                value = token_cost
            elif key == "async_concurrency_ratio":
                value = async_ratio
            else:
                value = metrics.get(key)
            if value is None:
                continue
            penalty_value = abs(value) if key == "max_drawdown_pct" else max(0.0, value)
            fitness -= penalty * penalty_value

        bonus = 0.0
        for tag in extra_tags:
            bonus += rules.bonus_tags.get(tag, 0.0)
        fitness += bonus
        candidate.fitness = fitness
    else:
        candidate.fitness = None

    return candidate


def discover_candidates(arena_dir: Path, generation: str | None, rules: SelectionRules) -> list[SelectionCandidate]:
    if not arena_dir.exists():
        raise FileNotFoundError(f"Arena directory not found: {arena_dir}")
    candidates: list[SelectionCandidate] = []
    for path in arena_dir.rglob("*.json"):
        if path.name.startswith("."):
            continue
        if path.name.endswith("events.json"):
            continue
        if "selection" in path.parts:
            continue
        try:
            candidate = evaluate_candidate(path, rules)
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError(f"Failed to evaluate arena summary {path}: {exc}") from exc
        if generation and candidate.generation_id != generation:
            continue
        candidates.append(candidate)
    return candidates


def apply_recommendations(candidates: list[SelectionCandidate], rules: SelectionRules) -> None:
    promotable = [
        candidate
        for candidate in candidates
        if candidate.fitness is not None and not candidate.gating_violations
    ]
    promotable.sort(key=lambda c: c.fitness if c.fitness is not None else float("-inf"), reverse=True)

    for candidate in promotable[: rules.top_n]:
        if candidate.fitness is not None and candidate.fitness >= rules.promote_threshold:
            candidate.recommendation = "promote"

    for candidate in candidates:
        if candidate.recommendation == "promote":
            continue
        if candidate.fitness is None or candidate.gating_violations:
            candidate.recommendation = "reject"
            continue
        if candidate.fitness >= rules.promote_threshold:
            candidate.recommendation = "hold"
        else:
            if candidate.fitness >= rules.hold_threshold:
                candidate.recommendation = "hold"
            else:
                candidate.recommendation = "reject"


def render_markdown(candidates: Sequence[SelectionCandidate], output_path: Path, *, generation: str | None) -> None:
    lines: list[str] = []
    header = f"# Arena Selection Report ({generation or 'all generations'})"
    lines.append(header)
    lines.append("")
    lines.append("| Label | Genome | Fitness | Recommendation | Sharpe | Return % | Drawdown % | Token Cost | Violations |")
    lines.append("| --- | --- | ---:| --- | ---:| ---:| ---:| ---:| --- |")
    for candidate in sorted(candidates, key=lambda c: c.fitness if c.fitness is not None else float("-inf"), reverse=True):
        metrics = candidate.metrics
        sharpe = metrics.get("sharpe_ratio")
        ret = metrics.get("portfolio_return_pct")
        drawdown = metrics.get("max_drawdown_pct")
        token_cost = candidate.token_cost_usd
        violations = "; ".join(candidate.gating_violations) if candidate.gating_violations else ""
        lines.append(
            "| {label} | {genome} | {fitness} | {rec} | {sharpe} | {ret} | {drawdown} | {token_cost} | {violations} |".format(
                label=candidate.label or "n/a",
                genome=candidate.genome_id or "n/a",
                fitness=f"{candidate.fitness:.3f}" if candidate.fitness is not None else "—",
                rec=candidate.recommendation or "n/a",
                sharpe=f"{sharpe:.2f}" if sharpe is not None else "—",
                ret=f"{ret:.2f}" if ret is not None else "—",
                drawdown=f"{drawdown:.2f}" if drawdown is not None else "—",
                token_cost=f"{token_cost:.2f}" if token_cost is not None else "—",
                violations=violations or "—",
            )
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def persist_report(
    *,
    output_dir: Path,
    candidates: Sequence[SelectionCandidate],
    rules_path: Path,
    rules: SelectionRules,
    generation: str | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = generation.replace("/", "_") if generation else "all"
    json_path = output_dir / f"selection_{slug}_{timestamp}.json"
    payload = {
        "timestamp": _now_iso(),
        "generation_id": generation,
        "rules_path": str(rules_path),
        "rules_version": rules.version,
        "metadata": rules.metadata,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    md_path = output_dir / f"selection_{slug}_{timestamp}.md"
    render_markdown(candidates, md_path, generation=generation)
    return json_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank arena participants and log promotion decisions.")
    parser.add_argument("--arena-dir", type=Path, default=Path("log") / "arena", help="Directory containing arena summaries.")
    parser.add_argument("--generation", help="Optional generation identifier to filter candidates.")
    parser.add_argument("--rules", type=Path, default=Path("configs") / "selection_rules.yaml", help="Path to selection rules YAML.")
    parser.add_argument("--output-dir", type=Path, default=Path("log") / "arena" / "selection", help="Directory to write selection reports.")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate and print results without writing reports.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rules = load_selection_rules(args.rules if args.rules.is_absolute() else Path.cwd() / args.rules)
    arena_dir = args.arena_dir if args.arena_dir.is_absolute() else Path.cwd() / args.arena_dir
    candidates = discover_candidates(arena_dir, args.generation, rules)
    if not candidates:
        print(f"No arena summaries found in {arena_dir}")
        return 1
    apply_recommendations(candidates, rules)

    if args.dry_run:
        for candidate in sorted(candidates, key=lambda c: c.fitness if c.fitness is not None else float("-inf"), reverse=True):
            fitness_text = f"{candidate.fitness:.3f}" if candidate.fitness is not None else "n/a"
            print(
                f"{candidate.label} :: fitness={fitness_text} "
                f"recommendation={candidate.recommendation} violations={len(candidate.gating_violations)}"
            )
        return 0

    output_dir = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir
    persist_report(
        output_dir=output_dir,
        candidates=candidates,
        rules_path=args.rules if args.rules.is_absolute() else Path.cwd() / args.rules,
        rules=rules,
        generation=args.generation,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
