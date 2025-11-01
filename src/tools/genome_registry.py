"""Evolution genome configuration structures and lineage persistence utilities."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sorted_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively sort mapping keys to ensure deterministic hashing."""

    def _convert(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): _convert(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple, set)):
            return [_convert(item) for item in value]
        return value

    return _convert(data)  # type: ignore[return-value]


@dataclass(slots=True)
class EvolutionGenome:
    """Structured description of an arena participant configuration."""

    genome_id: str
    label: str
    generation_id: str
    analysts: list[str]
    prompt_variants: dict[str, str] = field(default_factory=dict)
    trait_toggles: dict[str, bool] = field(default_factory=dict)
    analyst_weights: dict[str, float] = field(default_factory=dict)
    async_mode: str = "async"
    patriarch_variant: str | None = None
    risk_params: dict[str, Any] = field(default_factory=dict)
    portfolio_params: dict[str, Any] = field(default_factory=dict)
    execution_params: dict[str, Any] = field(default_factory=dict)
    novelty_penalty: float | None = None
    parent_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    genome_hash: str | None = None

    def canonical_payload(self) -> dict[str, Any]:
        """Return a deterministic representation suitable for hashing/serialization."""

        payload = asdict(self)
        payload.pop("genome_hash", None)
        return _sorted_dict(payload)

    def compute_hash(self) -> str:
        """Compute and persist the deterministic genome hash."""

        canonical = self.canonical_payload()
        encoded = json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        self.genome_hash = digest
        return digest

    def ensure_hash(self) -> str:
        if not self.genome_hash:
            return self.compute_hash()
        return self.genome_hash

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = self.canonical_payload()
        if include_hash:
            payload["genome_hash"] = self.ensure_hash()
        else:
            payload.pop("created_at", None)  # canonical already contains created_at
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvolutionGenome":
        data = dict(payload)
        genome_hash = data.pop("genome_hash", None)
        genome = cls(
            genome_id=str(data["genome_id"]),
            label=str(data["label"]),
            generation_id=str(data.get("generation_id") or ""),
            analysts=[str(item) for item in data.get("analysts", [])],
            prompt_variants=dict(data.get("prompt_variants", {})),
            trait_toggles={str(k): bool(v) for k, v in dict(data.get("trait_toggles", {})).items()},
            analyst_weights={str(k): float(v) for k, v in dict(data.get("analyst_weights", {})).items()},
            async_mode=str(data.get("async_mode") or "async"),
            patriarch_variant=(str(data["patriarch_variant"]) if data.get("patriarch_variant") else None),
            risk_params=dict(data.get("risk_params", {})),
            portfolio_params=dict(data.get("portfolio_params", {})),
            execution_params=dict(data.get("execution_params", {})),
            novelty_penalty=(float(data["novelty_penalty"]) if data.get("novelty_penalty") is not None else None),
            parent_ids=[str(item) for item in data.get("parent_ids", [])],
            metadata=dict(data.get("metadata", {})),
            created_at=str(data.get("created_at") or _now_iso()),
        )
        genome.genome_hash = str(genome_hash) if genome_hash else None
        return genome


class GenomeLineageRegistry:
    """Lightweight persistence layer for evolution genomes."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path("log") / "genomes"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, genome_id: str) -> Path:
        safe_id = genome_id.replace("/", "_")
        return self.root / f"{safe_id}.json"

    def save(self, genome: EvolutionGenome) -> Path:
        path = self._path_for(genome.genome_id)
        payload = genome.to_dict(include_hash=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def load(self, genome_id: str) -> EvolutionGenome:
        path = self._path_for(genome_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return EvolutionGenome.from_dict(payload)

    def list_ids(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.json"))

    def exists(self, genome_id: str) -> bool:
        return self._path_for(genome_id).exists()

    def archive_generation(self, generation_id: str, genomes: Sequence[EvolutionGenome]) -> list[Path]:
        paths: list[Path] = []
        for genome in genomes:
            genome.generation_id = generation_id
            genome.ensure_hash()
            paths.append(self.save(genome))
        return paths
