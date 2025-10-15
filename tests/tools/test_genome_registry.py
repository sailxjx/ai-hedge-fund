from __future__ import annotations

from pathlib import Path

from src.tools.genome_registry import EvolutionGenome, GenomeLineageRegistry


def test_genome_hash_deterministic(tmp_path: Path) -> None:
    genome = EvolutionGenome(
        genome_id="genome-001",
        label="baseline",
        generation_id="gen-0",
        analysts=["ben_graham", "warren_buffett"],
        prompt_variants={"ben_graham": "v1"},
        trait_toggles={"async": True},
        analyst_weights={"ben_graham": 0.6, "warren_buffett": 0.4},
        async_mode="async",
        patriarch_variant=None,
        risk_params={"max_gross": 1.0},
        portfolio_params={"rebal_freq": "daily"},
        execution_params={},
        novelty_penalty=0.1,
        parent_ids=["ancestor-000"],
        metadata={"notes": "seed genome"},
        created_at="2025-10-13T00:00:00+00:00",
    )

    first_hash = genome.compute_hash()
    second_hash = genome.compute_hash()
    assert first_hash == second_hash

    payload = genome.to_dict()
    assert payload["genome_hash"] == first_hash
    assert payload["analysts"] == ["ben_graham", "warren_buffett"]


def test_registry_roundtrip(tmp_path: Path) -> None:
    registry = GenomeLineageRegistry(root=tmp_path / "genomes")
    genome = EvolutionGenome(
        genome_id="player-1",
        label="baseline",
        generation_id="gen-0",
        analysts=["all"],
    )
    genome.metadata["notes"] = "smoke test"
    registry.save(genome)

    loaded = registry.load("player-1")
    assert loaded.genome_id == genome.genome_id
    assert loaded.metadata == genome.metadata
    assert registry.list_ids() == ["player-1"]
    assert registry.exists("player-1")
