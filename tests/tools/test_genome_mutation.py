from __future__ import annotations

import random

from src.tools.genome_mutation import crossover, mutate_genome, perturb_weights, swap_analyst, toggle_traits
from src.tools.genome_registry import EvolutionGenome


def seed_genome() -> EvolutionGenome:
    genome = EvolutionGenome(
        genome_id="seed",
        label="seed",
        generation_id="gen-0",
        analysts=["ben_graham", "warren_buffett"],
        analyst_weights={"ben_graham": 0.6, "warren_buffett": 0.4},
        trait_toggles={"async_override": False},
    )
    genome.compute_hash()
    return genome


def test_perturb_weights_changes_distribution() -> None:
    rng = random.Random(42)
    genome = seed_genome()
    mutated = perturb_weights(genome, strength=0.2, rng=rng)
    assert mutated.analyst_weights != genome.analyst_weights
    assert abs(sum(mutated.analyst_weights.values()) - 1.0) < 1e-6


def test_swap_analyst_replaces_member() -> None:
    rng = random.Random(7)
    genome = seed_genome()
    mutated = swap_analyst(genome, ["cathie_wood", "peter_lynch"], rng=rng)
    assert mutated.analysts != genome.analysts
    assert set(mutated.analysts) <= {"cathie_wood", "peter_lynch", "ben_graham", "warren_buffett"}


def test_toggle_traits_inverts_flags() -> None:
    genome = seed_genome()
    mutated = toggle_traits(genome, ["async_override"])
    assert mutated.trait_toggles["async_override"] is True


def test_crossover_blends_weights() -> None:
    rng = random.Random(3)
    parent_a = seed_genome()
    parent_b = EvolutionGenome(
        genome_id="seed_b",
        label="seed_b",
        generation_id="gen-0",
        analysts=["cathie_wood"],
        analyst_weights={"cathie_wood": 1.0},
    )
    child = crossover(parent_a, parent_b, rng=rng)
    assert set(child.analysts) == {"ben_graham", "warren_buffett", "cathie_wood"}
    assert set(child.parent_ids) == {"seed", "seed_b"}


def test_mutate_genome_tracks_parent() -> None:
    rng = random.Random(11)
    genome = seed_genome()
    mutated = mutate_genome(genome, rng=rng, available_analysts=["cathie_wood"])
    assert genome.genome_id in mutated.parent_ids
    assert "mutated_from=seed" in mutated.metadata.get("mutation_notes", [])
