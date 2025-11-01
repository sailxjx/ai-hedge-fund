"""Mutation and crossover utilities for evolutionary genomes."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Callable, Iterable, Mapping, Sequence

from .genome_registry import EvolutionGenome

WeightMutationFn = Callable[[float], float]


def _normalise_weights(weights: Mapping[str, float]) -> dict[str, float]:
    if not weights:
        return {}
    total = sum(abs(value) for value in weights.values())
    if total == 0.0:
        return dict(weights)
    return {key: value / total for key, value in weights.items()}


def perturb_weights(
    genome: EvolutionGenome,
    *,
    strength: float = 0.15,
    rng: random.Random | None = None,
) -> EvolutionGenome:
    """Apply Gaussian perturbations to analyst weights and normalise."""

    rng = rng or random
    weights = genome.analyst_weights or {analyst: 1.0 for analyst in genome.analysts}
    mutated: dict[str, float] = {}
    for key, value in weights.items():
        perturb = rng.gauss(0.0, strength)
        mutated[key] = value + perturb
    normalised = _normalise_weights(mutated)
    return replace(genome, analyst_weights=normalised)


def swap_analyst(
    genome: EvolutionGenome,
    available: Sequence[str],
    *,
    rng: random.Random | None = None,
) -> EvolutionGenome:
    """Swap one analyst from the genome with a new analyst from the pool."""

    rng = rng or random
    if not genome.analysts or not available:
        return genome
    current = list(genome.analysts)
    candidate_pool = [item for item in available if item not in current]
    if not candidate_pool:
        return genome
    remove_index = rng.randrange(len(current))
    added = rng.choice(candidate_pool)
    current[remove_index] = added
    weights = dict(genome.analyst_weights)
    if weights:
        weights.pop(genome.analysts[remove_index], None)
        weights[added] = weights.get(added, 0.0) + (1.0 / len(current))
    return replace(genome, analysts=current, analyst_weights=_normalise_weights(weights))


def toggle_traits(
    genome: EvolutionGenome,
    trait_keys: Iterable[str],
) -> EvolutionGenome:
    toggled = dict(genome.trait_toggles)
    for key in trait_keys:
        toggled[key] = not toggled.get(key, False)
    return replace(genome, trait_toggles=toggled)


def crossover(
    parent_a: EvolutionGenome,
    parent_b: EvolutionGenome,
    *,
    rng: random.Random | None = None,
) -> EvolutionGenome:
    """Combine analyst roster and weights from two parents."""

    rng = rng or random
    analysts = sorted(set(parent_a.analysts + parent_b.analysts))
    weights = {}
    for analyst in analysts:
        weight_a = parent_a.analyst_weights.get(analyst, 0.0)
        weight_b = parent_b.analyst_weights.get(analyst, 0.0)
        blend = rng.random()
        weights[analyst] = blend * weight_a + (1 - blend) * weight_b
        if weights[analyst] == 0.0:
            weights[analyst] = 1.0 / len(analysts)
    trait_toggles = dict(parent_a.trait_toggles)
    for key, value in parent_b.trait_toggles.items():
        trait_toggles[key] = trait_toggles.get(key, value if rng.random() < 0.5 else value)
    metadata = dict(parent_a.metadata)
    metadata.update(parent_b.metadata)
    child = EvolutionGenome(
        genome_id=f"{parent_a.genome_id}_x_{parent_b.genome_id}",
        label=f"{parent_a.label}×{parent_b.label}",
        generation_id=parent_a.generation_id,
        analysts=analysts,
        prompt_variants={**parent_a.prompt_variants, **parent_b.prompt_variants},
        trait_toggles=trait_toggles,
        analyst_weights=_normalise_weights(weights),
        async_mode=parent_a.async_mode,
        patriarch_variant=parent_a.patriarch_variant or parent_b.patriarch_variant,
        risk_params={**parent_a.risk_params, **parent_b.risk_params},
        portfolio_params={**parent_a.portfolio_params, **parent_b.portfolio_params},
        execution_params={**parent_a.execution_params, **parent_b.execution_params},
        novelty_penalty=None,
        parent_ids=[parent_a.genome_id, parent_b.genome_id],
        metadata=metadata,
    )
    return child


def mutate_genome(
    genome: EvolutionGenome,
    *,
    rng: random.Random | None = None,
    available_analysts: Sequence[str] | None = None,
    perturb_strength: float = 0.1,
) -> EvolutionGenome:
    rng = rng or random
    mutated = perturb_weights(genome, strength=perturb_strength, rng=rng)
    if available_analysts:
        if rng.random() < 0.5:
            mutated = swap_analyst(mutated, available_analysts, rng=rng)
    if rng.random() < 0.3:
        mutated = toggle_traits(mutated, ["async_override"])
    mutated.parent_ids = list(mutated.parent_ids) + [genome.genome_id]
    mutated.metadata = dict(mutated.metadata)
    mutated.metadata.setdefault("mutation_notes", []).append(
        f"mutated_from={genome.genome_id}"
    )
    mutated.compute_hash()
    return mutated
