from __future__ import annotations

from src.tools.genome_registry import EvolutionGenome
from src.tools.genome_validation import assess_genome_diversity, validate_diversity


def build_genome(genome_id: str, analysts: list[str], weights: dict[str, float], prompt_summary: str) -> EvolutionGenome:
    genome = EvolutionGenome(
        genome_id=genome_id,
        label=genome_id,
        generation_id="gen-0",
        analysts=analysts,
        analyst_weights=weights,
    )
    genome.metadata["prompt_summary"] = prompt_summary
    return genome


def test_assess_genome_diversity_detects_overlap() -> None:
    base = build_genome("base", ["ben_graham", "warren_buffett"], {"ben_graham": 0.6, "warren_buffett": 0.4}, "value baseline")
    variant = build_genome("variant", ["ben_graham", "warren_buffett"], {"ben_graham": 0.59, "warren_buffett": 0.41}, "value baseline")

    assessment = assess_genome_diversity(variant, [base])
    assert assessment.analyst_overlap == 1.0
    assert assessment.weight_similarity is not None and assessment.weight_similarity > 0.99
    assert assessment.prompt_similarity == 1.0
    assert assessment.warnings, "Expected overlap warnings"


def test_validate_diversity_accepts_unique_genome() -> None:
    base = build_genome("base", ["ben_graham", "warren_buffett"], {"ben_graham": 0.6, "warren_buffett": 0.4}, "value baseline")
    challenger = build_genome("growth", ["cathie_wood", "peter_lynch"], {"cathie_wood": 0.7, "peter_lynch": 0.3}, "growth challenger")

    is_valid, assessment = validate_diversity(challenger, [base])
    assert is_valid
    assert assessment.analyst_overlap == 0.0
    assert assessment.weight_similarity is not None and assessment.weight_similarity < 0.5
    assert assessment.prompt_similarity is not None and assessment.prompt_similarity < 0.9
