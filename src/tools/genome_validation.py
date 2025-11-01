"""Diversity scoring helpers for evolutionary genome validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .genome_registry import EvolutionGenome


def _jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    set_left = {item.lower() for item in left}
    set_right = {item.lower() for item in right}
    if not set_left and not set_right:
        return 1.0
    union = set_left | set_right
    if not union:
        return 1.0
    intersection = set_left & set_right
    return len(intersection) / len(union)


def _cosine_similarity(weights_left: Mapping[str, float], weights_right: Mapping[str, float]) -> float:
    keys = set(weights_left) | set(weights_right)
    if not keys:
        return 1.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for key in keys:
        left_value = float(weights_left.get(key, 0.0))
        right_value = float(weights_right.get(key, 0.0))
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def _entropy(weights: Mapping[str, float]) -> float:
    total = sum(abs(value) for value in weights.values())
    if total == 0.0:
        return 0.0
    entropy = 0.0
    for value in weights.values():
        proportion = abs(value) / total
        if proportion > 0.0:
            entropy -= proportion * math.log(proportion, 2)
    return entropy


def _prompt_hash_similarity(prompt_a: str | None, prompt_b: str | None) -> float | None:
    if not prompt_a or not prompt_b:
        return None
    if prompt_a == prompt_b:
        return 1.0
    # Placeholder heuristic: compare length overlap ratio.
    min_len = min(len(prompt_a), len(prompt_b))
    max_len = max(len(prompt_a), len(prompt_b))
    if max_len == 0:
        return 1.0
    return min_len / max_len


@dataclass
class DiversityAssessment:
    genome_id: str
    analyst_overlap: float
    weight_similarity: float | None
    prompt_similarity: float | None
    weight_entropy: float
    warnings: list[str]


def assess_genome_diversity(
    genome: EvolutionGenome,
    cohort: Sequence[EvolutionGenome],
    *,
    analyst_overlap_threshold: float = 0.8,
    weight_similarity_threshold: float = 0.9,
) -> DiversityAssessment:
    warnings: list[str] = []
    overlap_scores = [
        _jaccard_similarity(genome.analysts, other.analysts)
        for other in cohort
        if other.genome_id != genome.genome_id
    ]
    analyst_overlap = max(overlap_scores, default=0.0)
    if analyst_overlap > analyst_overlap_threshold:
        warnings.append(
            f"High analyst overlap detected (max Jaccard {analyst_overlap:.2f} > {analyst_overlap_threshold:.2f})."
        )

    weight_similarity_scores = [
        _cosine_similarity(genome.analyst_weights, other.analyst_weights)
        for other in cohort
        if other.genome_id != genome.genome_id
    ]
    weight_similarity = max(weight_similarity_scores, default=None)
    if (
        weight_similarity is not None
        and weight_similarity > weight_similarity_threshold
        and genome.analyst_weights
        and any(other.analyst_weights for other in cohort if other.genome_id != genome.genome_id)
    ):
        warnings.append(
            f"Weight vectors nearly identical to cohort (max cosine {weight_similarity:.2f} > {weight_similarity_threshold:.2f})."
        )

    prompt_similarity_scores = [
        _prompt_hash_similarity(
            (genome.metadata or {}).get("prompt_summary"),
            (other.metadata or {}).get("prompt_summary"),
        )
        for other in cohort
        if other.genome_id != genome.genome_id
    ]
    prompt_similarity = None
    filtered_prompt_scores = [score for score in prompt_similarity_scores if score is not None]
    if filtered_prompt_scores:
        prompt_similarity = max(filtered_prompt_scores)
        if prompt_similarity > 0.95:
            warnings.append("Prompt summaries appear identical; consider diversifying instructions.")

    entropy = _entropy(genome.analyst_weights)
    if entropy == 0.0 and genome.analyst_weights:
        warnings.append("Analyst weights collapse onto a single persona; consider spreading exposure.")

    return DiversityAssessment(
        genome_id=genome.genome_id,
        analyst_overlap=analyst_overlap,
        weight_similarity=weight_similarity,
        prompt_similarity=prompt_similarity,
        weight_entropy=entropy,
        warnings=warnings,
    )


def validate_diversity(
    candidate: EvolutionGenome,
    existing: Sequence[EvolutionGenome],
    *,
    analyst_overlap_threshold: float = 0.8,
    weight_similarity_threshold: float = 0.9,
) -> tuple[bool, DiversityAssessment]:
    assessment = assess_genome_diversity(
        candidate,
        existing,
        analyst_overlap_threshold=analyst_overlap_threshold,
        weight_similarity_threshold=weight_similarity_threshold,
    )
    is_valid = not assessment.warnings
    return is_valid, assessment
