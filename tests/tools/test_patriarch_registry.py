from __future__ import annotations

from pathlib import Path

from src.tools.patriarch_registry import PatriarchConfig, PatriarchRegistry


def test_registry_roundtrip(tmp_path: Path) -> None:
    registry = PatriarchRegistry(root=tmp_path / "patriarchs")
    config = PatriarchConfig(
        patriarch_id="baseline",
        label="Baseline Patriarch",
        prompt_path=registry.prompt_path("baseline"),
        aggregation_mode="weighted_majority",
        overrides_policy="respect_guardrails",
        metadata={"notes": "seed meta-agent"},
    )
    registry.save(config, prompt="You arbitrate analyst signals with discipline.")

    loaded = registry.load("baseline")
    assert loaded.patriarch_id == "baseline"
    assert loaded.metadata["notes"] == "seed meta-agent"
    assert "with discipline" in loaded.load_prompt()
