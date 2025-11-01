"""Registry utilities for patriarch (meta-agent) configurations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class PatriarchConfig:
    patriarch_id: str
    label: str
    prompt_path: Path
    aggregation_mode: str = "weighted_majority"
    overrides_policy: str = "respect_guardrails"
    metadata: dict[str, Any] = field(default_factory=dict)

    def load_prompt(self) -> str:
        return self.prompt_path.read_text(encoding="utf-8")


class PatriarchRegistry:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path("configs") / "patriarchs"
        self.prompt_root = self.root / "prompts"
        self.root.mkdir(parents=True, exist_ok=True)
        self.prompt_root.mkdir(parents=True, exist_ok=True)

    def config_path(self, patriarch_id: str) -> Path:
        return self.root / f"{patriarch_id}.json"

    def prompt_path(self, patriarch_id: str) -> Path:
        return self.prompt_root / f"{patriarch_id}.txt"

    def load(self, patriarch_id: str) -> PatriarchConfig:
        config_path = self.config_path(patriarch_id)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        prompt_path = self.prompt_path(patriarch_id)
        if payload.get("prompt_path"):
            prompt_path = Path(payload["prompt_path"])
        return PatriarchConfig(
            patriarch_id=patriarch_id,
            label=payload.get("label", patriarch_id),
            prompt_path=prompt_path,
            aggregation_mode=payload.get("aggregation_mode", "weighted_majority"),
            overrides_policy=payload.get("overrides_policy", "respect_guardrails"),
            metadata=dict(payload.get("metadata", {})),
        )

    def save(self, config: PatriarchConfig, *, prompt: str | None = None) -> Path:
        payload: dict[str, Any] = {
            "patriarch_id": config.patriarch_id,
            "label": config.label,
            "prompt_path": str(config.prompt_path),
            "aggregation_mode": config.aggregation_mode,
            "overrides_policy": config.overrides_policy,
            "metadata": config.metadata,
        }
        config_path = self.config_path(config.patriarch_id)
        config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        if prompt is not None:
            config.prompt_path.write_text(prompt, encoding="utf-8")
        return config_path
