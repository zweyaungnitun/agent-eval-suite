"""Single source of truth for an eval run (specs/01-harness-interface.md)."""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class EvalConfig:
    model: str = "gpt-4o-mini"
    concurrency: int = 1
    dataset_path: str | None = None
    metrics: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    tracing: dict = field(default_factory=lambda: {"backend": "none"})

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(
            model=raw.get("model", "gpt-4o-mini"),
            concurrency=raw.get("concurrency", 1),
            dataset_path=raw.get("dataset_path"),
            metrics=raw.get("metrics", []),
            thresholds=raw.get("thresholds", {}),
            tracing=raw.get("tracing", {"backend": "none"}),
        )
