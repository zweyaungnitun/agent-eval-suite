"""Core data models shared across every metric layer.

These are the contract from Phase 1 (specs/01-harness-interface.md). Each later
phase reads ``AgentResult`` / ``EvalCase`` and writes columns into ``Scorecard``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import pandas as pd


@dataclass
class StepRecord:
    """One step in an agent's trajectory (tool call, llm turn, retrieval...)."""

    step_type: str  # "tool_call" | "llm" | "retrieval" | "final"
    name: str | None = None  # tool name, if applicable
    args: dict | None = None
    output: str | None = None


@dataclass
class AgentResult:
    """Normalized output of a single agent run, regardless of agent flavor."""

    answer: str
    retrieved_contexts: list[str] = field(default_factory=list)
    trajectory: list[StepRecord] = field(default_factory=list)
    raw_trace: dict | None = None
    latency_s: float = 0.0
    token_usage: dict | None = None
    error: str | None = None  # set when the case failed; answer may be empty


@dataclass
class EvalCase:
    """One test case from a dataset."""

    id: str
    question: str
    ground_truth: str | None = None
    reference_contexts: list[str] | None = None
    expected_tools: list[str] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "EvalCase":
        return cls(
            id=d["id"],
            question=d["question"],
            ground_truth=d.get("ground_truth"),
            reference_contexts=d.get("reference_contexts"),
            expected_tools=d.get("expected_tools"),
        )


def load_dataset(path: str) -> list[EvalCase]:
    """Load a list of EvalCase from a JSON file."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return [EvalCase.from_dict(d) for d in raw]


class AgentCallable(Protocol):
    """The single interface every adapter normalizes an agent to."""

    def run(self, query: str) -> AgentResult:  # pragma: no cover - protocol
        ...


@dataclass
class Scorecard:
    """Per-case + aggregate results. Metric layers add columns to ``per_case``."""

    per_case: pd.DataFrame
    aggregate: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_csv(self, path: str) -> None:
        self.per_case.to_csv(path, index=False)

    def to_json(self, path: str) -> None:
        # Drop private keys (e.g. _results holding AgentResult objects) from export.
        public_meta = {k: v for k, v in self.metadata.items() if not k.startswith("_")}
        payload = {
            "aggregate": self.aggregate,
            "metadata": public_meta,
            "per_case": self.per_case.to_dict(orient="records"),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

    @classmethod
    def empty(cls, cases: list[EvalCase], metadata: dict | None = None) -> "Scorecard":
        """A well-formed scorecard with one row per case and no metric columns yet."""
        df = pd.DataFrame([{"id": c.id, "question": c.question} for c in cases])
        return cls(per_case=df, aggregate={}, metadata=metadata or {})
