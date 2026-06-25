"""The test runner: run an agent over a dataset, collect results, build a scorecard.

Per-case exceptions are isolated and recorded — a single bad case never aborts
the whole run (specs/01-harness-interface.md, Definition of Done).
"""

from __future__ import annotations

from typing import Any

from .adapter import adapt
from .config import EvalConfig
from .models import AgentResult, EvalCase, Scorecard
from .tracing import NoOpTracer, Tracer, build_tracer


class EvalRunner:
    def __init__(self, agent: Any, config: EvalConfig | None = None, tracer: Tracer | None = None):
        self.agent = adapt(agent)
        self.config = config or EvalConfig()
        self.tracer: Tracer = tracer if tracer is not None else build_tracer(self.config.tracing)

    def run(self, dataset: list[EvalCase]) -> Scorecard:
        self.tracer.on_run_start(self.config, len(dataset))

        results: dict[str, AgentResult] = {}
        for case in dataset:
            self.tracer.on_case_start(case)
            result = self._run_case(case)
            results[case.id] = result
            self.tracer.on_case_end(case, result)

        scorecard = Scorecard.empty(
            dataset,
            metadata={
                "model": self.config.model,
                "n_cases": len(dataset),
                "n_errors": sum(1 for r in results.values() if r.error),
            },
        )
        # Surface basic per-case run data; metric layers add their columns later.
        scorecard.per_case["latency_s"] = [results[c.id].latency_s for c in dataset]
        scorecard.per_case["error"] = [results[c.id].error for c in dataset]
        scorecard.metadata["_results"] = results  # handed to metric layers

        self.tracer.on_run_end(scorecard)
        return scorecard

    def _run_case(self, case: EvalCase) -> AgentResult:
        try:
            return self.agent.run(case.question)
        except Exception as exc:  # isolate: one bad case must not kill the run
            return AgentResult(answer="", error=f"{type(exc).__name__}: {exc}")
