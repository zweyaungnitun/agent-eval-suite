"""The test runner: run an agent over a dataset, collect results, build a scorecard.

Per-case exceptions are isolated and recorded — a single bad case never aborts
the whole run (specs/01-harness-interface.md, Definition of Done).

Concurrency: EvalConfig.concurrency > 1 enables ThreadPoolExecutor parallelism.
Row order in the scorecard always matches the input dataset order.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .adapter import adapt
from .config import EvalConfig
from .models import AgentResult, EvalCase, Scorecard
from .tracing import Tracer, build_tracer


class EvalRunner:
    def __init__(self, agent: Any, config: EvalConfig | None = None, tracer: Tracer | None = None):
        self.agent = adapt(agent)
        self.config = config or EvalConfig()
        self.tracer: Tracer = tracer if tracer is not None else build_tracer(self.config.tracing)
        self._tracer_lock = threading.Lock()

    def run(self, dataset: list[EvalCase]) -> Scorecard:
        self.tracer.on_run_start(self.config, len(dataset))

        concurrency = max(1, self.config.concurrency)
        if concurrency == 1:
            results = self._run_sequential(dataset)
        else:
            results = self._run_concurrent(dataset, concurrency)

        scorecard = Scorecard.empty(
            dataset,
            metadata={
                "model": self.config.model,
                "n_cases": len(dataset),
                "n_errors": sum(1 for r in results.values() if r.error),
                "concurrency": concurrency,
            },
        )
        scorecard.per_case["latency_s"] = [results[c.id].latency_s for c in dataset]
        scorecard.per_case["error"] = [results[c.id].error for c in dataset]
        scorecard.metadata["_results"] = results

        self.tracer.on_run_end(scorecard)
        return scorecard

    def _run_sequential(self, dataset: list[EvalCase]) -> dict[str, AgentResult]:
        results: dict[str, AgentResult] = {}
        for case in dataset:
            self.tracer.on_case_start(case)
            result = self._run_case(case)
            results[case.id] = result
            self.tracer.on_case_end(case, result)
        return results

    def _run_concurrent(self, dataset: list[EvalCase], workers: int) -> dict[str, AgentResult]:
        results: dict[str, AgentResult] = {}

        def _task(case: EvalCase) -> tuple[str, AgentResult]:
            with self._tracer_lock:
                self.tracer.on_case_start(case)
            result = self._run_case(case)
            with self._tracer_lock:
                self.tracer.on_case_end(case, result)
            return case.id, result

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_task, case): case for case in dataset}
            for future in as_completed(futures):
                case_id, result = future.result()
                results[case_id] = result

        return results

    def _run_case(self, case: EvalCase) -> AgentResult:
        try:
            return self.agent.run(case.question)
        except Exception as exc:  # isolate: one bad case must not kill the run
            return AgentResult(answer="", error=f"{type(exc).__name__}: {exc}")
