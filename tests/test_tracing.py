"""Tests for the observability hook (Phase 6, specs/06-observability.md).

Uses a spy tracer — no network, no LangSmith package needed.
Verifies: hook call order, correct arguments, fallback to NoOpTracer,
          unknown-backend warning, and that existing tests still pass.
"""

from __future__ import annotations

import warnings

import pytest

from agent_eval_suite import AgentResult, EvalCase, EvalRunner, StepRecord
from agent_eval_suite.harness.config import EvalConfig
from agent_eval_suite.harness.models import Scorecard
from agent_eval_suite.harness.tracing import NoOpTracer, Tracer, build_tracer


# ---------------------------------------------------------------------------
# Spy tracer
# ---------------------------------------------------------------------------

class SpyTracer:
    """Records every hook call for assertion."""

    def __init__(self):
        self.calls: list[tuple] = []

    def on_run_start(self, config: EvalConfig, n_cases: int) -> None:
        self.calls.append(("on_run_start", n_cases))

    def on_case_start(self, case: EvalCase) -> None:
        self.calls.append(("on_case_start", case.id))

    def on_case_end(self, case: EvalCase, result: AgentResult) -> None:
        self.calls.append(("on_case_end", case.id, result.error is None))

    def on_run_end(self, scorecard: Scorecard) -> None:
        self.calls.append(("on_run_end", len(scorecard.per_case)))

    def event_names(self) -> list[str]:
        return [c[0] for c in self.calls]


CASES = [
    EvalCase(id="q1", question="What is 2+2?", ground_truth="4"),
    EvalCase(id="q2", question="Capital of France?", ground_truth="Paris"),
]


def _stub(query: str) -> AgentResult:
    return AgentResult(answer=f"answer: {query}")


# ---------------------------------------------------------------------------
# Hook order
# ---------------------------------------------------------------------------

def test_hooks_fire_in_correct_order():
    spy = SpyTracer()
    EvalRunner(_stub, tracer=spy).run(CASES)

    names = spy.event_names()
    assert names[0] == "on_run_start"
    assert names[-1] == "on_run_end"
    assert names.count("on_case_start") == 2
    assert names.count("on_case_end") == 2

    # interleaved: start then end for each case
    case_events = [n for n in names if n.startswith("on_case")]
    assert case_events == [
        "on_case_start", "on_case_end",
        "on_case_start", "on_case_end",
    ]


def test_on_run_start_receives_case_count():
    spy = SpyTracer()
    EvalRunner(_stub, tracer=spy).run(CASES)
    run_start = next(c for c in spy.calls if c[0] == "on_run_start")
    assert run_start[1] == 2


def test_on_case_hooks_receive_correct_ids():
    spy = SpyTracer()
    EvalRunner(_stub, tracer=spy).run(CASES)
    starts = [c[1] for c in spy.calls if c[0] == "on_case_start"]
    ends = [c[1] for c in spy.calls if c[0] == "on_case_end"]
    assert starts == ["q1", "q2"]
    assert ends == ["q1", "q2"]


def test_on_run_end_receives_scorecard_row_count():
    spy = SpyTracer()
    EvalRunner(_stub, tracer=spy).run(CASES)
    run_end = next(c for c in spy.calls if c[0] == "on_run_end")
    assert run_end[1] == 2


# ---------------------------------------------------------------------------
# Error isolation: tracer still fires even when a case errors
# ---------------------------------------------------------------------------

def test_hooks_fire_for_error_cases():
    def flaky(q: str) -> AgentResult:
        if "boom" in q:
            raise RuntimeError("kaboom")
        return AgentResult(answer="ok")

    cases = [
        EvalCase(id="ok", question="fine"),
        EvalCase(id="bad", question="boom"),
    ]
    spy = SpyTracer()
    EvalRunner(flaky, tracer=spy).run(cases)

    ends = {c[1]: c[2] for c in spy.calls if c[0] == "on_case_end"}
    assert ends["ok"] is True   # no error
    assert ends["bad"] is False  # had error


# ---------------------------------------------------------------------------
# build_tracer factory
# ---------------------------------------------------------------------------

def test_build_tracer_none_returns_noop():
    tracer = build_tracer({"backend": "none"})
    assert isinstance(tracer, NoOpTracer)


def test_build_tracer_empty_dict_returns_noop():
    tracer = build_tracer({})
    assert isinstance(tracer, NoOpTracer)


def test_build_tracer_unknown_backend_warns_and_falls_back():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tracer = build_tracer({"backend": "nonexistent_backend"})
    assert isinstance(tracer, NoOpTracer)
    assert any("nonexistent_backend" in str(warning.message) for warning in w)


def test_build_tracer_langsmith_falls_back_when_not_installed(monkeypatch):
    import sys
    # Simulate langsmith not being installed
    monkeypatch.setitem(sys.modules, "langsmith", None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tracer = build_tracer({"backend": "langsmith", "project": "test"})
    assert isinstance(tracer, NoOpTracer)


def test_build_tracer_phoenix_falls_back_when_not_installed(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tracer = build_tracer({"backend": "phoenix"})
    assert isinstance(tracer, NoOpTracer)


# ---------------------------------------------------------------------------
# Tracer protocol compliance
# ---------------------------------------------------------------------------

def test_noop_tracer_satisfies_protocol():
    assert isinstance(NoOpTracer(), Tracer)


def test_spy_tracer_satisfies_protocol():
    assert isinstance(SpyTracer(), Tracer)


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

def test_runner_uses_noop_by_default():
    config = EvalConfig()  # tracing: {backend: none}
    runner = EvalRunner(_stub, config)
    assert isinstance(runner.tracer, NoOpTracer)


def test_runner_accepts_explicit_tracer():
    spy = SpyTracer()
    runner = EvalRunner(_stub, tracer=spy)
    assert runner.tracer is spy
