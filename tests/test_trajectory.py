"""Tests for trajectory & agent-specific metrics (no LLM, no network).

Covers:
- Four columns added to per_case
- tool_call_accuracy: fraction of expected tools called
- tool_call_accuracy: NaN when no expected_tools
- tool_call_order_correct: True/False/NaN
- step_count: total steps including non-tool steps
- redundant_steps: excess steps beyond minimum
- Error cases → NaN / -1 sentinels
"""

from __future__ import annotations

import math

import pytest

from agent_eval_suite import AgentResult, EvalCase, EvalRunner, StepRecord
from agent_eval_suite.harness.config import EvalConfig
from agent_eval_suite.metrics.trajectory import _COLS, compute_trajectory_metrics


def _make_result(steps: list[StepRecord], error: str | None = None) -> AgentResult:
    return AgentResult(answer="ok", trajectory=steps, error=error)


def _run(cases, agent):
    config = EvalConfig()
    sc = EvalRunner(agent, config).run(cases)
    return compute_trajectory_metrics(sc, cases, config)


def _tool(name: str) -> StepRecord:
    return StepRecord(step_type="tool_call", name=name)


def _final() -> StepRecord:
    return StepRecord(step_type="final", output="done")


# ---------------------------------------------------------------------------
# Column presence
# ---------------------------------------------------------------------------

def test_all_columns_added():
    case = EvalCase(id="c", question="q", expected_tools=["search"])
    agent = lambda q: _make_result([_tool("search"), _final()])
    sc = _run([case], agent)
    for col in _COLS:
        assert col in sc.per_case.columns, f"missing column: {col}"


# ---------------------------------------------------------------------------
# tool_call_accuracy
# ---------------------------------------------------------------------------

def test_accuracy_all_expected_called():
    case = EvalCase(id="c", question="q", expected_tools=["search", "calc"])
    agent = lambda q: _make_result([_tool("search"), _tool("calc"), _final()])
    sc = _run([case], agent)
    assert sc.per_case.loc[0, "tool_call_accuracy"] == 1.0


def test_accuracy_partial():
    case = EvalCase(id="c", question="q", expected_tools=["search", "calc"])
    agent = lambda q: _make_result([_tool("search"), _final()])
    sc = _run([case], agent)
    assert sc.per_case.loc[0, "tool_call_accuracy"] == pytest.approx(0.5)


def test_accuracy_none_called():
    case = EvalCase(id="c", question="q", expected_tools=["search"])
    agent = lambda q: _make_result([_final()])
    sc = _run([case], agent)
    assert sc.per_case.loc[0, "tool_call_accuracy"] == 0.0


def test_accuracy_nan_when_no_expected_tools():
    case = EvalCase(id="c", question="q", expected_tools=None)
    agent = lambda q: _make_result([_tool("search"), _final()])
    sc = _run([case], agent)
    assert math.isnan(sc.per_case.loc[0, "tool_call_accuracy"])


def test_accuracy_extra_unexpected_tools_dont_hurt():
    case = EvalCase(id="c", question="q", expected_tools=["search"])
    agent = lambda q: _make_result([_tool("search"), _tool("extra"), _final()])
    sc = _run([case], agent)
    assert sc.per_case.loc[0, "tool_call_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# tool_call_order_correct
# ---------------------------------------------------------------------------

def test_order_correct_matching():
    case = EvalCase(id="c", question="q", expected_tools=["a", "b", "c"])
    agent = lambda q: _make_result([_tool("a"), _tool("b"), _tool("c")])
    sc = _run([case], agent)
    assert sc.per_case.loc[0, "tool_call_order_correct"] == True


def test_order_correct_with_interleaved_steps():
    # Extra steps between expected tools still pass if relative order holds
    case = EvalCase(id="c", question="q", expected_tools=["a", "c"])
    agent = lambda q: _make_result([_tool("a"), _tool("b"), _tool("c")])
    sc = _run([case], agent)
    assert sc.per_case.loc[0, "tool_call_order_correct"] == True


def test_order_wrong():
    case = EvalCase(id="c", question="q", expected_tools=["a", "b"])
    agent = lambda q: _make_result([_tool("b"), _tool("a")])
    sc = _run([case], agent)
    assert sc.per_case.loc[0, "tool_call_order_correct"] == False


def test_order_nan_when_no_expected_tools():
    case = EvalCase(id="c", question="q", expected_tools=None)
    agent = lambda q: _make_result([_tool("x")])
    sc = _run([case], agent)
    assert math.isnan(sc.per_case.loc[0, "tool_call_order_correct"])


# ---------------------------------------------------------------------------
# step_count
# ---------------------------------------------------------------------------

def test_step_count_includes_all_step_types():
    case = EvalCase(id="c", question="q", expected_tools=["search"])
    steps = [_tool("search"), StepRecord(step_type="llm"), _final()]
    agent = lambda q: _make_result(steps)
    sc = _run([case], agent)
    assert sc.per_case.loc[0, "step_count"] == 3


def test_step_count_zero_for_empty_trajectory():
    case = EvalCase(id="c", question="q")
    agent = lambda q: _make_result([])
    sc = _run([case], agent)
    assert sc.per_case.loc[0, "step_count"] == 0


# ---------------------------------------------------------------------------
# redundant_steps
# ---------------------------------------------------------------------------

def test_redundant_steps_none():
    case = EvalCase(id="c", question="q", expected_tools=["a", "b"])
    agent = lambda q: _make_result([_tool("a"), _tool("b")])
    sc = _run([case], agent)
    assert sc.per_case.loc[0, "redundant_steps"] == 0


def test_redundant_steps_positive():
    case = EvalCase(id="c", question="q", expected_tools=["a"])
    agent = lambda q: _make_result([_tool("a"), _tool("b"), _tool("c"), _final()])
    sc = _run([case], agent)
    assert sc.per_case.loc[0, "redundant_steps"] == 3  # 4 steps − 1 expected


def test_redundant_steps_never_negative():
    # Fewer steps than expected → redundant = 0, not negative
    case = EvalCase(id="c", question="q", expected_tools=["a", "b", "c"])
    agent = lambda q: _make_result([_tool("a")])
    sc = _run([case], agent)
    assert sc.per_case.loc[0, "redundant_steps"] == 0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_error_case_sentinels():
    def flaky(q):
        raise RuntimeError("boom")

    case = EvalCase(id="err", question="q", expected_tools=["search"])
    config = EvalConfig()
    sc = EvalRunner(flaky, config).run([case])
    sc = compute_trajectory_metrics(sc, [case], config)

    row = sc.per_case.loc[0]
    assert math.isnan(row["tool_call_accuracy"])
    assert math.isnan(row["tool_call_order_correct"])
    assert row["step_count"] == -1
    assert row["redundant_steps"] == -1


# ---------------------------------------------------------------------------
# All prior tests still unaffected (smoke check via import)
# ---------------------------------------------------------------------------

def test_does_not_affect_other_columns():
    case = EvalCase(id="c", question="q", ground_truth="Paris", expected_tools=["search"])
    agent = lambda q: _make_result([_tool("search")])
    config = EvalConfig()
    sc = EvalRunner(agent, config).run([case])
    # Columns from runner should still be present
    assert "latency_s" in sc.per_case.columns
    assert "error" in sc.per_case.columns
    sc = compute_trajectory_metrics(sc, [case], config)
    assert "latency_s" in sc.per_case.columns
