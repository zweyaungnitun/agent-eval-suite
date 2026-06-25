"""Tests for the DeepEval metric layer (mock only — no network, no LLM).

Covers:
- Six columns added (three score + three pass)
- Aggregate means computed for score columns
- Error cases → NaN scores, False passes
- Pass/fail logic respects thresholds (higher-is-better vs lower-is-better)
- Custom thresholds via EvalConfig
"""

from __future__ import annotations

import math

import pytest

from agent_eval_suite import AgentResult, EvalCase, EvalRunner
from agent_eval_suite.harness.config import EvalConfig
from agent_eval_suite.metrics.deepeval_layer import (
    _PASS_COLS,
    _SCORE_COLS,
    compute_metrics_mock,
)

CASES = [
    EvalCase(
        id="q1",
        question="What is the capital of France?",
        ground_truth="Paris",
        reference_contexts=["Paris is the capital of France."],
    ),
    EvalCase(
        id="q2",
        question="Explain photosynthesis.",
        ground_truth="Plants convert sunlight to energy.",
        reference_contexts=["Photosynthesis converts light to chemical energy."],
    ),
]


def _good_agent(query: str) -> AgentResult:
    return AgentResult(answer=f"Paris is the capital of France. {query}")


def _run_and_score(cases=None, agent=None, thresholds=None):
    cases = cases or CASES
    agent = agent or _good_agent
    config = EvalConfig(thresholds=thresholds or {})
    scorecard = EvalRunner(agent, config).run(cases)
    return compute_metrics_mock(scorecard, cases, config), config


# ---------------------------------------------------------------------------
# Column presence
# ---------------------------------------------------------------------------

def test_score_columns_added():
    sc, _ = _run_and_score()
    for col in _SCORE_COLS:
        assert col in sc.per_case.columns, f"missing score column: {col}"


def test_pass_columns_added():
    sc, _ = _run_and_score()
    for col in _PASS_COLS:
        assert col in sc.per_case.columns, f"missing pass column: {col}"


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def test_aggregate_keys_present():
    sc, _ = _run_and_score()
    for col in _SCORE_COLS:
        assert col in sc.aggregate, f"missing aggregate key: {col}"


def test_aggregate_values_finite():
    sc, _ = _run_and_score()
    for col in _SCORE_COLS:
        assert math.isfinite(sc.aggregate[col]), f"aggregate[{col}] not finite"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_error_case_score_is_nan():
    def flaky(query: str) -> AgentResult:
        if "fail" in query:
            raise RuntimeError("boom")
        return AgentResult(answer="Paris is the capital of France.")

    err_case = EvalCase(id="err", question="fail", ground_truth="x", reference_contexts=["x"])
    ok_case = EvalCase(id="ok", question="Capital?", ground_truth="Paris", reference_contexts=["Paris"])
    cases = [ok_case, err_case]
    config = EvalConfig()
    sc = EvalRunner(flaky, config).run(cases)
    sc = compute_metrics_mock(sc, cases, config)

    df = sc.per_case.set_index("id")
    for col in _SCORE_COLS:
        assert math.isnan(df.loc["err", col]), f"{col} should be NaN for error case"
        assert math.isfinite(df.loc["ok", col]), f"{col} should be finite for ok case"


def test_error_case_pass_is_false():
    def flaky(query: str) -> AgentResult:
        if "fail" in query:
            raise RuntimeError("boom")
        return AgentResult(answer="Paris is the capital of France.")

    err_case = EvalCase(id="err", question="fail", ground_truth="x", reference_contexts=["x"])
    ok_case = EvalCase(id="ok", question="Capital?", ground_truth="Paris", reference_contexts=["Paris"])
    cases = [ok_case, err_case]
    config = EvalConfig()
    sc = EvalRunner(flaky, config).run(cases)
    sc = compute_metrics_mock(sc, cases, config)

    df = sc.per_case.set_index("id")
    for col in _PASS_COLS:
        assert df.loc["err", col] is False or df.loc["err", col] == False


# ---------------------------------------------------------------------------
# Pass/fail logic
# ---------------------------------------------------------------------------

def test_answer_correctness_pass_when_ground_truth_in_answer():
    # ground_truth "Paris" appears in answer → score = 1.0 → passes default 0.7
    case = EvalCase(id="c", question="q", ground_truth="Paris", reference_contexts=[])
    agent = lambda q: AgentResult(answer="Paris is the answer.")
    config = EvalConfig()
    sc = EvalRunner(agent, config).run([case])
    sc = compute_metrics_mock(sc, [case], config)
    assert sc.per_case.loc[0, "answer_correctness"] == 1.0
    assert sc.per_case.loc[0, "answer_correctness_pass"] is True or sc.per_case.loc[0, "answer_correctness_pass"] == True


def test_answer_correctness_fallback_score():
    # ground_truth not in answer → score = 0.5 → fails default 0.7
    case = EvalCase(id="c", question="q", ground_truth="Berlin", reference_contexts=[])
    agent = lambda q: AgentResult(answer="Paris is the capital.")
    config = EvalConfig()
    sc = EvalRunner(agent, config).run([case])
    sc = compute_metrics_mock(sc, [case], config)
    assert sc.per_case.loc[0, "answer_correctness"] == 0.5
    assert sc.per_case.loc[0, "answer_correctness_pass"] is False or sc.per_case.loc[0, "answer_correctness_pass"] == False


def test_hallucination_and_toxicity_default_to_zero_and_pass():
    case = EvalCase(id="c", question="q", ground_truth="Paris", reference_contexts=[])
    agent = lambda q: AgentResult(answer="Paris")
    config = EvalConfig()
    sc = EvalRunner(agent, config).run([case])
    sc = compute_metrics_mock(sc, [case], config)
    assert sc.per_case.loc[0, "hallucination"] == 0.0
    assert sc.per_case.loc[0, "hallucination_pass"] == True
    assert sc.per_case.loc[0, "toxicity"] == 0.0
    assert sc.per_case.loc[0, "toxicity_pass"] == True


def test_custom_threshold_tightens_pass():
    # Lower the answer_correctness threshold to 0.6 → score 0.5 still fails
    case = EvalCase(id="c", question="q", ground_truth="Berlin", reference_contexts=[])
    agent = lambda q: AgentResult(answer="Paris is the capital.")
    config = EvalConfig(thresholds={"answer_correctness": 0.6})
    sc = EvalRunner(agent, config).run([case])
    sc = compute_metrics_mock(sc, [case], config)
    assert sc.per_case.loc[0, "answer_correctness_pass"] == False


def test_custom_threshold_relaxes_pass():
    # Lower the answer_correctness threshold to 0.4 → score 0.5 now passes
    case = EvalCase(id="c", question="q", ground_truth="Berlin", reference_contexts=[])
    agent = lambda q: AgentResult(answer="Paris is the capital.")
    config = EvalConfig(thresholds={"answer_correctness": 0.4})
    sc = EvalRunner(agent, config).run([case])
    sc = compute_metrics_mock(sc, [case], config)
    assert sc.per_case.loc[0, "answer_correctness_pass"] == True


# ---------------------------------------------------------------------------
# Aggregate skips NaN
# ---------------------------------------------------------------------------

def test_aggregate_skips_nan_rows():
    def flaky(query: str) -> AgentResult:
        if "fail" in query:
            raise RuntimeError("boom")
        return AgentResult(answer="Paris is the capital of France.")

    err_case = EvalCase(id="err", question="fail", ground_truth="x", reference_contexts=["x"])
    ok_case = EvalCase(id="ok", question="q", ground_truth="Paris", reference_contexts=["Paris"])
    cases = [ok_case, err_case]
    config = EvalConfig()
    sc = EvalRunner(flaky, config).run(cases)
    sc = compute_metrics_mock(sc, cases, config)
    for col in _SCORE_COLS:
        assert math.isfinite(sc.aggregate[col]), f"aggregate[{col}] should skip NaN"
