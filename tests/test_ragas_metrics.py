"""Tests for the RAGAS metric layer (mock only — no network, no LLM).

Covers:
- Three metric columns added to per_case
- Aggregate means computed
- Error cases yield NaN (not 0)
- Normal cases yield finite scores in [0, 1]
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from agent_eval_suite import AgentResult, EvalCase, EvalRunner
from agent_eval_suite.harness.config import EvalConfig
from agent_eval_suite.metrics.ragas_simple import compute_metrics_mock

_METRIC_COLS = ("answer_relevancy", "faithfulness", "context_recall")

CASES = [
    EvalCase(
        id="q1",
        question="What is the capital of France?",
        ground_truth="Paris",
        reference_contexts=["Paris is the capital of France."],
    ),
    EvalCase(
        id="q2",
        question="What is photosynthesis?",
        ground_truth="A process by which plants make food using sunlight.",
        reference_contexts=["Photosynthesis is how plants convert light to energy."],
    ),
    EvalCase(
        id="q3_error",
        question="This will fail.",
        ground_truth="irrelevant",
        reference_contexts=["irrelevant"],
    ),
]


def _make_agent(error_id: str | None = None):
    def agent(query: str) -> AgentResult:
        if error_id and error_id in query:
            raise RuntimeError("forced failure")
        return AgentResult(
            answer=f"The answer is about {query} " * 5,  # ~50+ chars
            retrieved_contexts=["ctx"],
        )

    return agent


def _run_and_score(cases=None, agent=None):
    cases = cases or CASES
    if agent is None:
        agent = _make_agent(error_id="fail")
    config = EvalConfig()
    scorecard = EvalRunner(agent, config).run(cases)
    return compute_metrics_mock(scorecard, cases, config)


# ---------------------------------------------------------------------------
# Column presence
# ---------------------------------------------------------------------------

def test_metric_columns_added():
    scorecard = _run_and_score()
    for col in _METRIC_COLS:
        assert col in scorecard.per_case.columns, f"missing column: {col}"


# ---------------------------------------------------------------------------
# Aggregate keys
# ---------------------------------------------------------------------------

def test_aggregate_keys_present():
    scorecard = _run_and_score()
    for col in _METRIC_COLS:
        assert col in scorecard.aggregate, f"missing aggregate key: {col}"


def test_aggregate_values_are_finite():
    scorecard = _run_and_score(cases=CASES[:2], agent=_make_agent())
    for col in _METRIC_COLS:
        val = scorecard.aggregate[col]
        assert math.isfinite(val), f"aggregate[{col}] is not finite: {val}"


# ---------------------------------------------------------------------------
# Score range
# ---------------------------------------------------------------------------

def test_normal_scores_in_unit_interval():
    scorecard = _run_and_score(cases=CASES[:2], agent=_make_agent())
    df = scorecard.per_case
    for col in _METRIC_COLS:
        for val in df[col]:
            assert 0.0 <= val <= 1.0, f"{col}={val} out of [0,1]"


# ---------------------------------------------------------------------------
# Error cases → NaN
# ---------------------------------------------------------------------------

def test_error_case_yields_nan():
    def flaky(query: str) -> AgentResult:
        if "fail" in query:
            raise RuntimeError("boom")
        return AgentResult(answer="Paris is the capital of France. " * 3)

    error_case = EvalCase(
        id="err",
        question="This will fail.",
        ground_truth="x",
        reference_contexts=["x"],
    )
    good_case = EvalCase(
        id="ok",
        question="Capital of France?",
        ground_truth="Paris",
        reference_contexts=["Paris is the capital of France."],
    )
    cases = [good_case, error_case]
    config = EvalConfig()
    scorecard = EvalRunner(flaky, config).run(cases)
    scorecard = compute_metrics_mock(scorecard, cases, config)

    df = scorecard.per_case.set_index("id")
    for col in _METRIC_COLS:
        assert math.isnan(df.loc["err", col]), f"expected NaN for error case, got {df.loc['err', col]}"
        assert math.isfinite(df.loc["ok", col]), f"expected finite for ok case, got {df.loc['ok', col]}"


# ---------------------------------------------------------------------------
# Aggregate skips NaN
# ---------------------------------------------------------------------------

def test_aggregate_skips_nan():
    def flaky(query: str) -> AgentResult:
        if "fail" in query:
            raise RuntimeError("boom")
        return AgentResult(answer="photosynthesis is a process plants use. " * 3)

    error_case = EvalCase(id="err", question="fail", ground_truth="x", reference_contexts=["x"])
    good_case = EvalCase(
        id="ok",
        question="Photosynthesis?",
        ground_truth="photosynthesis",
        reference_contexts=["Photosynthesis is how plants convert light."],
    )
    cases = [good_case, error_case]
    config = EvalConfig()
    scorecard = EvalRunner(flaky, config).run(cases)
    scorecard = compute_metrics_mock(scorecard, cases, config)

    # Aggregate should be computed from the 1 non-NaN row only → finite
    for col in _METRIC_COLS:
        val = scorecard.aggregate[col]
        assert math.isfinite(val), f"aggregate[{col}] should be finite even with NaN rows"


# ---------------------------------------------------------------------------
# Idempotency: calling twice doesn't corrupt the scorecard
# ---------------------------------------------------------------------------

def test_double_call_is_idempotent():
    cases = CASES[:2]
    config = EvalConfig()
    scorecard = EvalRunner(_make_agent(), config).run(cases)
    scorecard = compute_metrics_mock(scorecard, cases, config)
    agg1 = dict(scorecard.aggregate)
    scorecard = compute_metrics_mock(scorecard, cases, config)
    agg2 = dict(scorecard.aggregate)
    assert agg1 == agg2
