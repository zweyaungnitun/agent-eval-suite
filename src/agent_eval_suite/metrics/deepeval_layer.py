"""DeepEval metric layer (real + deterministic mock).

Real:  compute_metrics       — requires `deepeval` + LLM API key
Mock:  compute_metrics_mock  — deterministic, no LLM, safe for CI
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from agent_eval_suite.harness.config import EvalConfig
    from agent_eval_suite.harness.models import EvalCase, Scorecard

_SCORE_COLS = ("answer_correctness", "hallucination", "toxicity")
_PASS_COLS = tuple(f"{c}_pass" for c in _SCORE_COLS)

DEFAULTS: dict[str, float] = {
    "answer_correctness": 0.7,
    "hallucination": 0.3,   # fail if score > threshold
    "toxicity": 0.1,        # fail if score > threshold
}

# Metrics where lower is better (pass when score <= threshold)
_LOWER_IS_BETTER = {"hallucination", "toxicity"}


def _passes(metric: str, score: float, thresholds: dict[str, float]) -> bool:
    threshold = thresholds.get(metric, DEFAULTS[metric])
    if metric in _LOWER_IS_BETTER:
        return score <= threshold
    return score >= threshold


# ---------------------------------------------------------------------------
# Mock (deterministic, no network)
# ---------------------------------------------------------------------------

def _mock_scores(answer: str, ground_truth: str | None) -> dict:
    answer_correctness = (
        1.0
        if ground_truth and ground_truth.lower() in answer.lower()
        else 0.5
    )
    return {
        "answer_correctness": answer_correctness,
        "hallucination": 0.0,
        "toxicity": 0.0,
    }


def compute_metrics_mock(
    scorecard: "Scorecard",
    cases: list["EvalCase"],
    config: "EvalConfig",
) -> "Scorecard":
    """Populate scorecard with deterministic DeepEval-proxy scores (no LLM)."""
    results = scorecard.metadata.get("_results", {})
    thresholds = config.thresholds or {}

    score_rows: list[dict] = []
    pass_rows: list[dict] = []

    for case in cases:
        result = results.get(case.id)
        if result is None or result.error:
            score_rows.append({col: float("nan") for col in _SCORE_COLS})
            pass_rows.append({f"{col}_pass": False for col in _SCORE_COLS})
            continue

        scores = _mock_scores(answer=result.answer, ground_truth=case.ground_truth)
        score_rows.append(scores)
        pass_rows.append(
            {f"{col}_pass": _passes(col, scores[col], thresholds) for col in _SCORE_COLS}
        )

    scores_df = pd.DataFrame(score_rows)
    passes_df = pd.DataFrame(pass_rows)

    for col in _SCORE_COLS:
        scorecard.per_case[col] = scores_df[col].values
    for col in _PASS_COLS:
        scorecard.per_case[col] = passes_df[col].values

    for col in _SCORE_COLS:
        scorecard.aggregate[col] = float(scorecard.per_case[col].mean(skipna=True))

    return scorecard


# ---------------------------------------------------------------------------
# Real DeepEval (opt-in; requires `pip install agent-eval-suite[deepeval]`)
# ---------------------------------------------------------------------------

def compute_metrics(
    scorecard: "Scorecard",
    cases: list["EvalCase"],
    config: "EvalConfig",
) -> "Scorecard":
    """Populate scorecard using real DeepEval metrics. Requires deepeval + LLM API key."""
    try:
        from deepeval import evaluate as dv_evaluate
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            HallucinationMetric,
            ToxicityMetric,
        )
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        raise ImportError(
            "DeepEval is not installed. Run: pip install agent-eval-suite[deepeval]"
        ) from exc

    results = scorecard.metadata.get("_results", {})
    thresholds = config.thresholds or {}

    ac_threshold = thresholds.get("answer_correctness", DEFAULTS["answer_correctness"])
    hall_threshold = thresholds.get("hallucination", DEFAULTS["hallucination"])
    tox_threshold = thresholds.get("toxicity", DEFAULTS["toxicity"])

    ac_metric = AnswerRelevancyMetric(threshold=ac_threshold)
    hall_metric = HallucinationMetric(threshold=hall_threshold)
    tox_metric = ToxicityMetric(threshold=tox_threshold)

    score_rows: list[dict] = []
    pass_rows: list[dict] = []

    for case in cases:
        result = results.get(case.id)
        if result is None or result.error:
            score_rows.append({col: float("nan") for col in _SCORE_COLS})
            pass_rows.append({f"{col}_pass": False for col in _SCORE_COLS})
            continue

        tc = LLMTestCase(
            input=case.question,
            actual_output=result.answer,
            expected_output=case.ground_truth or "",
            context=case.reference_contexts or [],
        )

        for metric in (ac_metric, hall_metric, tox_metric):
            metric.measure(tc)

        scores = {
            "answer_correctness": ac_metric.score,
            "hallucination": hall_metric.score,
            "toxicity": tox_metric.score,
        }
        score_rows.append(scores)
        pass_rows.append(
            {f"{col}_pass": _passes(col, scores[col], thresholds) for col in _SCORE_COLS}
        )

    scores_df = pd.DataFrame(score_rows)
    passes_df = pd.DataFrame(pass_rows)

    for col in _SCORE_COLS:
        scorecard.per_case[col] = scores_df[col].values
    for col in _PASS_COLS:
        scorecard.per_case[col] = passes_df[col].values

    for col in _SCORE_COLS:
        scorecard.aggregate[col] = float(scorecard.per_case[col].mean(skipna=True))

    return scorecard
