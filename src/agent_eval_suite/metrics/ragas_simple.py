"""RAGAS metric layer (real + deterministic mock).

Real:  compute_metrics       — requires `ragas` + OPENAI_API_KEY
Mock:  compute_metrics_mock  — deterministic, no LLM, safe for CI
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from agent_eval_suite.harness.config import EvalConfig
    from agent_eval_suite.harness.models import EvalCase, Scorecard

_METRIC_COLS = ("answer_relevancy", "faithfulness", "context_recall")


# ---------------------------------------------------------------------------
# Mock (deterministic, no network)
# ---------------------------------------------------------------------------

def _mock_scores(answer: str, reference_contexts: list[str], ground_truth: str | None) -> dict:
    answer_relevancy = min(1.0, len(answer) / 200)

    faithfulness = (
        1.0
        if any(
            word in answer.lower()
            for ctx in reference_contexts
            for word in ctx.lower().split()[:5]
        )
        else 0.5
    )

    context_recall = (
        1.0
        if ground_truth
        and any(
            ground_truth.lower() in ctx.lower()
            for ctx in (reference_contexts or [])
        )
        else 0.0
    )

    return {
        "answer_relevancy": answer_relevancy,
        "faithfulness": faithfulness,
        "context_recall": context_recall,
    }


def compute_metrics_mock(
    scorecard: "Scorecard",
    cases: list["EvalCase"],
    config: "EvalConfig",
) -> "Scorecard":
    """Populate scorecard with deterministic RAGAS-proxy scores (no LLM)."""
    results = scorecard.metadata.get("_results", {})

    rows: list[dict] = []
    for case in cases:
        result = results.get(case.id)
        if result is None or result.error:
            rows.append({col: float("nan") for col in _METRIC_COLS})
            continue

        rows.append(
            _mock_scores(
                answer=result.answer,
                reference_contexts=case.reference_contexts or [],
                ground_truth=case.ground_truth,
            )
        )

    scores_df = pd.DataFrame(rows)
    for col in _METRIC_COLS:
        scorecard.per_case[col] = scores_df[col].values

    for col in _METRIC_COLS:
        scorecard.aggregate[col] = float(scorecard.per_case[col].mean(skipna=True))

    return scorecard


# ---------------------------------------------------------------------------
# Real RAGAS (opt-in; requires `pip install agent-eval-suite[metrics]`)
# ---------------------------------------------------------------------------

def compute_metrics(
    scorecard: "Scorecard",
    cases: list["EvalCase"],
    config: "EvalConfig",
) -> "Scorecard":
    """Populate scorecard using real RAGAS metrics. Requires ragas + OPENAI_API_KEY."""
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_recall, faithfulness
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(
            "RAGAS is not installed. Run: pip install agent-eval-suite[metrics]"
        ) from exc

    results = scorecard.metadata.get("_results", {})

    rows: list[dict] = []
    error_indices: list[int] = []

    for i, case in enumerate(cases):
        result = results.get(case.id)
        if result is None or result.error:
            error_indices.append(i)
            rows.append(
                {
                    "question": case.question,
                    "answer": "",
                    "contexts": [],
                    "ground_truth": case.ground_truth or "",
                }
            )
        else:
            rows.append(
                {
                    "question": case.question,
                    "answer": result.answer,
                    "contexts": case.reference_contexts or [],
                    "ground_truth": case.ground_truth or "",
                }
            )

    dataset = Dataset.from_list(rows)
    result_ds = evaluate(
        dataset,
        metrics=[answer_relevancy, faithfulness, context_recall],
    )
    result_df = result_ds.to_pandas()

    for col, ragas_col in [
        ("answer_relevancy", "answer_relevancy"),
        ("faithfulness", "faithfulness"),
        ("context_recall", "context_recall"),
    ]:
        values = result_df[ragas_col].tolist()
        for idx in error_indices:
            values[idx] = float("nan")
        scorecard.per_case[col] = values

    for col in _METRIC_COLS:
        scorecard.aggregate[col] = float(scorecard.per_case[col].mean(skipna=True))

    return scorecard
