"""Phase 2 dataset tests: schema, uniqueness, load_dataset round-trip.

No LLM, no network. Validates specs/02-synthetic-testset.md Definition of Done.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval_suite import AgentResult, EvalRunner, load_dataset
from agent_eval_suite.harness.models import EvalCase

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "wiki_50.json"


@pytest.fixture(scope="module")
def raw_records() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dataset() -> list[EvalCase]:
    return load_dataset(str(DATASET_PATH))


# ---------------------------------------------------------------------------
# Schema and content validation
# ---------------------------------------------------------------------------


def test_dataset_has_50_records(dataset):
    assert len(dataset) == 50


def test_all_ids_unique(raw_records):
    ids = [r["id"] for r in raw_records]
    assert len(ids) == len(set(ids)), "Duplicate ids found"


def test_id_format(raw_records):
    for r in raw_records:
        assert r["id"].startswith("wiki_"), f"Bad id format: {r['id']}"


def test_all_questions_non_empty(dataset):
    for case in dataset:
        assert case.question.strip(), f"{case.id}: empty question"


def test_all_ground_truths_non_empty(dataset):
    for case in dataset:
        assert case.ground_truth and case.ground_truth.strip(), f"{case.id}: empty ground_truth"


def test_all_have_at_least_one_reference_context(dataset):
    for case in dataset:
        assert case.reference_contexts, f"{case.id}: no reference_contexts"
        for ctx in case.reference_contexts:
            assert ctx.strip(), f"{case.id}: blank reference_context entry"


def test_expected_tools_is_null(raw_records):
    for r in raw_records:
        assert r.get("expected_tools") is None, f"{r['id']}: expected_tools should be null"


# ---------------------------------------------------------------------------
# load_dataset round-trip
# ---------------------------------------------------------------------------


def test_load_dataset_returns_eval_cases(dataset):
    assert all(isinstance(c, EvalCase) for c in dataset)


def test_load_dataset_preserves_reference_contexts(dataset):
    for case in dataset:
        assert isinstance(case.reference_contexts, list)
        assert len(case.reference_contexts) >= 1


# ---------------------------------------------------------------------------
# End-to-end: stub agent runs the full dataset without errors
# ---------------------------------------------------------------------------


def test_stub_agent_runs_full_dataset(dataset):
    def stub(query: str) -> AgentResult:
        return AgentResult(answer=f"stub: {query[:20]}")

    scorecard = EvalRunner(stub).run(dataset)

    assert scorecard.metadata["n_cases"] == 50
    assert scorecard.metadata["n_errors"] == 0
    assert len(scorecard.per_case) == 50
    assert "latency_s" in scorecard.per_case.columns


# ---------------------------------------------------------------------------
# generate_testset_demo validation mode
# ---------------------------------------------------------------------------


def test_generate_demo_validate_only(capsys):
    from datasets.generate_testset_demo import main

    main(["--validate-only"])
    captured = capsys.readouterr()
    assert "OK" in captured.out
    assert "50 records" in captured.out
