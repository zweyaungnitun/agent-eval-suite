"""Sprint 0 walking-skeleton test: stub agent -> well-formed empty scorecard.

No LLM, no network. Proves the harness spine: adapter -> runner -> scorecard,
and that per-case exceptions are isolated (specs/01 Definition of Done).
"""

from agent_eval_suite import AgentResult, EvalCase, EvalRunner, StepRecord


def stub_agent(query: str) -> AgentResult:
    return AgentResult(
        answer=f"answer to {query}",
        retrieved_contexts=["ctx"],
        trajectory=[StepRecord(step_type="final", output="done")],
    )


DATASET = [
    EvalCase(id="q1", question="What is 2+2?", ground_truth="4"),
    EvalCase(id="q2", question="Capital of France?", ground_truth="Paris"),
    EvalCase(id="q3", question="Color of the sky?", ground_truth="Blue"),
]


def test_runner_produces_wellformed_scorecard():
    scorecard = EvalRunner(stub_agent).run(DATASET)

    assert len(scorecard.per_case) == 3
    assert list(scorecard.per_case["id"]) == ["q1", "q2", "q3"]
    assert scorecard.metadata["n_cases"] == 3
    assert scorecard.metadata["n_errors"] == 0
    # No metric columns yet — but the structural columns exist.
    assert "latency_s" in scorecard.per_case.columns


def test_per_case_exceptions_are_isolated():
    def flaky(query: str) -> AgentResult:
        if "boom" in query:
            raise RuntimeError("kaboom")
        return AgentResult(answer="ok")

    dataset = [
        EvalCase(id="ok1", question="fine"),
        EvalCase(id="bad", question="boom"),
        EvalCase(id="ok2", question="also fine"),
    ]
    scorecard = EvalRunner(flaky).run(dataset)

    assert len(scorecard.per_case) == 3  # run did not abort
    assert scorecard.metadata["n_errors"] == 1
    import pandas as pd

    errors = dict(zip(scorecard.per_case["id"], scorecard.per_case["error"]))
    assert pd.notna(errors["bad"]) and "kaboom" in errors["bad"]
    assert pd.isna(errors["ok1"])


def test_string_and_dict_outputs_coerce():
    scorecard = EvalRunner(lambda q: "plain string").run([EvalCase(id="s", question="q")])
    assert scorecard.metadata["n_errors"] == 0

    scorecard = EvalRunner(lambda q: {"answer": "a", "retrieved_contexts": ["c"]}).run(
        [EvalCase(id="d", question="q")]
    )
    assert scorecard.metadata["n_errors"] == 0


def test_exports(tmp_path):
    scorecard = EvalRunner(stub_agent).run(DATASET)
    csv_path = tmp_path / "sc.csv"
    json_path = tmp_path / "sc.json"
    scorecard.to_csv(str(csv_path))
    scorecard.to_json(str(json_path))
    assert csv_path.exists() and json_path.exists()
    assert csv_path.read_text().startswith("id,question")
