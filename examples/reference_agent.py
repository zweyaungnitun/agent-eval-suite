"""Reference agent: a minimal retrieval-augmented agent for demo and benchmarking.

Runs without any API key in mock mode (--mock flag or AGENT_EVAL_MOCK=1).
With a real OPENAI_API_KEY it calls gpt-4o-mini.

Usage:
    # Mock mode (no API key, deterministic answers)
    python examples/reference_agent.py --mock

    # Real mode
    OPENAI_API_KEY=sk-... python examples/reference_agent.py

    # Run against wiki_50 dataset with all metric layers
    python examples/reference_agent.py --mock --dataset datasets/wiki_50.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the package importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_eval_suite import AgentResult, EvalCase, EvalRunner, StepRecord, load_dataset
from agent_eval_suite.harness.config import EvalConfig
from agent_eval_suite.metrics.deepeval_layer import compute_metrics_mock as deepeval_mock
from agent_eval_suite.metrics.ragas_simple import compute_metrics_mock as ragas_mock
from agent_eval_suite.metrics.trajectory import compute_trajectory_metrics


# ---------------------------------------------------------------------------
# Mock agent: deterministic, no API key needed
# ---------------------------------------------------------------------------

class MockRAGAgent:
    """Simulates a RAG agent by echoing the ground-truth span from reference_contexts."""

    def __init__(self, dataset: list[EvalCase]):
        self._index: dict[str, EvalCase] = {c.question: c for c in dataset}

    def __call__(self, query: str) -> AgentResult:
        case = self._index.get(query)
        trajectory = [
            StepRecord(step_type="tool_call", name="retrieve", args={"query": query}),
            StepRecord(step_type="llm", output="generating answer"),
            StepRecord(step_type="final", output="done"),
        ]
        if case and case.reference_contexts:
            ctx = case.reference_contexts[0]
            answer = case.ground_truth or ctx[:120]
            trajectory[0].output = ctx
        else:
            answer = f"I don't know the answer to: {query}"

        return AgentResult(
            answer=answer,
            retrieved_contexts=case.reference_contexts or [] if case else [],
            trajectory=trajectory,
            token_usage={"prompt_tokens": 120, "completion_tokens": 40},
        )


# ---------------------------------------------------------------------------
# Real agent: gpt-4o-mini via LangChain
# ---------------------------------------------------------------------------

def _build_real_agent(dataset: list[EvalCase]):
    """Build a LangChain-based RAG agent. Requires openai + langchain packages."""
    try:
        from langchain.schema import HumanMessage
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "Run: pip install langchain langchain-openai"
        ) from exc

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    index: dict[str, EvalCase] = {c.question: c for c in dataset}

    def agent(query: str) -> AgentResult:
        case = index.get(query)
        contexts = case.reference_contexts or [] if case else []
        context_text = "\n\n".join(contexts) or "No context available."

        trajectory = [
            StepRecord(step_type="tool_call", name="retrieve", args={"query": query}, output=context_text),
        ]

        prompt = (
            f"Answer the question using only the provided context.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()

        trajectory.append(StepRecord(step_type="final", output=answer))
        usage = getattr(response, "usage_metadata", None)
        return AgentResult(
            answer=answer,
            retrieved_contexts=contexts,
            trajectory=trajectory,
            token_usage=dict(usage) if usage else None,
        )

    return agent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reference_agent", description=__doc__)
    parser.add_argument("--mock", action="store_true", default=os.environ.get("AGENT_EVAL_MOCK") == "1")
    parser.add_argument("--dataset", default="datasets/wiki_50.json")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N cases")
    parser.add_argument("--out", default="artifacts", help="Output directory for scorecard")
    args = parser.parse_args(argv)

    dataset = load_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]

    print(f"Dataset: {args.dataset} ({len(dataset)} cases)")

    if args.mock:
        print("Mode: mock (deterministic, no API key)")
        agent = MockRAGAgent(dataset)
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not set. Use --mock for a keyless run.", file=sys.stderr)
            return 1
        print("Mode: real (gpt-4o-mini via LangChain)")
        agent = _build_real_agent(dataset)

    config = EvalConfig(
        model="gpt-4o-mini",
        thresholds={"answer_relevancy": 0.5, "answer_correctness": 0.6, "hallucination": 0.3},
    )

    print("Running eval harness…")
    scorecard = EvalRunner(agent, config).run(dataset)
    scorecard = ragas_mock(scorecard, dataset, config)
    scorecard = deepeval_mock(scorecard, dataset, config)
    scorecard = compute_trajectory_metrics(scorecard, dataset, config)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(str(out / "scorecard.csv"))
    scorecard.to_json(str(out / "scorecard.json"))

    print("\n=== Aggregate scores ===")
    for metric, value in scorecard.aggregate.items():
        print(f"  {metric}: {value:.3f}")

    print(f"\nScorecard written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
