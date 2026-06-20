# agent-eval-suite

A reusable evaluation harness for LangChain / LangGraph agents. Point it at any
agent and a test dataset; get back a scorecard covering answer quality, agent
trajectory, and observability — wired into CI so regressions fail the build.

```
Agent under test → RAGAS → DeepEval → LangSmith trace → CI gate (pass/fail)
```

> **Status: Sprint 0 (walking skeleton).** The harness spine works end-to-end
> with a stub agent. Metric layers land in later sprints — see
> [specs/](specs/) for the phase specs and [SPRINTS.md](SPRINTS.md) for the plan.

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart (current)

```python
from agent_eval_suite import EvalRunner, EvalCase

def my_agent(query: str) -> str:
    return "..."   # any callable; LangChain/LangGraph adapters in Sprint 1

dataset = [EvalCase(id="q1", question="What is 2+2?", ground_truth="4")]
scorecard = EvalRunner(my_agent).run(dataset)
print(scorecard.per_case)
```

## Test

```bash
pytest          # no LLM, no network — fast inner loop
```

## Roadmap

The 8 build phases are specified in [specs/](specs/); the agile sprint sequence
is in [SPRINTS.md](SPRINTS.md).
