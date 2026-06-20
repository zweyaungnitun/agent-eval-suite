# agent-eval-suite — Project Overview Spec

## Purpose
`agent-eval-suite` is a reusable evaluation harness for LangChain / LangGraph agents. Point it at any agent (a callable) and a test dataset, and get back a scorecard covering answer quality, agent trajectory, and observability — wired into CI so regressions fail the build.

## Pipeline
```
Agent under test → RAGAS → DeepEval → LangSmith trace → CI gate (pass/fail)
```

## Design principles
- **Reusable, not one-off.** A thin adapter accepts any agent that conforms to the harness interface. No modification of the agent under test.
- **Layered metrics.** RAGAS for RAG quality, DeepEval for unit-test-style assertions, custom Python for agent-specific trajectory metrics.
- **Observable.** Every run emits inspectable traces (per-step latency, token cost, failure points), not just a final number.
- **CI-native.** The suite runs on every push; scorecards post to PRs; thresholds gate merges.

## Repo structure
```
agent-eval-suite/
  harness/                  # adapter + test runner
  metrics/                  # ragas.py, deepeval.py, trajectory.py
  datasets/                 # synthetic test sets (json)
  examples/                 # reference agent to demo against
  .github/workflows/eval.yml
  README.md
```

## Phase specs
| Phase | Spec | Title |
|-------|------|-------|
| 1 | [01-harness-interface.md](01-harness-interface.md) | Define the harness interface |
| 2 | [02-synthetic-testset.md](02-synthetic-testset.md) | Synthetic test set generation |
| 3 | [03-ragas-metrics.md](03-ragas-metrics.md) | RAGAS metric layer |
| 4 | [04-deepeval-layer.md](04-deepeval-layer.md) | DeepEval unit-test layer |
| 5 | [05-trajectory-metrics.md](05-trajectory-metrics.md) | Trajectory & agent-specific metrics |
| 6 | [06-observability.md](06-observability.md) | Observability hook |
| 7 | [07-ci-integration.md](07-ci-integration.md) | CI integration |
| 8 | [08-readme-benchmark.md](08-readme-benchmark.md) | README & benchmark page |

## Tech stack
Python 3.10+, LangChain / LangGraph, RAGAS, DeepEval, pytest, Pandas, LangSmith (Arize Phoenix optional), GitHub Actions.

## Global definition of done
- Each phase ships with tests and a runnable example against the reference agent in `examples/`.
- All metrics are configurable via a single config file (thresholds, model choice, dataset path).
- No secrets in the repo; API keys read from environment.
