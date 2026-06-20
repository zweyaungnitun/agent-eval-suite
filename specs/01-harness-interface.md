# Phase 1 — Define the Harness Interface

## Goal
Build a thin adapter so `agent-eval-suite` can evaluate any LangChain / LangGraph agent without modifying it. The caller passes a callable, a dataset, and a config; the harness returns a scorecard. This reusability is what makes it a "suite" rather than a one-off script.

## Scope
- `harness/` package: adapter, test runner, scorecard data model.
- No metric logic yet — only the contract that later phases plug into.

## Tech
- Python 3.10+
- LangChain `Runnable` interface

## Interface contract

### Agent under test
The agent is any object satisfying one of:
- A LangChain `Runnable` (`.invoke(input)` / `.ainvoke(input)`).
- A plain callable `fn(input: dict) -> dict`.
- A LangGraph compiled graph (exposes `.invoke` and a streamable state).

The adapter normalizes all three to a common `AgentCallable` protocol:
```python
class AgentCallable(Protocol):
    def run(self, query: str) -> AgentResult: ...
```

### AgentResult
```python
@dataclass
class AgentResult:
    answer: str
    retrieved_contexts: list[str]      # for RAG metrics (may be empty)
    trajectory: list[StepRecord]        # tool calls, intermediate steps
    raw_trace: dict | None              # adapter-specific passthrough
    latency_s: float
    token_usage: dict | None            # {prompt, completion, total}
```

```python
@dataclass
class StepRecord:
    step_type: str          # "tool_call" | "llm" | "retrieval" | "final"
    name: str | None        # tool name, if applicable
    args: dict | None
    output: str | None
```

### Dataset
A list of `EvalCase`:
```python
@dataclass
class EvalCase:
    id: str
    question: str
    ground_truth: str | None
    reference_contexts: list[str] | None
    expected_tools: list[str] | None   # for trajectory metrics
```
Loadable from JSON (`datasets/*.json`).

### Runner
```python
class EvalRunner:
    def __init__(self, agent: AgentCallable, config: EvalConfig): ...
    def run(self, dataset: list[EvalCase]) -> Scorecard: ...
```
- Runs each case, collects `AgentResult`.
- Catches per-case exceptions → records as a failed case, never aborts the whole run.
- Supports concurrency (configurable, default sequential for determinism).

### Scorecard
```python
@dataclass
class Scorecard:
    per_case: pd.DataFrame     # one row per case, columns filled by metric layers
    aggregate: dict[str, float]
    metadata: dict             # run id, timestamp, config snapshot
    def to_csv(self, path): ...
    def to_json(self, path): ...
```

### Config
Single source of truth (`EvalConfig`, loaded from YAML/JSON):
```yaml
model: gpt-4o-mini
concurrency: 1
dataset_path: datasets/wiki_50.json
metrics: [faithfulness, answer_relevancy, tool_call_accuracy]
thresholds:
  faithfulness: 0.8
```

## Deliverables
- `harness/__init__.py`, `harness/adapter.py`, `harness/runner.py`, `harness/models.py`, `harness/config.py`
- Adapters for: plain callable, LangChain Runnable, LangGraph graph.
- Unit tests with a stub agent (no LLM calls).

## Definition of done
- A stub agent + 3-case in-memory dataset runs end-to-end and produces an empty-but-well-formed `Scorecard`.
- All three adapter types normalize to `AgentResult` (tested).
- Per-case exceptions are isolated and recorded.
