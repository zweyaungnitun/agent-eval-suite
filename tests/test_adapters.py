"""Tests for adapter.py — all adapter types, coercion, and concurrency.

No LangChain or LangGraph packages required: we build minimal duck-type fakes
that match the interfaces those libraries expose.
"""

from __future__ import annotations

import time

import pytest

from agent_eval_suite import AgentResult, EvalCase, EvalRunner, StepRecord
from agent_eval_suite.harness.adapter import (
    CallableAdapter,
    LangChainAdapter,
    LangGraphAdapter,
    adapt,
)
from agent_eval_suite.harness.config import EvalConfig


# ---------------------------------------------------------------------------
# Fakes (no real LangChain/LangGraph import needed)
# ---------------------------------------------------------------------------

class FakeRunnable:
    """Minimal LangChain Runnable duck-type."""

    def __init__(self, response="chain answer", use_dict_input=True):
        self._response = response
        self._use_dict_input = use_dict_input

    def invoke(self, input_):
        if self._use_dict_input and not isinstance(input_, dict):
            raise TypeError("expected dict")
        return {"output": self._response}


class FakeAgentExecutor:
    """AgentExecutor-style runnable with intermediate_steps."""

    def invoke(self, input_):
        class FakeAction:
            tool = "search"
            tool_input = {"query": input_.get("input", "")}

        return {
            "output": "final answer",
            "intermediate_steps": [(FakeAction(), "search result")],
        }


class FakeBaseMessage:
    def __init__(self, content, type_="ai", tool_calls=None):
        self.content = content
        self.type = type_
        self.tool_calls = tool_calls or []


class FakeLangGraphGraph:
    """Minimal compiled LangGraph graph duck-type."""

    def __init__(self, events):
        self._events = events

    def invoke(self, state):
        return self._events[-1] if self._events else {}

    def stream(self, state, stream_mode=None):
        return iter(self._events)

    def get_graph(self):
        return object()  # just needs to exist


# ---------------------------------------------------------------------------
# adapt() factory
# ---------------------------------------------------------------------------

def test_adapt_plain_callable():
    adapter = adapt(lambda q: "answer")
    assert isinstance(adapter, CallableAdapter)


def test_adapt_langchain_runnable():
    adapter = adapt(FakeRunnable())
    assert isinstance(adapter, LangChainAdapter)


def test_adapt_langgraph_graph():
    graph = FakeLangGraphGraph([{"messages": [FakeBaseMessage("hi")]}])
    adapter = adapt(graph)
    assert isinstance(adapter, LangGraphAdapter)


def test_adapt_unknown_type_raises():
    with pytest.raises(TypeError, match="Don't know how to adapt"):
        adapt(42)


# ---------------------------------------------------------------------------
# CallableAdapter
# ---------------------------------------------------------------------------

def test_callable_adapter_str_output():
    adapter = CallableAdapter(lambda q: "hello")
    result = adapter.run("q")
    assert result.answer == "hello"
    assert result.error is None


def test_callable_adapter_dict_output():
    adapter = CallableAdapter(lambda q: {"answer": "hi", "retrieved_contexts": ["ctx"]})
    result = adapter.run("q")
    assert result.answer == "hi"
    assert result.retrieved_contexts == ["ctx"]


def test_callable_adapter_agent_result_passthrough():
    expected = AgentResult(answer="direct")
    adapter = CallableAdapter(lambda q: expected)
    result = adapter.run("q")
    assert result is expected


def test_callable_adapter_records_latency():
    def slow(q):
        time.sleep(0.01)
        return "done"

    adapter = CallableAdapter(slow)
    result = adapter.run("q")
    assert result.latency_s >= 0.01


def test_callable_adapter_unsupported_type_raises():
    adapter = CallableAdapter(lambda q: 12345)
    with pytest.raises(TypeError):
        adapter.run("q")


# ---------------------------------------------------------------------------
# LangChainAdapter
# ---------------------------------------------------------------------------

def test_langchain_adapter_dict_output():
    adapter = LangChainAdapter(FakeRunnable())
    result = adapter.run("what is 2+2?")
    assert result.answer == "chain answer"
    assert result.error is None


def test_langchain_adapter_string_fallback():
    class StringRunnable:
        def invoke(self, input_):
            if isinstance(input_, dict):
                raise TypeError
            return "plain string"

    adapter = LangChainAdapter(StringRunnable())
    result = adapter.run("q")
    assert result.answer == "plain string"


def test_langchain_adapter_basemessage_output():
    class MessageRunnable:
        def invoke(self, input_):
            return FakeBaseMessage("message content")

    adapter = LangChainAdapter(MessageRunnable())
    result = adapter.run("q")
    assert result.answer == "message content"


def test_langchain_adapter_extracts_intermediate_steps():
    adapter = LangChainAdapter(FakeAgentExecutor())
    result = adapter.run("find something")
    assert result.answer == "final answer"
    tool_steps = [s for s in result.trajectory if s.step_type == "tool_call"]
    assert len(tool_steps) == 1
    assert tool_steps[0].name == "search"
    assert tool_steps[0].output == "search result"


def test_langchain_adapter_error_becomes_agent_result():
    class BrokenRunnable:
        def invoke(self, input_):
            raise RuntimeError("LLM down")

    adapter = LangChainAdapter(BrokenRunnable())
    result = adapter.run("q")
    assert result.error is not None
    assert "LLM down" in result.error


def test_langchain_adapter_answer_key_variants():
    for key in ("output", "answer", "result", "text"):
        class KRunnable:
            def __init__(self, k):
                self._k = k
            def invoke(self, input_):
                return {self._k: f"value-from-{self._k}"}

        adapter = LangChainAdapter(KRunnable(key))
        result = adapter.run("q")
        assert result.answer == f"value-from-{key}"


# ---------------------------------------------------------------------------
# LangGraphAdapter
# ---------------------------------------------------------------------------

def test_langgraph_adapter_extracts_final_answer():
    events = [
        {"messages": [FakeBaseMessage("thinking...", type_="ai")]},
        {"messages": [FakeBaseMessage("final answer", type_="ai")]},
    ]
    adapter = LangGraphAdapter(FakeLangGraphGraph(events))
    result = adapter.run("q")
    assert result.answer == "final answer"


def test_langgraph_adapter_extracts_tool_calls():
    tc = {"name": "calculator", "args": {"expr": "2+2"}}
    events = [
        {"messages": [FakeBaseMessage("", type_="ai", tool_calls=[tc])]},
        {"messages": [FakeBaseMessage("4", type_="tool")]},
        {"messages": [FakeBaseMessage("The answer is 4", type_="ai")]},
    ]
    adapter = LangGraphAdapter(FakeLangGraphGraph(events))
    result = adapter.run("q")
    tool_steps = [s for s in result.trajectory if s.step_type == "tool_call"]
    assert len(tool_steps) == 1
    assert tool_steps[0].name == "calculator"


def test_langgraph_adapter_stream_error_becomes_agent_result():
    class BrokenGraph:
        def invoke(self, s): pass
        def stream(self, s, stream_mode=None): raise RuntimeError("graph error")
        def get_graph(self): return object()

    adapter = LangGraphAdapter(BrokenGraph())
    result = adapter.run("q")
    assert result.error is not None
    assert "graph error" in result.error


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrent_runner_produces_same_results_as_sequential():
    dataset = [EvalCase(id=f"c{i}", question=f"q{i}") for i in range(10)]
    agent = lambda q: AgentResult(answer=f"answer:{q}")

    seq_sc = EvalRunner(agent, EvalConfig(concurrency=1)).run(dataset)
    con_sc = EvalRunner(agent, EvalConfig(concurrency=4)).run(dataset)

    assert list(seq_sc.per_case["id"]) == list(con_sc.per_case["id"])
    assert list(seq_sc.per_case["error"]) == list(con_sc.per_case["error"])


def test_concurrent_runner_row_order_matches_dataset():
    dataset = [EvalCase(id=f"c{i}", question=f"q{i}") for i in range(20)]

    def slow_agent(q):
        time.sleep(0.005)
        return AgentResult(answer=q)

    sc = EvalRunner(slow_agent, EvalConfig(concurrency=8)).run(dataset)
    assert list(sc.per_case["id"]) == [c.id for c in dataset]


def test_concurrent_runner_isolates_errors():
    dataset = [
        EvalCase(id="ok1", question="fine"),
        EvalCase(id="bad", question="boom"),
        EvalCase(id="ok2", question="also fine"),
    ]

    def flaky(q):
        if "boom" in q:
            raise ValueError("exploded")
        return AgentResult(answer="ok")

    sc = EvalRunner(flaky, EvalConfig(concurrency=3)).run(dataset)
    assert sc.metadata["n_errors"] == 1
    assert sc.metadata["n_cases"] == 3


def test_concurrency_metadata_stored():
    sc = EvalRunner(lambda q: "a", EvalConfig(concurrency=4)).run(
        [EvalCase(id="x", question="q")]
    )
    assert sc.metadata["concurrency"] == 4
