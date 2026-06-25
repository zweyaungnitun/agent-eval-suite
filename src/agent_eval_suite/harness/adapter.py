"""Adapters that normalize any agent flavor to AgentCallable.

Supported:
  * Plain callable     fn(query) -> AgentResult | dict | str
  * LangChain Runnable .invoke(input) — chains, AgentExecutor, LCEL pipelines
  * LangGraph graph    .stream(state) — compiled StateGraph
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .models import AgentResult, StepRecord


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def adapt(agent: Any) -> Any:
    """Return an adapter wrapping any supported agent flavor."""
    # Already adapted
    if hasattr(agent, "run") and not callable(type(agent).__dict__.get("run", None) is None and True):
        if isinstance(agent, (CallableAdapter, LangChainAdapter, LangGraphAdapter)):
            return agent

    # LangGraph compiled graph: has both .stream() and .invoke() and a graph structure
    if _is_langgraph(agent):
        return LangGraphAdapter(agent)

    # LangChain Runnable: has .invoke() (but not a graph)
    if _is_langchain_runnable(agent):
        return LangChainAdapter(agent)

    # Plain callable
    if callable(agent):
        return CallableAdapter(agent)

    raise TypeError(f"Don't know how to adapt agent of type {type(agent)!r}")


def _is_langgraph(agent: Any) -> bool:
    return (
        hasattr(agent, "invoke")
        and hasattr(agent, "stream")
        and hasattr(agent, "get_graph")  # CompiledGraph-specific
    )


def _is_langchain_runnable(agent: Any) -> bool:
    return hasattr(agent, "invoke") and not callable(agent)


# ---------------------------------------------------------------------------
# Plain callable adapter
# ---------------------------------------------------------------------------

class CallableAdapter:
    """Wraps fn(query) -> AgentResult | dict | str."""

    def __init__(self, fn: Callable[[str], Any]):
        self._fn = fn

    def run(self, query: str) -> AgentResult:
        start = time.perf_counter()
        out = self._fn(query)
        latency = time.perf_counter() - start
        result = _coerce(out)
        if result.latency_s == 0.0:
            result.latency_s = latency
        return result


# ---------------------------------------------------------------------------
# LangChain Runnable adapter
# ---------------------------------------------------------------------------

class LangChainAdapter:
    """Wraps any LangChain Runnable (chain, AgentExecutor, LCEL pipeline)."""

    def __init__(self, runnable: Any):
        self._runnable = runnable

    def run(self, query: str) -> AgentResult:
        start = time.perf_counter()
        try:
            # AgentExecutor and most chains accept a dict; LCEL often accepts str
            try:
                out = self._runnable.invoke({"input": query})
            except (TypeError, KeyError, Exception):
                out = self._runnable.invoke(query)
        except Exception as exc:
            return AgentResult(answer="", error=f"{type(exc).__name__}: {exc}")

        latency = time.perf_counter() - start
        result = _coerce_langchain(out)
        if result.latency_s == 0.0:
            result.latency_s = latency
        return result


def _coerce_langchain(out: Any) -> AgentResult:
    """Normalize LangChain output to AgentResult."""
    if isinstance(out, AgentResult):
        return out

    # LangChain BaseMessage (from ChatModel)
    if hasattr(out, "content") and isinstance(out.content, str):
        return AgentResult(answer=out.content)

    if isinstance(out, str):
        return AgentResult(answer=out)

    if isinstance(out, dict):
        answer = (
            out.get("output")
            or out.get("answer")
            or out.get("result")
            or out.get("text")
            or ""
        )
        if hasattr(answer, "content"):  # BaseMessage in dict
            answer = answer.content

        # AgentExecutor intermediate_steps → trajectory
        trajectory: list[StepRecord] = []
        for action, tool_output in out.get("intermediate_steps", []):
            trajectory.append(
                StepRecord(
                    step_type="tool_call",
                    name=getattr(action, "tool", None),
                    args=getattr(action, "tool_input", None),
                    output=str(tool_output),
                )
            )
        if trajectory:
            trajectory.append(StepRecord(step_type="final", output=str(answer)))

        return AgentResult(
            answer=str(answer),
            retrieved_contexts=out.get("source_documents", []) or [],
            trajectory=trajectory,
        )

    raise TypeError(f"LangChain agent returned unsupported type {type(out)!r}")


# ---------------------------------------------------------------------------
# LangGraph adapter
# ---------------------------------------------------------------------------

class LangGraphAdapter:
    """Wraps a compiled LangGraph StateGraph, streaming all node events."""

    def __init__(self, graph: Any):
        self._graph = graph

    def run(self, query: str) -> AgentResult:
        start = time.perf_counter()
        try:
            events = list(self._graph.stream(
                {"messages": [("human", query)]},
                stream_mode="values",
            ))
        except Exception as exc:
            return AgentResult(answer="", error=f"{type(exc).__name__}: {exc}")

        latency = time.perf_counter() - start
        result = _coerce_langgraph(events, query)
        if result.latency_s == 0.0:
            result.latency_s = latency
        return result


def _coerce_langgraph(events: list[Any], query: str) -> AgentResult:
    """Extract answer and trajectory from a LangGraph stream."""
    trajectory: list[StepRecord] = []
    answer = ""

    for event in events:
        if not isinstance(event, dict):
            continue
        messages = event.get("messages", [])
        if not messages:
            continue
        last = messages[-1]
        # AIMessage with tool_calls → tool call step
        tool_calls = getattr(last, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                trajectory.append(
                    StepRecord(
                        step_type="tool_call",
                        name=tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None),
                        args=tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {}),
                    )
                )
        # ToolMessage → record output on the last tool step
        elif hasattr(last, "content") and getattr(last, "type", None) == "tool":
            if trajectory and trajectory[-1].step_type == "tool_call":
                trajectory[-1].output = last.content
        # Final AIMessage (no tool calls)
        elif hasattr(last, "content") and getattr(last, "type", None) in ("ai", "assistant"):
            answer = last.content

    if answer:
        trajectory.append(StepRecord(step_type="final", output=answer))

    return AgentResult(answer=answer, trajectory=trajectory)


# ---------------------------------------------------------------------------
# Generic coerce (plain callable output)
# ---------------------------------------------------------------------------

def _coerce(out: Any) -> AgentResult:
    if isinstance(out, AgentResult):
        return out
    if isinstance(out, str):
        return AgentResult(answer=out)
    if isinstance(out, dict):
        return AgentResult(
            answer=out.get("answer", ""),
            retrieved_contexts=out.get("retrieved_contexts", []) or [],
            trajectory=out.get("trajectory", []) or [],
            raw_trace=out.get("raw_trace"),
            latency_s=out.get("latency_s", 0.0),
            token_usage=out.get("token_usage"),
        )
    raise TypeError(f"Agent returned unsupported type {type(out)!r}")
