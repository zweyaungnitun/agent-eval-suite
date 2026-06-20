"""Adapters that normalize any agent flavor to ``AgentCallable``.

Sprint 0 ships the plain-callable adapter. LangChain Runnable and LangGraph
adapters land in Sprint 1 (specs/01-harness-interface.md) but the shape is here.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .models import AgentResult


def adapt(agent: Any) -> "CallableAdapter":
    """Return an AgentCallable for any supported agent flavor.

    Supported now:
      * a plain callable ``fn(query: str) -> AgentResult | dict | str``

    Planned (Sprint 1): LangChain Runnable (.invoke), LangGraph compiled graph.
    """
    if callable(agent):
        return CallableAdapter(agent)
    raise TypeError(f"Don't know how to adapt agent of type {type(agent)!r}")


class CallableAdapter:
    """Wraps a plain callable and times it, coercing output to AgentResult."""

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
