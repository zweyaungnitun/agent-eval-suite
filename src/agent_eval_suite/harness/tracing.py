"""Observability hook: backend-agnostic Tracer protocol + implementations.

Default: NoOpTracer — zero overhead, no packages needed.
Optional: LangSmithTracer (requires langsmith + LANGCHAIN_API_KEY)
Optional: PhoenixTracer  (requires arize-phoenix + PHOENIX_COLLECTOR_ENDPOINT)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .config import EvalConfig
    from .models import AgentResult, EvalCase, Scorecard


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Tracer(Protocol):
    def on_run_start(self, config: "EvalConfig", n_cases: int) -> None: ...
    def on_case_start(self, case: "EvalCase") -> None: ...
    def on_case_end(self, case: "EvalCase", result: "AgentResult") -> None: ...
    def on_run_end(self, scorecard: "Scorecard") -> None: ...


# ---------------------------------------------------------------------------
# No-op (default)
# ---------------------------------------------------------------------------

class NoOpTracer:
    def on_run_start(self, config: "EvalConfig", n_cases: int) -> None:
        pass

    def on_case_start(self, case: "EvalCase") -> None:
        pass

    def on_case_end(self, case: "EvalCase", result: "AgentResult") -> None:
        pass

    def on_run_end(self, scorecard: "Scorecard") -> None:
        pass


# ---------------------------------------------------------------------------
# LangSmith
# ---------------------------------------------------------------------------

class LangSmithTracer:
    """Emit one RunTree entry per case with child runs for each trajectory step."""

    def __init__(self, project: str = "agent-eval-suite", tags: list[str] | None = None):
        try:
            from langsmith import Client
        except ImportError as exc:
            raise ImportError(
                "langsmith is not installed. Run: pip install langsmith"
            ) from exc

        self._client = Client()
        self._project = project
        self._tags = tags or []
        self._run_id: str | None = None

    def on_run_start(self, config: "EvalConfig", n_cases: int) -> None:
        pass  # no top-level run tree needed; each case is its own entry

    def on_case_start(self, case: "EvalCase") -> None:
        pass

    def on_case_end(self, case: "EvalCase", result: "AgentResult") -> None:
        try:
            from langsmith.run_trees import RunTree

            run = RunTree(
                name=f"eval:{case.id}",
                run_type="chain",
                inputs={"question": case.question},
                project_name=self._project,
                tags=self._tags,
            )
            for step in (result.trajectory or []):
                child = run.create_child(
                    name=step.name or step.step_type,
                    run_type="tool" if step.step_type == "tool_call" else "llm",
                    inputs=step.args or {},
                )
                child.end(outputs={"output": step.output})

            run.end(
                outputs={"answer": result.answer},
                metadata={
                    "latency_s": result.latency_s,
                    "error": result.error,
                    "token_usage": result.token_usage,
                },
            )
            run.post()
        except Exception as exc:  # never let tracing kill an eval run
            warnings.warn(f"LangSmithTracer.on_case_end failed: {exc}", RuntimeWarning, stacklevel=2)

    def on_run_end(self, scorecard: "Scorecard") -> None:
        pass


# ---------------------------------------------------------------------------
# Arize Phoenix
# ---------------------------------------------------------------------------

class PhoenixTracer:
    """Emit OpenTelemetry spans to an Arize Phoenix collector."""

    def __init__(self, endpoint: str | None = None):
        import os
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:
            raise ImportError(
                "opentelemetry packages are not installed. "
                "Run: pip install opentelemetry-sdk opentelemetry-exporter-otlp"
            ) from exc

        collector_endpoint = (
            endpoint
            or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
        )
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=collector_endpoint)))
        self._tracer = provider.get_tracer("agent-eval-suite")
        self._spans: dict[str, object] = {}

    def on_run_start(self, config: "EvalConfig", n_cases: int) -> None:
        pass

    def on_case_start(self, case: "EvalCase") -> None:
        span = self._tracer.start_span(f"eval:{case.id}")
        span.set_attribute("eval.case_id", case.id)
        span.set_attribute("eval.question", case.question)
        self._spans[case.id] = span

    def on_case_end(self, case: "EvalCase", result: "AgentResult") -> None:
        span = self._spans.pop(case.id, None)
        if span is None:
            return
        try:
            span.set_attribute("eval.answer", result.answer or "")
            span.set_attribute("eval.latency_s", result.latency_s)
            if result.error:
                span.set_attribute("eval.error", result.error)
        finally:
            span.end()

    def on_run_end(self, scorecard: "Scorecard") -> None:
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_tracer(tracing_config: dict) -> Tracer:
    """Instantiate the right tracer from EvalConfig.tracing dict."""
    backend = (tracing_config or {}).get("backend", "none")

    if backend == "none":
        return NoOpTracer()

    if backend == "langsmith":
        try:
            return LangSmithTracer(
                project=tracing_config.get("project", "agent-eval-suite"),
                tags=tracing_config.get("tags", []),
            )
        except ImportError as exc:
            warnings.warn(
                f"LangSmithTracer unavailable ({exc}); falling back to NoOpTracer.",
                RuntimeWarning,
                stacklevel=2,
            )
            return NoOpTracer()

    if backend == "phoenix":
        try:
            return PhoenixTracer(endpoint=tracing_config.get("endpoint"))
        except ImportError as exc:
            warnings.warn(
                f"PhoenixTracer unavailable ({exc}); falling back to NoOpTracer.",
                RuntimeWarning,
                stacklevel=2,
            )
            return NoOpTracer()

    warnings.warn(
        f"Unknown tracing backend {backend!r}; falling back to NoOpTracer.",
        RuntimeWarning,
        stacklevel=2,
    )
    return NoOpTracer()
