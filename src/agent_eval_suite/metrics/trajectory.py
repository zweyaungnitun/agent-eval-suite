"""Trajectory & agent-specific metrics (pure Python, no LLM).

Evaluates the *how*: tool call accuracy, ordering, step count, redundancy.
Requires AgentResult.trajectory (list[StepRecord]) and EvalCase.expected_tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from agent_eval_suite.harness.config import EvalConfig
    from agent_eval_suite.harness.models import AgentResult, EvalCase, Scorecard, StepRecord

_COLS = ("tool_call_accuracy", "tool_call_order_correct", "step_count", "redundant_steps")
_ERROR_SENTINEL = {
    "tool_call_accuracy": float("nan"),
    "tool_call_order_correct": float("nan"),
    "step_count": -1,
    "redundant_steps": -1,
}


def _order_correct(trajectory: list["StepRecord"], expected_tools: list[str]) -> bool:
    """Return True if expected_tools appear in trajectory in the same relative order."""
    called_names = [s.name for s in trajectory if s.step_type == "tool_call" and s.name is not None]
    it = iter(called_names)
    return all(tool in it for tool in expected_tools)


def _case_metrics(result: "AgentResult", expected_tools: list[str] | None) -> dict:
    trajectory = result.trajectory or []
    step_count = len(trajectory)

    if expected_tools:
        called = {s.name for s in trajectory if s.step_type == "tool_call"}
        expected_set = set(expected_tools)
        tool_call_accuracy = len(called & expected_set) / len(expected_set)
        tool_call_order_correct = _order_correct(trajectory, expected_tools)
        redundant_steps = max(0, step_count - len(expected_tools))
    else:
        tool_call_accuracy = float("nan")
        tool_call_order_correct = float("nan")
        redundant_steps = max(0, step_count)

    return {
        "tool_call_accuracy": tool_call_accuracy,
        "tool_call_order_correct": tool_call_order_correct,
        "step_count": step_count,
        "redundant_steps": redundant_steps,
    }


def compute_trajectory_metrics(
    scorecard: "Scorecard",
    cases: list["EvalCase"],
    config: "EvalConfig",
) -> "Scorecard":
    """Add trajectory columns to scorecard. No LLM or network calls."""
    results = scorecard.metadata.get("_results", {})

    rows: list[dict] = []
    for case in cases:
        result = results.get(case.id)
        if result is None or result.error:
            rows.append(dict(_ERROR_SENTINEL))
        else:
            rows.append(_case_metrics(result, case.expected_tools))

    df = pd.DataFrame(rows)
    for col in _COLS:
        scorecard.per_case[col] = df[col].values

    return scorecard
