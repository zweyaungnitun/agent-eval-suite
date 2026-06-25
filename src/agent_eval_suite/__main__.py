"""CLI entrypoint: run the eval harness with all metric layers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .harness import EvalConfig, EvalRunner, load_dataset
from .metrics.deepeval_layer import compute_metrics_mock as deepeval_mock
from .metrics.ragas_simple import compute_metrics_mock as ragas_mock
from .metrics.trajectory import compute_trajectory_metrics


def _demo_agent(query: str) -> str:
    return f"(stub answer to: {query})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-eval-suite",
        description="Run the evaluation harness against an agent and dataset.",
    )
    parser.add_argument("--config", help="Path to eval config YAML")
    parser.add_argument("--out", default="artifacts", help="Output directory for scorecard files")
    parser.add_argument("--no-gate", action="store_true", help="Skip threshold gate (always exit 0)")
    args = parser.parse_args(argv)

    config = EvalConfig.from_yaml(args.config) if args.config else EvalConfig()

    if not config.dataset_path:
        print("No dataset configured — stub run (set dataset_path in config).")
        return 0

    dataset = load_dataset(config.dataset_path)
    print(f"Loaded {len(dataset)} cases from {config.dataset_path}")

    scorecard = EvalRunner(_demo_agent, config).run(dataset)
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

    if args.no_gate:
        return 0

    failures = [
        f"{metric}={score:.3f} < {threshold}"
        for metric, threshold in config.thresholds.items()
        if (score := scorecard.aggregate.get(metric)) is not None and score < threshold
    ]
    if failures:
        print("\nTHRESHOLD FAILURES:", failures, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
