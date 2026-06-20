"""CLI entrypoint. Sprint 0 = skeleton; Phase 7 fills in metrics + the gate."""

from __future__ import annotations

import argparse

from .harness import EvalConfig, EvalRunner, load_dataset


def _demo_agent(query: str) -> str:
    return f"(stub answer to: {query})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-eval-suite")
    parser.add_argument("--config", help="path to eval config YAML")
    parser.add_argument("--out", default="artifacts", help="output dir")
    args = parser.parse_args(argv)

    config = EvalConfig.from_yaml(args.config) if args.config else EvalConfig()
    if not config.dataset_path:
        print("No dataset configured — skeleton run with stub agent.")
        return 0

    dataset = load_dataset(config.dataset_path)
    scorecard = EvalRunner(_demo_agent, config).run(dataset)
    print(scorecard.per_case.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
