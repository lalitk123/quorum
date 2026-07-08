import argparse
import asyncio
import logging
import sys

import yaml

from agent_team.bench import run_bench
from agent_team.decomposer import DecompositionRejected
from agent_team.orchestrator import run
from agent_team.runner import BudgetExceeded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the multi-agent engineering team over a backlog."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--bench", nargs="?", const="bench/gold", default=None, metavar="GOLD_DIR",
        help="Benchmark the council against a labeled gold set instead of "
             "running the build loop (default dir: bench/gold).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.bench is not None:
        asyncio.run(run_bench(cfg, args.bench))
        return

    try:
        asyncio.run(run(cfg))
    except (DecompositionRejected, BudgetExceeded) as exc:
        print(f"\nRun aborted before any ticket started: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
