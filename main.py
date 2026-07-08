import argparse
import asyncio
import logging
import sys

import yaml

from agent_team.decomposer import DecompositionRejected
from agent_team.orchestrator import run
from agent_team.runner import BudgetExceeded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the multi-agent engineering team over a backlog."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    try:
        asyncio.run(run(cfg))
    except (DecompositionRejected, BudgetExceeded) as exc:
        print(f"\nRun aborted before any ticket started: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
