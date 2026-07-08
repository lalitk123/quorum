import argparse
import asyncio
import logging
import sys

import yaml

from agent_team.bench import run_bench
from agent_team.canary import run_canary_sweep
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
    parser.add_argument(
        "--canary", nargs="?", const="bench/gold", default=None, metavar="GOLD_DIR",
        help="Run every bad gold case against the LIVE council once and alert on "
             "any that are approved or under-dissented (verify the verifier).",
    )
    parser.add_argument(
        "--verify-receipts", nargs="?", const="", default=None, metavar="RECEIPTS_FILE",
        help="Verify the signed audit-receipt log (signatures + hash chain) and "
             "exit. Defaults to audit.receipts_file from the config.",
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

    if args.canary is not None:
        asyncio.run(run_canary_sweep(cfg, args.canary))
        return

    if args.verify_receipts is not None:
        from agent_team import receipts
        audit = cfg.get("audit") or {}
        path = args.verify_receipts or audit.get("receipts_file", ".agent_team/receipts.jsonl")
        results = receipts.verify_log(path)
        print(receipts.format_verification(results))
        if any(not (r["sig_ok"] and r["chain_ok"]) for r in results):
            sys.exit(1)
        return

    try:
        asyncio.run(run(cfg))
    except (DecompositionRejected, BudgetExceeded) as exc:
        print(f"\nRun aborted before any ticket started: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
