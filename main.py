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
    parser.add_argument(
        "--reliability", nargs="?", const="", default=None, metavar="JUDGMENT_LOG",
        help="Print the estimated per-seat reviewer reliability (sensitivity/"
             "specificity with credible intervals) from the judgment log and exit. "
             "Defaults to reliability.log_file from the config.",
    )
    parser.add_argument(
        "--drift", nargs="?", const="", default=None, metavar="JUDGMENT_LOG",
        help="Run CUSUM drift detection over each seat's canary outcomes and exit "
             "non-zero if any seat is degrading. Defaults to reliability.log_file.",
    )
    parser.add_argument(
        "--mutants", default=None, metavar="FILE_OR_DIR",
        help="Generate single-site mutants under the path and score how many the "
             "council kills (mutation-based review evaluation; grows the labeled "
             "judgment corpus).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of mutants evaluated (with --mutants).",
    )
    parser.add_argument(
        "--select", nargs="?", const="", default=None, metavar="JUDGMENT_LOG",
        help="Recommend a minimum-cost council reaching the target detection from "
             "measured reliability (RQ2). Defaults to reliability.log_file.",
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

    if args.reliability is not None:
        from agent_team import judgments, reliability
        path = args.reliability or (cfg.get("reliability") or {}).get(
            "log_file", ".agent_team/judgments.jsonl")
        model = reliability.estimate_reliability(judgments.load_judgments(path))
        print(reliability.format_reliability(model))
        return

    if args.mutants is not None:
        from agent_team.mutation import run_mutation_eval
        asyncio.run(run_mutation_eval(cfg, args.mutants, limit=args.limit))
        return

    if args.select is not None:
        from agent_team import judgments, reliability, selection
        sel_cfg = cfg.get("selection") or {}
        path = args.select or (cfg.get("reliability") or {}).get(
            "log_file", ".agent_team/judgments.jsonl")
        model = reliability.estimate_reliability(judgments.load_judgments(path))
        sel = selection.select_reviewers(
            model, costs=sel_cfg.get("seat_costs") or {},
            target_detection=float(sel_cfg.get("target_detection", 0.95)),
            budget=sel_cfg.get("budget"))
        print(selection.format_selection(sel, model))
        return

    if args.drift is not None:
        from agent_team import drift, judgments
        path = args.drift or (cfg.get("reliability") or {}).get(
            "log_file", ".agent_team/judgments.jsonl")
        d = cfg.get("drift") or {}
        results = drift.detect_drift(
            judgments.load_judgments(path),
            p0=float(d.get("baseline_miss_rate", 0.05)),
            p1=float(d.get("degraded_miss_rate", 0.5)),
            threshold=float(d.get("threshold", 4.0)))
        print(drift.format_drift(results))
        if any(r.alarm for r in results.values()):
            sys.exit(1)
        return

    try:
        asyncio.run(run(cfg))
    except (DecompositionRejected, BudgetExceeded) as exc:
        print(f"\nRun aborted before any ticket started: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
