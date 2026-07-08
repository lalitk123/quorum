"""The Orchestrator: turns a goal into a backlog (goal mode), schedules tickets
respecting dependencies, prevents file-level collisions between parallel
workers, and reports the final state of every ticket."""

import asyncio
import logging
from pathlib import Path

from . import canary as canary_mod
from .backlog import load_backlog
from .decomposer import decompose
from .models import Outcome, Ticket, TicketResult
from .runner import Budget
from .worker import run_ticket

log = logging.getLogger("orchestrator")

SUCCESS_OUTCOMES = (Outcome.PR_OPENED, Outcome.APPROVED_DRY_RUN)


class FileLocks:
    """Path-level claims so two workers never edit the same files concurrently.

    A worker claims its plan's file list after plan approval; a second worker
    whose plan overlaps simply waits until the first releases.
    """

    def __init__(self):
        self._claims: dict[str, set[str]] = {}
        self._cond = asyncio.Condition()

    def _conflicts(self, paths: set[str]) -> bool:
        return any(paths & claimed for claimed in self._claims.values())

    async def acquire(self, ticket_id: str, paths: list[str]) -> None:
        wanted = set(paths)
        async with self._cond:
            while self._conflicts(wanted):
                log.info("[%s] waiting on file overlap with another worker", ticket_id)
                await self._cond.wait()
            self._claims[ticket_id] = wanted

    async def release(self, ticket_id: str) -> None:
        async with self._cond:
            self._claims.pop(ticket_id, None)
            self._cond.notify_all()


async def _load_tickets(cfg: dict, run_budget: Budget) -> list[Ticket]:
    backlog_cfg = cfg.get("backlog") or {}
    # Autonomous goal mode is the default: no backlog config needed, just goal.md.
    if backlog_cfg.get("source", "goal") == "goal":
        goal = Path(backlog_cfg.get("goal_file", "goal.md")).read_text()
        log.info("goal mode: decomposing goal into tickets")
        tickets = await decompose(cfg, goal, run_budget=run_budget)
        for t in tickets:
            deps = f" (depends on {', '.join(t.depends_on)})" if t.depends_on else ""
            log.info("  %s: %s%s", t.id, t.title, deps)
        return tickets
    return load_backlog(cfg)


async def run(cfg: dict) -> list[TicketResult]:
    budget_cfg = cfg.get("budget") or {}
    run_budget = Budget(budget_cfg.get("per_run_usd"), "run")

    tickets = await _load_tickets(cfg, run_budget)
    log.info("backlog ready: %d ticket(s)", len(tickets))

    canary = canary_mod.build_monitor(cfg)
    if canary:
        log.info("canary monitor armed: %d bad case(s), 1 every %d review(s)",
                 len(canary.cases), canary.sample_every)
    locks = FileLocks()
    semaphore = asyncio.Semaphore(int(cfg["max_concurrent_workers"]))
    tasks: dict[str, asyncio.Task] = {}
    ready = asyncio.Event()  # released once every ticket's task exists

    async def bounded(ticket: Ticket) -> TicketResult:
        await ready.wait()

        # Wait for dependencies before taking a worker slot; a failed or
        # escalated dependency blocks this ticket instead of building on air.
        dep_results = [await tasks[dep] for dep in ticket.depends_on]
        bad = [r for r in dep_results if r.outcome not in SUCCESS_OUTCOMES]
        if bad:
            names = ", ".join(f"{r.ticket.id} ({r.outcome.value})" for r in bad)
            return TicketResult(ticket, Outcome.BLOCKED,
                                detail=f"dependency did not complete: {names}")
        dep_branches = [r.branch for r in dep_results if r.branch]

        async with semaphore:
            log.info("[%s] worker started: %s", ticket.id, ticket.title)
            try:
                result = await run_ticket(ticket, cfg, locks, dep_branches,
                                          run_budget=run_budget, canary=canary)
            except Exception as exc:  # a crashed worker must not sink the run
                log.exception("[%s] worker crashed", ticket.id)
                result = TicketResult(ticket, Outcome.FAILED, detail=str(exc))
            log.info("[%s] finished: %s", ticket.id, result.outcome.value)
            return result

    for ticket in tickets:
        tasks[ticket.id] = asyncio.create_task(bounded(ticket))
    ready.set()

    results = list(await asyncio.gather(*tasks.values()))
    log.info("run spend: $%.2f%s", run_budget.spent,
             f" of ${run_budget.limit:.2f} cap" if run_budget.limit else "")
    _report(results)
    if canary and canary.results:
        print(canary_mod.format_summary(canary.results))
    return results


def _report(results: list[TicketResult]) -> None:
    print("\n=== Run summary ===")
    for r in results:
        line = f"{r.ticket.id:>8}  {r.outcome.value:<18} {r.ticket.title}"
        if r.branch:
            line += f"  [{r.branch}]"
        print(line)
        if r.detail:
            print(f"          {r.detail.splitlines()[0][:120]}")
    attention = [r for r in results if r.outcome not in SUCCESS_OUTCOMES]
    if attention:
        print(f"\n{len(attention)} ticket(s) need human attention; "
              "worktrees were kept for inspection.")
