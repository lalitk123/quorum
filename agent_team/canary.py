"""Canary diffs — verify the verifier.

The council is itself a model, and models drift: a sycophantic seat, a bad model
update, or a subtly poisoned context can quietly turn the gate into a rubber
stamp. A one-time benchmark can't catch that — the failure happens *at runtime*,
on the live configuration. So we periodically slip a known-bad diff (a seeded
defect from the gold set, `expected_approve == False`) into the live review
stream and check what the council does with it:

- **The gate approved it** — CRITICAL. A seeded defect just cleared the gate;
  the same blind spot would ship a real one. Fail loud.
- **Dissent collapsed** — WARNING. Even though the gate rejected, too few seats
  objected to a known defect: the panel is one bad seat away from waving it
  through. This is the leading indicator that catches drift before it flips a
  verdict.
- **Which seats approved the known-bad** — recorded per seat, so a consistently
  fooled seat can be re-routed or retired.

The scheduling and alert logic here are deterministic and unit-tested; the only
API cost is the periodic extra council review, charged to the run budget and
skipped (never fatal) if the budget is tight.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import bench, judgments
from .models import Verdict
from .runner import Budget, BudgetExceeded

log = logging.getLogger("canary")


@dataclass
class CanaryResult:
    case_id: str
    gate_approved: bool            # True => the gate FAILED (approved a known defect)
    rejecting_seats: list[str]
    approving_seats: list[str]
    alerts: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    error: str = ""

    @property
    def critical(self) -> bool:
        return self.gate_approved


def evaluate_canary(case_id: str, gate_approved: bool, verdicts: list[Verdict],
                    min_dissent: int) -> CanaryResult:
    """Deterministic verdict on a known-bad canary review.

    `min_dissent` is the number of seats that must reject a known defect before
    the panel is considered healthy; fewer is a dissent-collapse warning.
    """
    approving = sorted(v.reviewer for v in verdicts if v.approve)
    rejecting = sorted(v.reviewer for v in verdicts if not v.approve)
    alerts: list[str] = []
    if gate_approved:
        alerts.append(
            f"CRITICAL: the council gate APPROVED known-bad canary '{case_id}' — "
            "a seeded defect would have shipped. The live gate has a blind spot.")
    elif len(rejecting) < min_dissent:
        alerts.append(
            f"WARNING: dissent collapsed on canary '{case_id}' — only "
            f"{len(rejecting)} seat(s) rejected a known defect (healthy floor is "
            f"{min_dissent}). The panel is close to waving defects through.")
    if approving:
        alerts.append(
            f"seats fooled by '{case_id}': {', '.join(approving)}")
    return CanaryResult(case_id, gate_approved, rejecting, approving, alerts)


class CanaryMonitor:
    """Injects a known-bad canary into the live stream every N real reviews.

    One instance is shared across all workers; a lock serializes case selection
    (round-robin over the bad gold cases) so concurrent workers don't all fire on
    the same tick or race the counter.
    """

    def __init__(self, cases: list, sample_every: int, min_dissent: int,
                 budget_cap: float | None, judgment_log: str | None = None):
        self.cases = cases
        self.sample_every = sample_every
        self.min_dissent = min_dissent
        self.budget_cap = budget_cap
        self.judgment_log = judgment_log
        self.results: list[CanaryResult] = []
        self._reviews = 0
        self._idx = 0
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.cases) and self.sample_every > 0

    async def _claim_due_case(self):
        """Advance the review counter; return a case on the sampling tick."""
        async with self._lock:
            self._reviews += 1
            if self._reviews % self.sample_every != 0:
                return None
            case = self.cases[self._idx % len(self.cases)]
            self._idx += 1
            return case

    async def run_if_due(self, cfg: dict, budgets: tuple[Budget, ...] = ()) -> None:
        """Called after each real diff review. Fires a canary on the sampling tick."""
        if not self.enabled:
            return
        case = await self._claim_due_case()
        if case is None:
            return
        for b in budgets:
            try:
                b.precheck()
            except BudgetExceeded:
                log.info("canary skipped: %s budget exhausted", b.label)
                return
        log.info("injecting canary '%s' into the live review stream", case.id)
        result = await self._review(case, cfg, budgets)
        self.results.append(result)
        for alert in result.alerts:
            log.warning("[canary] %s", alert)

    async def _review(self, case, cfg: dict, budgets: tuple[Budget, ...]) -> CanaryResult:
        cap = self.budget_cap
        try:
            case_result = await bench._run_case(case, cfg, cap)
        except Exception as exc:  # a canary must never sink a real ticket
            log.warning("canary '%s' errored: %s", case.id, exc)
            return CanaryResult(case.id, False, [], [], error=str(exc))
        for b in budgets:
            b.add(case_result.cost_usd)
        if case_result.error:
            return CanaryResult(case.id, False, [], [], error=case_result.error,
                                cost_usd=case_result.cost_usd)
        self._log_judgments(case, case_result.verdicts)
        result = evaluate_canary(case.id, case_result.decision_approved,
                                 case_result.verdicts, self.min_dissent)
        result.cost_usd = case_result.cost_usd
        return result

    def _log_judgments(self, case, verdicts) -> None:
        """Persist each seat's verdict on this known-bad canary (label=defect) so
        it feeds both the reliability model (RQ1) and the drift time series (RQ3)."""
        if not self.judgment_log or not verdicts:
            return
        ts = datetime.now(timezone.utc).isoformat()
        rows = [{"seat": v.reviewer,
                 "verdict": "approve" if v.approve else "reject",
                 "label": "defect", "source": "canary",
                 "case_id": case.id, "timestamp": ts} for v in verdicts]
        try:
            judgments.append_judgments(self.judgment_log, rows)
        except OSError as exc:
            log.warning("could not persist canary judgments: %s", exc)


def build_monitor(cfg: dict) -> CanaryMonitor | None:
    """Construct a CanaryMonitor from config, or None when canaries are off."""
    canary_cfg = cfg.get("canary") or {}
    if not canary_cfg.get("enabled"):
        return None
    gold_dir = canary_cfg.get("gold_dir", "bench/gold")
    cases = [c for c in bench.load_gold(gold_dir) if not c.expected_approve]
    if not cases:
        log.warning("canary enabled but no bad gold cases in %s; disabling", gold_dir)
        return None
    budget_cap = (cfg.get("budget") or {}).get("per_ticket_usd")
    return CanaryMonitor(
        cases=cases,
        sample_every=int(canary_cfg.get("sample_every", 5)),
        min_dissent=int(canary_cfg.get("min_dissent", 2)),
        budget_cap=budget_cap,
        judgment_log=(cfg.get("reliability") or {}).get("log_file"),
    )


def format_summary(results: list[CanaryResult]) -> str:
    if not results:
        return ""
    critical = [r for r in results if r.critical]
    dissent = [r for r in results if not r.critical and r.alerts]
    lines = ["", "=== Canary monitor (verify the verifier) ==="]
    for r in results:
        status = "ERROR" if r.error else ("APPROVED-DEFECT" if r.gate_approved else "caught")
        lines.append(f"  [{status:<15}] {r.case_id:<24}"
                     f" rejected={len(r.rejecting_seats)} approved_by={r.approving_seats or '-'}"
                     + (f"  ({r.error[:50]})" if r.error else ""))
    lines.append(
        f"{len(results)} canary review(s): {len(critical)} approved a known defect, "
        f"{len(dissent)} dissent-collapse warning(s).")
    if critical:
        lines.append("  >>> ACTION: the live gate approved a seeded defect — "
                     "do not trust auto-approvals until this is understood.")
    return "\n".join(lines)


async def run_canary_sweep(cfg: dict, gold_dir: str) -> list[CanaryResult]:
    """On-demand sweep: run every bad gold case against the live council once."""
    cases = [c for c in bench.load_gold(gold_dir) if not c.expected_approve]
    if not cases:
        raise ValueError(f"no bad (label: reject) gold cases in {gold_dir}")
    min_dissent = int((cfg.get("canary") or {}).get("min_dissent", 2))
    budget_cap = (cfg.get("budget") or {}).get("per_ticket_usd")
    monitor = CanaryMonitor(cases, sample_every=1, min_dissent=min_dissent,
                            budget_cap=budget_cap,
                            judgment_log=(cfg.get("reliability") or {}).get("log_file"))
    sem = asyncio.Semaphore(int(cfg.get("max_concurrent_workers", 3)))

    async def one(case) -> CanaryResult:
        async with sem:
            log.info("canary sweep: reviewing '%s'", case.id)
            return await monitor._review(case, cfg, ())

    results = list(await asyncio.gather(*(one(c) for c in cases)))
    for r in results:
        for alert in r.alerts:
            log.warning("[canary] %s", alert)
    print(format_summary(results))
    return results
