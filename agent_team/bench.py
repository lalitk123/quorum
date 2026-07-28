"""Council quality benchmark.

You cannot tune a review gate you do not measure. This runs the council against
a labeled corpus of diffs — seeded defects that *should* be rejected and
known-good diffs that *should* be approved — and reports the confusion matrix
(true/false approve/reject), per-seat detection rates, and cost per review.

The two numbers that matter: **defect recall** (fraction of bad diffs the gate
caught — a false-approve is a defect shipped) and **false-reject rate**
(fraction of good diffs the gate blocked — pure noise / alert fatigue). Use the
per-seat breakdown to weight or re-route seats, and the whole report as a
public, reproducible trust artifact.

Gold case layout (one directory per case under the gold root):

    <case_id>/
      case.yaml       # id, title, description, label: approve|reject, note
      diff.patch      # the diff under review
      workspace/      # optional post-change tree the file-tool reviewers browse
"""

import asyncio
import itertools
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import council, judgments, reliability
from .checks import tamper_scan
from .models import Verdict
from .runner import Budget

log = logging.getLogger("bench")


@dataclass
class GoldCase:
    id: str
    title: str
    description: str
    expected_approve: bool
    diff: str
    note: str = ""
    workspace_src: Path | None = None


@dataclass
class CaseResult:
    case_id: str
    expected_approve: bool
    decision_approved: bool
    verdicts: list[Verdict] = field(default_factory=list)
    cost_usd: float = 0.0
    error: str = ""


def load_gold(gold_dir: str) -> list[GoldCase]:
    root = Path(gold_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"gold directory not found: {gold_dir}")
    cases: list[GoldCase] = []
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        meta_path = case_dir / "case.yaml"
        diff_path = case_dir / "diff.patch"
        if not meta_path.exists() or not diff_path.exists():
            log.warning("skipping %s: missing case.yaml or diff.patch", case_dir.name)
            continue
        meta = yaml.safe_load(meta_path.read_text()) or {}
        label = str(meta.get("label", "")).lower()
        if label not in ("approve", "reject"):
            log.warning("skipping %s: label must be approve|reject", case_dir.name)
            continue
        ws = case_dir / "workspace"
        cases.append(GoldCase(
            id=str(meta.get("id", case_dir.name)),
            title=str(meta.get("title", case_dir.name)),
            description=str(meta.get("description", "")),
            expected_approve=(label == "approve"),
            diff=diff_path.read_text(),
            note=str(meta.get("note", "")),
            workspace_src=ws if ws.is_dir() else None,
        ))
    return cases


def _bench_material(case: GoldCase) -> str:
    """Mirror the worker's diff-review material (minus the approved plan)."""
    findings = tamper_scan(case.diff)
    tamper = ("\n".join(f"- {f}" for f in findings)
              + "\nTreat any finding the diff does not convincingly justify "
                "as grounds for rejection." if findings else "No findings.")
    return (
        "Review this DIFF as you would before it is pushed to GitHub. The "
        "working directory contains the changed codebase.\n\n"
        f"## Ticket {case.id}: {case.title}\n\n{case.description}\n\n"
        f"## Automated tamper scan\n{tamper}\n\n"
        f"## Diff\n```diff\n{case.diff}\n```"
    )


async def _run_case(case: GoldCase, cfg: dict, budget_cap: float | None) -> CaseResult:
    tmp = tempfile.mkdtemp(prefix=f"bench-{case.id}-")
    try:
        if case.workspace_src:
            shutil.copytree(case.workspace_src, tmp, dirs_exist_ok=True)
        budget = Budget(budget_cap, f"bench {case.id}")
        try:
            decision = await council.convene(
                material=_bench_material(case),
                cwd=tmp,
                model=cfg["council_model"],
                rule=cfg["push_approval"],
                models=cfg.get("council_models") or {},
                budgets=(budget,),
            )
        except Exception as exc:  # keep the run alive; report the failure
            log.exception("[%s] case errored", case.id)
            return CaseResult(case.id, case.expected_approve, False,
                              cost_usd=budget.spent, error=str(exc))
        return CaseResult(case.id, case.expected_approve, decision.approved,
                          verdicts=decision.verdicts, cost_usd=budget.spent)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Cohen's κ for two binary raters over the same items.

    κ corrects raw agreement for the agreement expected by chance, so it does not
    flatter a gate that just says "approve" on a mostly-good corpus (the way plain
    accuracy would). Returns None when it is undefined: no items, or one rater is
    constant so chance agreement is already total.
    """
    n = len(pairs)
    if n == 0:
        return None
    agree = sum(a == b for a, b in pairs)
    p_o = agree / n
    pa = sum(a for a, _ in pairs) / n          # rater A's rate of "True"
    pb = sum(b for _, b in pairs) / n          # rater B's rate of "True"
    p_e = pa * pb + (1 - pa) * (1 - pb)         # chance agreement
    if p_e >= 1.0:
        return None
    return (p_o - p_e) / (1 - p_e)


def _inter_seat_agreement(results: list["CaseResult"]) -> dict:
    """How correlated the seats are — the health check on the heterogeneous jury.

    The whole design bets that routing seats to different model families
    decorrelates their errors. If seats agree far more than chance (high κ), the
    jury is redundant and the extra seats buy little; genuinely diverse seats
    should show *low* pairwise κ. Reported as mean pairwise Cohen κ over the cases
    where both seats voted.
    """
    per_case: list[dict[str, bool]] = [
        {v.reviewer: v.approve for v in r.verdicts} for r in results if r.verdicts
    ]
    seats = sorted({s for case in per_case for s in case})
    pair_kappas: dict[str, float] = {}
    for a, b in itertools.combinations(seats, 2):
        pairs = [(case[a], case[b]) for case in per_case if a in case and b in case]
        k = cohen_kappa(pairs)
        if k is not None:
            pair_kappas[f"{a}~{b}"] = k
    mean = sum(pair_kappas.values()) / len(pair_kappas) if pair_kappas else None
    return {"mean_pairwise_kappa": mean, "pairwise_kappa": pair_kappas}


def compute_metrics(results: list[CaseResult]) -> dict:
    """Confusion matrix for the gate plus per-seat detection rates.

    Positive class = "should reject" (a defect is present); catching it is a
    true reject, missing it (approving a bad diff) is a false approve.
    """
    scored = [r for r in results if not r.error]
    n_good = sum(r.expected_approve for r in scored)
    n_bad = sum(not r.expected_approve for r in scored)

    true_approve = sum(r.expected_approve and r.decision_approved for r in scored)
    false_reject = sum(r.expected_approve and not r.decision_approved for r in scored)
    true_reject = sum(not r.expected_approve and not r.decision_approved for r in scored)
    false_approve = sum(not r.expected_approve and r.decision_approved for r in scored)

    seats: dict[str, dict] = {}
    for r in scored:
        for v in r.verdicts:
            s = seats.setdefault(v.reviewer, {
                "true_reject": 0, "false_approve": 0,
                "false_reject": 0, "true_approve": 0})
            if r.expected_approve:
                s["true_approve" if v.approve else "false_reject"] += 1
            else:
                s["false_approve" if v.approve else "true_reject"] += 1
    for s in seats.values():
        bad = s["true_reject"] + s["false_approve"]
        good = s["false_reject"] + s["true_approve"]
        s["defect_recall"] = s["true_reject"] / bad if bad else None
        s["false_reject_rate"] = s["false_reject"] / good if good else None

    # κ vs the human gold label — per seat and for the whole gate. Positive-class
    # neutral: we agree/disagree on the raw approve|reject call vs the label.
    for name, s in seats.items():
        seat_pairs = [(r.expected_approve, v.approve)
                      for r in scored for v in r.verdicts if v.reviewer == name]
        s["kappa_vs_human"] = cohen_kappa(seat_pairs)
    gate_kappa = cohen_kappa([(r.expected_approve, r.decision_approved) for r in scored])

    costs = [r.cost_usd for r in results]
    return {
        "n": len(results), "n_scored": len(scored),
        "n_errored": len(results) - len(scored),
        "n_good": n_good, "n_bad": n_bad,
        "gate": {
            "true_approve": true_approve, "false_reject": false_reject,
            "true_reject": true_reject, "false_approve": false_approve,
            "defect_recall": true_reject / n_bad if n_bad else None,
            "false_reject_rate": false_reject / n_good if n_good else None,
            "kappa_vs_human": gate_kappa,
        },
        "seats": seats,
        "inter_seat": _inter_seat_agreement(scored),
        "cost_usd_total": sum(costs),
        "cost_usd_mean": sum(costs) / len(costs) if costs else 0.0,
    }


def format_report(metrics: dict, results: list[CaseResult]) -> str:
    g = metrics["gate"]
    lines = ["", "=== Council benchmark ==="]
    for r in results:
        exp = "approve" if r.expected_approve else "reject"
        got = "approve" if r.decision_approved else "reject"
        mark = "err " if r.error else ("ok  " if exp == got else "MISS")
        lines.append(f"  [{mark}] {r.case_id:<24} expected={exp:<7} got={got:<7}"
                     f" ${r.cost_usd:.3f}" + (f"  ({r.error[:60]})" if r.error else ""))

    def pct(x):
        return "n/a" if x is None else f"{x * 100:.0f}%"

    def kap(x):
        return "n/a" if x is None else f"{x:+.2f}"

    lines += [
        "",
        f"cases: {metrics['n_scored']} scored"
        f" ({metrics['n_good']} good, {metrics['n_bad']} bad)"
        + (f", {metrics['n_errored']} errored" if metrics["n_errored"] else ""),
        f"gate: defect_recall={pct(g['defect_recall'])}"
        f"  false_reject_rate={pct(g['false_reject_rate'])}"
        f"  kappa_vs_human={kap(g['kappa_vs_human'])}",
        f"  true_approve={g['true_approve']} false_reject={g['false_reject']}"
        f"  true_reject={g['true_reject']} false_approve={g['false_approve']}"
        f"  <- false_approve is a defect shipped",
        f"cost: ${metrics['cost_usd_total']:.2f} total,"
        f" ${metrics['cost_usd_mean']:.3f}/case",
        "",
        "per-seat (defect_recall / false_reject_rate / kappa_vs_human):",
    ]
    for name, s in sorted(metrics["seats"].items()):
        lines.append(f"  {name:<12} {pct(s['defect_recall'])} / "
                     f"{pct(s['false_reject_rate'])} / {kap(s.get('kappa_vs_human'))}")

    inter = metrics.get("inter_seat") or {}
    mean_k = inter.get("mean_pairwise_kappa")
    lines += [
        "",
        f"jury correlation: mean pairwise κ={kap(mean_k)}"
        "  (lower = more decorrelated seats = healthier heterogeneous jury)",
    ]
    for pair, k in sorted((inter.get("pairwise_kappa") or {}).items(),
                          key=lambda kv: kv[1], reverse=True):
        lines.append(f"  {pair:<24} {kap(k)}")
    return "\n".join(lines)


async def run_bench(cfg: dict, gold_dir: str) -> dict:
    cases = load_gold(gold_dir)
    if not cases:
        raise ValueError(f"no valid gold cases found in {gold_dir}")
    log.info("benchmarking council on %d gold case(s)", len(cases))

    budget_cap = (cfg.get("budget") or {}).get("per_ticket_usd")
    sem = asyncio.Semaphore(int(cfg.get("max_concurrent_workers", 3)))

    async def bounded(case: GoldCase) -> CaseResult:
        async with sem:
            log.info("[%s] reviewing (%s)", case.id,
                     "good" if case.expected_approve else "bad")
            return await _run_case(case, cfg, budget_cap)

    results = list(await asyncio.gather(*(bounded(c) for c in cases)))
    metrics = compute_metrics(results)
    print(format_report(metrics, results))

    # Grow the reviewer-judgment dataset and estimate reliability from the whole
    # accumulated log. Gold cases are ground-truth-labeled, so every seat verdict
    # is a labeled training row for the reliability model (RQ1).
    rel_log = (cfg.get("reliability") or {}).get("log_file")
    if rel_log:
        judgments.append_judgments(rel_log, judgments_from_results(results, "gold"))
        model = reliability.estimate_reliability(judgments.load_judgments(rel_log))
        print(reliability.format_reliability(model))
        print(_bayesian_vs_gate(results, model, metrics))
    return metrics


def judgments_from_results(results: list[CaseResult], source: str) -> list[dict]:
    """One labeled judgment per seat verdict on a ground-truth-labeled case."""
    rows: list[dict] = []
    for r in results:
        if r.error:
            continue
        label = "clean" if r.expected_approve else "defect"
        for v in r.verdicts:
            rows.append({
                "seat": v.reviewer,
                "verdict": "approve" if v.approve else "reject",
                "label": label, "source": source, "case_id": r.case_id,
            })
    return rows


def _bayesian_vs_gate(results: list[CaseResult], model: dict, metrics: dict) -> str:
    """Show the calibrated posterior next to the deterministic gate decision.

    In-sample (the model was estimated from a log that includes this run), so this
    demonstrates the aggregation, not a held-out result. Reject if the posterior
    P(defect) crosses 0.5; compare against what the actual gate did.

    The prior is deliberately a neutral 0.5, NOT the gold set's own defect ratio.
    A gold corpus over-samples defects by construction (this one is ~65% bad),
    while a real review stream is overwhelmingly clean — feeding that ratio in as
    a prior would bias every posterior toward "defect" and manufacture false
    rejects that say more about how the benchmark was assembled than about the
    council. The prior is printed so the numbers below are reproducible.
    """
    base_rate = 0.5
    lines = ["", f"Bayesian trust vs gate (in-sample demo; prior P(defect)="
                 f"{base_rate:g}; reject if P(defect) >= 0.5):"]
    agree = 0
    for r in results:
        if r.error:
            continue
        votes = [(v.reviewer, not v.approve) for v in r.verdicts]
        p_def = reliability.posterior_defect_prob(votes, model, base_rate=base_rate)
        bayes_reject = p_def >= 0.5
        gate_reject = not r.decision_approved
        agree += bayes_reject == gate_reject
        lines.append(f"  {r.case_id:<24} trust={1 - p_def:4.0%}  "
                     f"bayes={'reject' if bayes_reject else 'approve':<7} "
                     f"gate={'reject' if gate_reject else 'approve'}")
    lines.append(f"  agreement with gate: {agree}/{metrics['n_scored']}")
    return "\n".join(lines)
