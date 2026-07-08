"""Adaptive reviewer selection (idea #1 / RQ2): the right council, not every one.

Once reviewers are *measured* (`reliability.py`), the every-ticket-uses-all-five
council is wasteful: some seats add little on some changes, and every seat costs
tokens. RQ2 asks whether we can choose reviewers to **maximize defect recall per
dollar**.

Model the team as independent detectors of a defect (the heterogeneous jury is
designed for exactly this independence). A subset's detection probability — the
chance at least one seat catches a real defect — is

    P(detect) = 1 - Π_i (1 - sensitivity_i).

We greedily add the seat with the best *marginal detection per dollar* until a
target detection is met (or a budget is exhausted). Condition the reliability
estimates on the change (risk class, language, bug type) and the selected council
adapts: a security-heavy diff pulls in the seats that have measurably caught
security defects before, and drops the ones that never do.

Deterministic; the greedy trajectory is an auditable record of why each seat was
chosen. Unit-tested, including the cost-aware case where two cheap-but-good seats
beat one expensive star.
"""

from dataclasses import dataclass, field

from .reliability import SeatReliability, estimate_reliability


@dataclass
class Selection:
    seats: list[str]
    detection: float                      # P(the team catches a defect)
    cost: float
    steps: list[dict] = field(default_factory=list)   # audit trail of the greedy adds


def for_condition(judgments, **conds) -> list[dict]:
    """Judgments whose `condition` dict matches all key=value pairs (RQ2 slicing)."""
    def ok(j):
        c = j.get("condition") or {}
        return all(str(c.get(k)) == str(v) for k, v in conds.items())
    return [j for j in judgments if ok(j)]


def team_detection(seats, model: dict[str, SeatReliability],
                   default_sens: float = 0.5, conservative: bool = False) -> float:
    p_miss = 1.0
    for s in seats:
        r = model.get(s)
        sens = default_sens if r is None else (r.sens_lo if conservative else r.sens_mean)
        p_miss *= (1.0 - min(max(sens, 0.0), 1.0))
    return 1.0 - p_miss


def select_reviewers(model: dict[str, SeatReliability], costs: dict[str, float] | None = None,
                     target_detection: float = 0.95, budget: float | None = None,
                     default_cost: float = 1.0, conservative: bool = False) -> Selection:
    """Greedy min-cost council reaching `target_detection` (or best under `budget`)."""
    costs = costs or {}

    def sens(s):
        r = model[s]
        return min(max(r.sens_lo if conservative else r.sens_mean, 0.0), 1.0)

    chosen: list[str] = []
    remaining = set(model)
    detection, cost, steps = 0.0, 0.0, []
    while detection < target_detection - 1e-12 and remaining:
        best = None
        for s in remaining:
            c = costs.get(s, default_cost) or default_cost
            if budget is not None and cost + c > budget + 1e-9:
                continue
            new_det = 1.0 - (1.0 - detection) * (1.0 - sens(s))
            marginal = new_det - detection
            score = marginal / c
            if best is None or score > best["score"]:
                best = {"seat": s, "cost": c, "new_det": new_det,
                        "marginal": marginal, "score": score}
        if best is None or best["marginal"] <= 1e-12:
            break                                     # nothing affordable/helpful left
        chosen.append(best["seat"])
        remaining.discard(best["seat"])
        detection, cost = best["new_det"], cost + best["cost"]
        steps.append({"seat": best["seat"], "detection": round(detection, 4),
                      "marginal_per_dollar": round(best["score"], 4)})
    return Selection(chosen, detection, cost, steps)


def recall_per_dollar(model: dict[str, SeatReliability],
                      costs: dict[str, float] | None = None,
                      default_cost: float = 1.0) -> list[tuple[str, float]]:
    """Seats ranked by standalone sensitivity per unit cost."""
    costs = costs or {}
    ranked = [(s, r.sens_mean / (costs.get(s, default_cost) or default_cost))
              for s, r in model.items()]
    return sorted(ranked, key=lambda kv: kv[1], reverse=True)


def format_selection(sel: Selection, model: dict[str, SeatReliability]) -> str:
    lines = ["", "=== Adaptive council selection (RQ2) ==="]
    if not sel.seats:
        return "\n".join(lines + ["  (no seats selected — empty reliability model)"])
    lines.append(f"  selected: {', '.join(sel.seats)}")
    lines.append(f"  team detection P(catch defect) = {sel.detection:.1%}  "
                 f"at cost {sel.cost:g}")
    lines.append("  greedy trail (seat: cumulative detection, marginal/$):")
    for st in sel.steps:
        lines.append(f"    + {st['seat']:<12} -> {st['detection']:.1%}  "
                     f"({st['marginal_per_dollar']:.3f}/$)")
    return "\n".join(lines)


def recommend_council(judgments, costs: dict[str, float] | None = None,
                      target_detection: float = 0.95, budget: float | None = None,
                      **conds) -> Selection:
    """End-to-end: estimate reliability (optionally condition-sliced) then select."""
    rows = for_condition(judgments, **conds) if conds else list(judgments)
    model = estimate_reliability(rows)
    return select_reviewers(model, costs, target_detection, budget)
