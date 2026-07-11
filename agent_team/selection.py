"""Adaptive reviewer selection (idea #1 / RQ2): the right council, not every one.

Once reviewers are *measured* (`reliability.py`), the every-ticket-uses-all-five
council is wasteful: some seats add little on some changes, and every seat costs
tokens. RQ2 asks whether we can choose reviewers to **maximize defect recall per
dollar**.

A subset's detection probability is the chance at least one of its seats catches a
real defect. If seats erred independently that would be

    P(detect) = 1 - Π_i (1 - sensitivity_i).

The heterogeneous jury is *designed* for that independence, but design intent is
not measurement, and the formula fails unsafely when it is wrong: two seats that
miss the same defects look additive, so the greedy hits its target early and
returns a council that is too small. So by default we do not assume it — every
seat reviews every case, so `independence.make_empirical_detector` reads each
candidate subset's **measured** joint detection straight out of the judgment log,
and the product of marginals is used only as a fallback where joint data is thin.
`--independence` reports how far apart the two estimators actually are.

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
from .stability import _majority_reject_prob, accuracy_ceiling


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
                     default_cost: float = 1.0, conservative: bool = False,
                     joint=None) -> Selection:
    """Greedy min-cost council reaching `target_detection` (or best under `budget`).

    `joint` is an optional `seats -> detection | None` callable (see
    `independence.make_empirical_detector`). When supplied and it has data for a
    candidate subset, its **measured** joint detection replaces the independence
    product — so two seats that miss the same defects stop looking additive. It
    returns `None` for subsets with too little joint data, and we fall back to the
    independence model for those, recording which estimator decided each step.

    To let the greedy also *buy self-repeats* of a cheap flaky seat (majority-voting
    one seat `k` times to raise its effective sensitivity), see
    `select_with_repeats`, which the CLI reaches via `recommend_council(max_repeats>1)`.
    """
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
            new_det, source = None, "measured"
            if joint is not None:
                new_det = joint(chosen + [s])
            if new_det is None:
                new_det = 1.0 - (1.0 - detection) * (1.0 - sens(s))
                source = "independence"
            # True P(at least one catches) is monotone in the subset; any dip is
            # estimation noise from the shifting joint-case support.
            new_det = max(new_det, detection)
            marginal = new_det - detection
            score = marginal / c
            if best is None or score > best["score"]:
                best = {"seat": s, "cost": c, "new_det": new_det, "source": source,
                        "marginal": marginal, "score": score}
        if best is None or best["marginal"] <= 1e-12:
            break                                     # nothing affordable/helpful left
        chosen.append(best["seat"])
        remaining.discard(best["seat"])
        detection, cost = best["new_det"], cost + best["cost"]
        steps.append({"seat": best["seat"], "detection": round(detection, 4),
                      "marginal_per_dollar": round(best["score"], 4),
                      "estimator": best["source"]})
    return Selection(chosen, detection, cost, steps)


def select_with_repeats(model: dict[str, SeatReliability],
                        flip_rates: dict[str, float] | None = None,
                        costs: dict[str, float] | None = None,
                        target_detection: float = 0.95, budget: float | None = None,
                        default_cost: float = 1.0, conservative: bool = False,
                        max_repeats: int = 5) -> Selection:
    """Greedy selection that can *buy majority-voted self-repeats* of a flaky seat.

    This is the consumer of `stability.py`'s pricing: a stochastic seat cannot have
    accuracy above its consistency ceiling `p = min(sensitivity, ceiling(flip_rate))`,
    but running it `k` (odd) times and majority-voting drives its *effective*
    sensitivity toward 1 at `k×` the cost. So `k` runs of a cheap flaky seat can beat
    one run of an expensive stable one when target detection would otherwise force
    the expensive purchase.

    Each greedy step makes the best detection-per-dollar move among: (a) add a new
    seat at one run, or (b) buy two more repeats of an already-chosen seat (raising
    its effective sensitivity). Detection is the independence combine
    `1 - Π(1 - eff_i)` throughout — the measured-joint estimator is only defined for
    single-run distinct-seat subsets, so repeats deliberately use the independence
    model (and a repeat's evidence is the same conditional-independence assumption
    the fallback already makes). With `max_repeats=1` this reduces to a cost-aware
    single-run greedy under the independence model.
    """
    costs = costs or {}
    flip_rates = flip_rates or {}

    def base_p(s):
        r = model[s]
        p = min(max(r.sens_lo if conservative else r.sens_mean, 0.0), 1.0)
        if s in flip_rates:
            p = min(p, accuracy_ceiling(flip_rates[s]))   # bounded by consistency
        return p

    def eff(s, k):
        return base_p(s) if k == 1 else _majority_reject_prob(base_p(s), k)

    reps: dict[str, int] = {}                 # chosen seat -> current odd repeat count
    order: list[str] = []
    detection, cost, steps = 0.0, 0.0, []

    def detection_without(s):
        """Current detection with seat s's contribution divided out (independence)."""
        e = eff(s, reps[s])
        return 1.0 - (1.0 - detection) / (1.0 - e) if e < 1.0 else detection

    while detection < target_detection - 1e-12:
        best = None
        # (a) add a new seat at one run
        for s in set(model) - set(reps):
            c = costs.get(s, default_cost) or default_cost
            if budget is not None and cost + c > budget + 1e-9:
                continue
            new_det = max(1.0 - (1.0 - detection) * (1.0 - eff(s, 1)), detection)
            score = (new_det - detection) / c
            if best is None or score > best["score"]:
                best = {"seat": s, "reps": 1, "add_cost": c, "new_det": new_det,
                        "score": score, "kind": "add"}
        # (b) buy two more repeats of an already-chosen (flaky) seat
        for s in list(reps):
            k2 = reps[s] + 2
            if k2 > max_repeats:
                continue
            c = 2.0 * (costs.get(s, default_cost) or default_cost)
            if budget is not None and cost + c > budget + 1e-9:
                continue
            base_det = detection_without(s)
            new_det = max(1.0 - (1.0 - base_det) * (1.0 - eff(s, k2)), detection)
            score = (new_det - detection) / c
            if best is None or score > best["score"]:
                best = {"seat": s, "reps": k2, "add_cost": c, "new_det": new_det,
                        "score": score, "kind": "repeat"}
        if best is None or best["new_det"] - detection <= 1e-12:
            break                                     # nothing affordable/helpful left
        s = best["seat"]
        if best["kind"] == "add":
            order.append(s)
        reps[s] = best["reps"]
        detection, cost = best["new_det"], cost + best["add_cost"]
        steps.append({"seat": s, "repeats": reps[s], "detection": round(detection, 4),
                      "marginal_per_dollar": round(best["score"], 4),
                      "estimator": "repeat" if best["kind"] == "repeat" else "independence"})
    return Selection(order, detection, cost, steps)


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
    lines.append("  greedy trail (seat: cumulative detection, marginal/$, estimator):")
    for st in sel.steps:
        reps = st.get("repeats", 1)
        rep_tag = f" ×{reps}" if reps > 1 else ""
        lines.append(f"    + {st['seat']:<12}{rep_tag:<4} -> {st['detection']:.1%}  "
                     f"({st['marginal_per_dollar']:.3f}/$)  [{st.get('estimator', 'independence')}]")
    if any(st.get("estimator") == "independence" for st in sel.steps):
        lines.append("  note: 'independence' steps assume seats miss defects independently — "
                     "run `--independence` to see whether that holds.")
    return "\n".join(lines)


def recommend_council(judgments, costs: dict[str, float] | None = None,
                      target_detection: float = 0.95, budget: float | None = None,
                      empirical: bool = True, conservative: bool = False,
                      max_repeats: int = 1, **conds) -> Selection:
    """End-to-end: estimate reliability (optionally condition-sliced) then select.

    `empirical=True` (default) measures each candidate subset's joint detection from
    the judgment log instead of assuming seats are independent, falling back to the
    independence product only where joint data is too thin. `max_repeats > 1` lets
    the selector buy majority-voted self-repeats of a flaky-but-cheap seat, priced
    from its measured repeat-run flip rate.
    """
    rows = for_condition(judgments, **conds) if conds else list(judgments)
    model = estimate_reliability(rows)
    if max_repeats > 1:
        from .stability import seat_stability
        flip_rates = {s: st.flip_rate
                      for s, st in seat_stability(rows, label="defect").items()
                      if st.observed}
        return select_with_repeats(model, flip_rates, costs, target_detection, budget,
                                   conservative=conservative, max_repeats=max_repeats)
    joint = None
    if empirical:
        from .independence import make_empirical_detector
        joint = make_empirical_detector(rows, conservative=conservative)
    return select_reviewers(model, costs, target_detection, budget,
                            conservative=conservative, joint=joint)
