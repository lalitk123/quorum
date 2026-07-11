"""Testing (and correcting) the independence assumption the council rests on.

Two load-bearing formulas in this codebase assume council seats err
*independently* given the latent fact "this change is defective":

    reliability.posterior_defect_prob:  posterior odds = prior odds × Π LRᵢ
    selection.team_detection:           P(detect)      = 1 - Π (1 - sᵢ)

Neither is safe to assume. The heterogeneous jury is *designed* to decorrelate
errors, but design intent is not measurement — that is the whole thesis of this
project applied to itself. And both formulas fail in the **unsafe direction** when
seats are positively correlated:

- the Bayesian posterior counts correlated evidence as if it were fresh, so the
  trust score is *overconfident* in both directions;
- the greedy selector treats a second seat as adding independent coverage when it
  is really a near-clone, so it reaches its detection target too early and
  recommends a council that is **too small**.

The good news is that we never needed the approximation. Every seat reviews every
case, so the judgment log (`case_id`, `seat`, `verdict`, `label`) already contains
the full **joint** outcome distribution. This module reads it directly:

- `empirical_detection` — measure P(at least one seat in S catches a defect) from
  the joint, with a Beta credible interval. No independence assumed.
- `independence_gap` — the headline number: predicted-minus-empirical detection.
  Positive inflation means the independence model is optimistic and the selector
  has been under-provisioning the council.
- `pairwise_dependence` — the φ coefficient per seat pair, plus the
  decision-relevant `conditional_coverage`: when seat i misses a defect, how often
  does seat j save us? Independence predicts that equals j's marginal sensitivity;
  a positively-correlated pair does far worse.
- `correlation_discount` — a Kish design-effect factor that shrinks the evidence
  weight in the Bayesian aggregator toward an *effective* number of independent
  seats, so correlated seats stop being double-counted.

Everything is deterministic and reuses the Beta machinery in `reliability.py`.
Where a subset has never been observed together, we say so rather than silently
falling back to the product of marginals.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .reliability import beta_ppf

CAUGHT = "reject"          # on a defect case, rejecting IS catching


# ---- Joint outcomes ----------------------------------------------------------

def case_outcomes(judgments, label: str = "defect") -> dict[str, dict[str, bool]]:
    """`{case_id: {seat: caught}}` over labeled cases — the joint distribution.

    `caught` means the seat gave the correct verdict for the label: `reject` on a
    defect, `approve` on a clean case. Rows without a case_id are skipped: without
    one we cannot align seats into a joint observation, and quietly pooling them
    would fabricate independence.

    A seat re-run on the same case collapses to its **latest** verdict — one case
    contributes one joint observation however many times it was judged, so a
    repeated case does not get extra weight here. How much that latest verdict is
    worth is a separate question, answered by `stability.py`.
    """
    want = CAUGHT if label == "defect" else "approve"
    out: dict[str, dict[str, bool]] = defaultdict(dict)
    for j in judgments:
        if j.get("label") != label:
            continue
        seat, verdict, case = j.get("seat"), j.get("verdict"), j.get("case_id")
        if not seat or not case or verdict not in ("approve", "reject"):
            continue
        out[case][seat] = (verdict == want)
    return dict(out)


def _complete_cases(outcomes: dict[str, dict[str, bool]], seats) -> list[dict[str, bool]]:
    """Cases where *every* seat in `seats` judged — the only ones with a defined joint."""
    seats = list(seats)
    return [o for o in outcomes.values() if all(s in o for s in seats)]


# ---- Empirical vs predicted detection ---------------------------------------

@dataclass
class DetectionEstimate:
    seats: list[str]
    n_cases: int              # complete joint observations backing this estimate
    n_detected: int           # cases where at least one seat in `seats` caught it
    mean: float               # Beta posterior mean
    lo: float                 # credible interval
    hi: float
    observed: bool = True     # False => no joint data; caller must not trust `mean`


def empirical_detection(judgments, seats, ci: float = 0.90,
                        prior_a: float = 0.5, prior_b: float = 0.5) -> DetectionEstimate:
    """P(subset catches a defect), measured from the joint. No independence assumed.

    This is the quantity `selection.team_detection` *approximates*. Where joint
    data exists it strictly dominates the approximation, because it absorbs
    whatever correlation the seats actually have.
    """
    seats = list(seats)
    cases = _complete_cases(case_outcomes(judgments, "defect"), seats)
    n = len(cases)
    if not seats or n == 0:
        return DetectionEstimate(seats, 0, 0, 0.0, 0.0, 1.0, observed=False)
    k = sum(1 for c in cases if any(c[s] for s in seats))
    a, b = prior_a + k, prior_b + (n - k)
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    return DetectionEstimate(seats, n, k, a / (a + b),
                             beta_ppf(lo_q, a, b), beta_ppf(hi_q, a, b))


@dataclass
class GapReport:
    seats: list[str]
    predicted: float          # 1 - Π(1 - sᵢ), the independence model
    empirical: float          # measured from the joint
    empirical_lo: float
    n_cases: int
    observed: bool

    @property
    def inflation(self) -> float:
        """predicted - empirical. > 0 means independence is optimistic (unsafe)."""
        return self.predicted - self.empirical


def independence_gap(judgments, seats, ci: float = 0.90) -> GapReport:
    """The headline research number: how wrong is `1 - Π(1 - sᵢ)` on real seats?"""
    from .reliability import estimate_reliability
    from .selection import team_detection

    seats = list(seats)
    model = estimate_reliability(judgments)
    predicted = team_detection(seats, model)
    emp = empirical_detection(judgments, seats, ci=ci)
    return GapReport(seats, predicted, emp.mean, emp.lo, emp.n_cases, emp.observed)


# ---- Pairwise dependence -----------------------------------------------------

@dataclass
class PairDependence:
    a: str
    b: str
    n: int
    phi: float | None            # binary correlation of *catching*; None if degenerate
    both_miss: float             # observed P(both miss)
    both_miss_independent: float # (1-sa)(1-sb) — what independence predicts
    conditional_coverage: float | None  # P(b catches | a misses); None if a never misses

    @property
    def excess_joint_miss(self) -> float:
        """Observed minus predicted joint-miss rate. > 0 = correlated blind spots."""
        return self.both_miss - self.both_miss_independent


def _phi(n11: int, n10: int, n01: int, n00: int) -> float | None:
    denom = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    if denom == 0:                       # a seat was constant — correlation undefined
        return None
    return (n11 * n00 - n10 * n01) / math.sqrt(denom)


def pairwise_dependence(judgments, seats=None) -> list[PairDependence]:
    """φ and blind-spot overlap for every seat pair, from defect cases they both saw."""
    outcomes = case_outcomes(judgments, "defect")
    if seats is None:
        seats = sorted({s for o in outcomes.values() for s in o})
    seats = sorted(seats)

    pairs: list[PairDependence] = []
    for i, a in enumerate(seats):
        for b in seats[i + 1:]:
            cases = _complete_cases(outcomes, [a, b])
            n = len(cases)
            if n == 0:
                continue
            n11 = sum(1 for c in cases if c[a] and c[b])
            n10 = sum(1 for c in cases if c[a] and not c[b])
            n01 = sum(1 for c in cases if not c[a] and c[b])
            n00 = n - n11 - n10 - n01
            sa, sb = (n11 + n10) / n, (n11 + n01) / n
            a_misses = n01 + n00
            pairs.append(PairDependence(
                a=a, b=b, n=n, phi=_phi(n11, n10, n01, n00),
                both_miss=n00 / n,
                both_miss_independent=(1 - sa) * (1 - sb),
                conditional_coverage=(n01 / a_misses) if a_misses else None,
            ))
    return pairs


def mean_phi(judgments, seats=None, clamp_negative: bool = True) -> float:
    """Mean defined pairwise φ. Negative (complementary) φ is clamped to 0 by default:
    treating complementarity as *extra* independent evidence would inflate confidence,
    and this codebase fails closed."""
    vals = [p.phi for p in pairwise_dependence(judgments, seats) if p.phi is not None]
    if not vals:
        return 0.0
    if clamp_negative:
        vals = [max(0.0, v) for v in vals]
    return sum(vals) / len(vals)


# ---- Conditional independence (slice φ by defect class) ----------------------

def _by_condition(judgments, key: str) -> dict[str, list[dict]]:
    """Group labeled rows by the value of `condition[key]`. Rows lacking that key
    are dropped — a slice must name the class it is about."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for j in judgments:
        cond = j.get("condition") or {}
        if key in cond:
            groups[str(cond[key])].append(j)
    return dict(groups)


def pairwise_dependence_sliced(judgments, key: str,
                               seats=None) -> dict[str, list[PairDependence]]:
    """Pairwise φ computed *within* each value of `condition[key]`.

    The pooled audit measures φ over every defect at once, so two seats can look
    decorrelated overall yet share a blind spot on exactly the class that matters
    (e.g. `condition={'bug_type': 'auth'}`). Slicing recomputes the same pairwise
    dependence per class, one φ per (slice, pair).
    """
    return {val: pairwise_dependence(rows, seats)
            for val, rows in _by_condition(judgments, key).items()}


@dataclass
class ConditionalHotspot:
    condition_key: str
    condition_value: str
    a: str
    b: str
    n: int
    sliced_phi: float           # φ within this defect class
    pooled_phi: float | None    # φ over all defects (None if undefined pooled)

    @property
    def excess(self) -> float:
        """How much more correlated the pair is inside this class than overall."""
        return self.sliced_phi - (self.pooled_phi or 0.0)


def conditional_hotspots(judgments, key: str, seats=None, min_excess: float = 0.2,
                         min_cases: int = 3) -> list[ConditionalHotspot]:
    """Seat pairs whose blind spots are far more correlated in one defect class than
    pooled — the hidden dependence the marginal audit misses.

    A pair is a hotspot when its φ *within* a slice exceeds its pooled φ by at least
    `min_excess`, on at least `min_cases` complete joint observations in that slice.
    These are the pairs to distrust for that class even when the overall jury looks
    decorrelated. Sorted by excess, worst first.
    """
    pooled = {tuple(sorted((p.a, p.b))): p.phi
              for p in pairwise_dependence(judgments, seats)}
    found: list[ConditionalHotspot] = []
    for val, pairs in pairwise_dependence_sliced(judgments, key, seats).items():
        for p in pairs:
            if p.phi is None or p.n < min_cases:
                continue
            pooled_phi = pooled.get(tuple(sorted((p.a, p.b))))
            if p.phi - (pooled_phi or 0.0) >= min_excess:
                found.append(ConditionalHotspot(
                    condition_key=key, condition_value=val, a=p.a, b=p.b, n=p.n,
                    sliced_phi=p.phi, pooled_phi=pooled_phi))
    return sorted(found, key=lambda h: h.excess, reverse=True)


# ---- Correcting the Bayesian aggregator --------------------------------------

def design_effect(k: int, phi_bar: float) -> float:
    """Kish design effect `1 + φ̄(k-1)` — how much correlation inflates apparent evidence."""
    if k <= 1:
        return 1.0
    return 1.0 + max(0.0, phi_bar) * (k - 1)


def effective_seats(k: int, phi_bar: float) -> float:
    """`k / deff` — the number of *independent* seats the correlated panel is worth."""
    if k <= 0:
        return 0.0
    return k / design_effect(k, phi_bar)


def correlation_discount(judgments, seats) -> float:
    """Weight in (0, 1] to apply to the summed log-likelihood-ratio evidence.

    Equals `effective_seats / k`. With φ̄ = 0 (truly independent seats) it is 1.0 and
    `posterior_defect_prob` is unchanged. With correlated seats it shrinks the
    evidence toward the base rate, which is the honest — and, when the panel
    approves, the conservative — direction.
    """
    seats = list(seats)
    k = len(seats)
    if k <= 1:
        return 1.0
    return effective_seats(k, mean_phi(judgments, seats)) / k


# ---- Plugging the measured joint into reviewer selection ---------------------

def make_empirical_detector(judgments, conservative: bool = False, min_cases: int = 3):
    """A `seats -> detection | None` callable for `selection.select_reviewers(joint=...)`.

    Returns `None` when fewer than `min_cases` complete joint observations back the
    subset, which tells the selector to fall back to the independence model rather
    than trust a detection rate measured on one or two diffs. `conservative=True`
    uses the lower credible bound — fail-closed, so a thinly-evidenced subset must
    earn its place instead of being assumed adequate.
    """
    outcomes = case_outcomes(judgments, "defect")

    def detect(seats) -> float | None:
        seats = list(seats)
        if not seats or len(_complete_cases(outcomes, seats)) < min_cases:
            return None
        est = empirical_detection(judgments, seats)
        return est.lo if conservative else est.mean

    return detect


# ---- Reporting ---------------------------------------------------------------

def format_independence(judgments, seats=None, slice_by: str | None = None) -> str:
    judgments = list(judgments)
    outcomes = case_outcomes(judgments, "defect")
    if seats is None:
        seats = sorted({s for o in outcomes.values() for s in o})
    seats = sorted(seats)

    lines = ["", "=== Independence audit (is the jury really decorrelated?) ==="]
    if not seats:
        return "\n".join(lines + ["  (no labeled-defect judgments with a case_id yet)"])

    gap = independence_gap(judgments, seats)
    if not gap.observed:
        lines.append("  (seats never judged the same case — no joint data)")
    else:
        lines += [
            f"  full council {seats}",
            f"    independence model predicts P(detect) = {gap.predicted:.1%}",
            f"    measured from the joint          = {gap.empirical:.1%} "
            f"(90% CI lower bound {gap.empirical_lo:.1%}, n={gap.n_cases})",
            f"    inflation (predicted - measured) = {gap.inflation:+.1%}"
            + ("   <-- independence is OPTIMISTIC; the selector under-provisions"
               if gap.inflation > 0.02 else
               "   <-- seats are complementary; independence is conservative here"
               if gap.inflation < -0.02 else "   <-- independence holds well"),
        ]

    pairs = pairwise_dependence(judgments, seats)
    if pairs:
        phi_bar = mean_phi(judgments, seats)
        k = len(seats)
        lines += [
            "",
            f"  mean pairwise φ = {phi_bar:.2f}   design effect = {design_effect(k, phi_bar):.2f}",
            f"  {k} seats are worth {effective_seats(k, phi_bar):.1f} independent ones "
            f"(evidence discount {correlation_discount(judgments, seats):.2f})",
            "",
            "  pair                     φ    joint-miss (obs vs indep)   P(b catches | a misses)",
        ]
        # Most correlated blind spots first — those are the pairs that fail together.
        for p in sorted(pairs, key=lambda p: p.excess_joint_miss, reverse=True):
            phi = "  n/a" if p.phi is None else f"{p.phi:+.2f}"
            cov = " n/a" if p.conditional_coverage is None else f"{p.conditional_coverage:.0%}"
            lines.append(
                f"  {p.a + ' + ' + p.b:<22} {phi}    "
                f"{p.both_miss:.0%} vs {p.both_miss_independent:.0%}"
                f"{' (!)' if p.excess_joint_miss > 0.05 else '    '}"
                f"           {cov}")
        lines.append("  (φ>0 = seats miss the same defects; a low P(b|a misses) means b "
                     "does not cover for a)")

    if slice_by:
        hotspots = conditional_hotspots(judgments, slice_by, seats)
        lines += ["", f"  conditional independence (φ sliced by '{slice_by}'):"]
        if not hotspots:
            lines.append(f"    no pair is materially more correlated within any "
                         f"'{slice_by}' class than pooled.")
        else:
            lines.append("    pair                  class            φ(slice) vs φ(pooled)   n")
            for h in hotspots:
                pooled = "  n/a" if h.pooled_phi is None else f"{h.pooled_phi:+.2f}"
                lines.append(
                    f"    {h.a + ' + ' + h.b:<20}  {h.condition_value:<14}  "
                    f"{h.sliced_phi:+.2f} vs {pooled}   ({h.n})  (!)")
            lines.append("    these pairs share a blind spot on a specific defect class "
                         "even if the jury looks decorrelated overall.")
    return "\n".join(lines)
