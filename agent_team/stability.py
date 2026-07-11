"""Repeat-run stability — a reviewer cannot be more accurate than it is consistent.

Every accuracy number this system reports (`reliability.py`) is measured from one
verdict per seat per case. But an LLM reviewer is stochastic: run the same seat on
the same diff twice and it may reject once and approve once. If a seat changes its
mind that easily, a single-pass sensitivity of 95% is not a property of the seat —
it is one draw from a coin the benchmark happened to catch on a good day.

This module measures **test-retest reliability**: how often a seat contradicts
itself on a re-run, and what that flip rate *proves* about the accuracy it can
possibly have. The judgment log already carries the data — every `--bench` or
`--canary` re-run appends another row for the same `(seat, case_id)`.

The bound
---------
Let `p_i = P(seat rejects case i)` on defect case `i`. Sensitivity is the average
over defect cases, `s = E[p_i]`. Run the seat twice on the same case: it disagrees
with itself with probability `2 p_i (1 - p_i)`. Pooled over defect cases, the flip
rate is

    f = E[2 p_i (1 - p_i)] = 2(s - E[p_i²])  ≤  2 s (1 - s)

because `E[p²] ≥ (E[p])²` — Jensen's inequality. Solving the quadratic:

    |s - ½|  ≤  √(1 - 2f) / 2

**A seat's measurable distance from a coin flip is capped by its self-consistency.**
A reviewer that changes its verdict on 32% of re-runs cannot have a sensitivity
above 80%, no matter what a single-pass benchmark reports. The bound is
distribution-free: it assumes nothing about how `p_i` varies across cases, only
that repeated runs on one case are exchangeable draws.

Three consequences worth stating plainly:

- `f = 0.5` is the maximum a real seat can have in expectation (it is `2p(1-p)`
  at `p = ½`), and it forces `s = ½`. A *sample* `f̂` can still land above 0.5 —
  one case run 4 times with 2 rejects scores `4/6` — so an observed `f̂ > 0.5` is
  noise, not something exotic. The ceiling clamps to ½: a coin flip.
- Repeating a coin flip does not make it a reviewer. `runs_for_stability` returns
  `None` for `f = 0.5`: no amount of self-majority-voting rescues a seat whose
  per-case reject probability is ½, because the majority of `k` fair coins is
  still a fair coin.
- `s` and `f` must be measured on the same case distribution for the bound to
  bind. `ceiling_violations` deliberately compares across supports, and what it
  detects is precisely that mismatch — see its docstring.

Estimator
---------
For a case seen `n` times with `r` rejects, the discordant unordered pairs number
`r(n - r)` out of `C(n, 2)`. Since `E[r(n - r)] = n(n - 1) p(1 - p)`, the pooled
ratio

    f̂ = Σ r_i(n_i - r_i) / Σ C(n_i, 2)

is unbiased for the `C(n_i, 2)`-weighted mean of `2 p_i(1 - p_i)` — no plug-in
bias from estimating `p_i` first. Pairs drawn from the same case share that case's
`p_i`, so they are not independent and a Beta-Binomial interval over pairs would
be too narrow; the interval here is a **cluster bootstrap resampling whole cases**,
seeded for reproducibility.
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass

REJECT = "reject"


def accuracy_ceiling(flip_rate: float) -> float:
    """The largest accuracy consistent with a seat that flips this often.

    `½ + √(1 - 2f)/2`. Also the implied per-case reject probability `p ≥ ½`
    solving `2p(1-p) = f` — the ceiling and the consistent root are the same
    number, which is why `runs_for_stability` reuses this.

    `f > ½` is unachievable by any `p`; it clamps to ½ (an indistinguishable-
    from-chance seat) rather than raising.
    """
    f = min(max(flip_rate, 0.0), 1.0)
    return 0.5 + math.sqrt(max(0.0, 1.0 - 2.0 * f)) / 2.0


def _majority_reject_prob(p: float, k: int) -> float:
    """P(a majority of k independent draws reject), each rejecting w.p. p."""
    return sum(math.comb(k, j) * p**j * (1 - p) ** (k - j)
               for j in range((k + 1) // 2, k + 1))


def runs_for_stability(flip_rate: float, target_flip: float = 0.05,
                       max_k: int = 99) -> int | None:
    """Odd number of self-repeats whose majority vote flips at most `target_flip`.

    Converts "this seat is flaky" into a price: `k` sampled reviews instead of one.
    Returns None when no `k ≤ max_k` suffices — which is the honest answer for a
    seat at `f = ½`, where the majority of `k` fair coins is still a fair coin.
    """
    p = accuracy_ceiling(flip_rate)
    for k in range(1, max_k + 1, 2):
        pk = _majority_reject_prob(p, k)
        if 2.0 * pk * (1.0 - pk) <= target_flip + 1e-12:
            return k
    return None


# ---- Measuring the flip rate -------------------------------------------------

def repeat_groups(judgments, label: str | None = None) -> dict[str, dict[str, list[bool]]]:
    """`{seat: {case_id: [is_reject, ...]}}` — one entry per run of that seat on that case.

    Rows without a `case_id` are skipped: two verdicts can only be recognized as
    repeats of each other if they name the same case.
    """
    out: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for j in judgments:
        if label is not None and j.get("label") != label:
            continue
        seat, verdict, case = j.get("seat"), j.get("verdict"), j.get("case_id")
        if not seat or not case or verdict not in ("approve", REJECT):
            continue
        out[seat][case].append(verdict == REJECT)
    return {s: dict(cases) for s, cases in out.items()}


def _counts(cases: dict[str, list[bool]]) -> list[tuple[int, int]]:
    """`[(n_runs, n_rejects)]` for cases run more than once."""
    return [(len(v), sum(v)) for v in cases.values() if len(v) >= 2]


def _pooled_flip(counts: list[tuple[int, int]]) -> tuple[int, int]:
    discordant = sum(r * (n - r) for n, r in counts)
    pairs = sum(n * (n - 1) // 2 for n, _ in counts)
    return discordant, pairs


def _bootstrap_flip(counts, ci: float = 0.90, reps: int = 2000,
                    seed: int = 17) -> tuple[float, float]:
    """Cluster bootstrap over whole cases — pairs within a case share that case's p."""
    k = len(counts)
    if k < 2:
        return 0.0, 1.0                      # one case says nothing about spread
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        disc, pairs = _pooled_flip([counts[rng.randrange(k)] for _ in range(k)])
        if pairs:
            vals.append(disc / pairs)
    if not vals:
        return 0.0, 1.0
    vals.sort()
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    return vals[int(lo_q * (len(vals) - 1))], vals[int(hi_q * (len(vals) - 1))]


@dataclass
class SeatStability:
    seat: str
    n_cases: int                  # cases this seat was run on more than once
    n_runs: int
    n_pairs: int
    n_discordant: int
    flip_rate: float
    flip_lo: float
    flip_hi: float
    reject_rate: float            # pooled, over repeated runs
    kappa: float | None           # test-retest agreement, chance-corrected
    observed: bool = True         # False => no repeats; nothing here is meaningful

    @property
    def feasible(self) -> bool:
        """False when f̂ > ½ — noise beyond what any per-case p can produce."""
        return self.flip_rate <= 0.5

    @property
    def ceiling(self) -> float:
        """Accuracy ceiling at the point estimate."""
        return accuracy_ceiling(self.flip_rate)

    @property
    def ceiling_generous(self) -> float:
        """Ceiling at the *lowest* plausible flip rate — the seat's best case.

        A ceiling this high is hard to violate, so a violation of *this* number is
        not an artifact of a noisy flip-rate estimate.
        """
        return accuracy_ceiling(self.flip_lo)

    @property
    def repeats_for_5pct(self) -> int | None:
        return runs_for_stability(self.flip_rate, 0.05)

    @property
    def chance_agreement(self) -> float:
        """P(two runs agree by chance alone), from the seat's marginal reject rate."""
        return self.reject_rate**2 + (1.0 - self.reject_rate) ** 2

    @property
    def kappa_informative(self) -> bool:
        """False when skewed marginals leave kappa no headroom (the kappa paradox).

        A seat that rejects 95% of the time already agrees with itself 90.5% of the
        time by chance, so kappa collapses toward 0 however consistent it actually
        is — and can go *negative* for a seat flipping only 10% of re-runs. Kappa
        is reported for completeness; the flip rate and the ceiling it implies are
        the metrics that mean something here.
        """
        return self.chance_agreement <= 0.7


def _kappa(flip_rate: float, reject_rate: float) -> float | None:
    """Chance-corrected test-retest agreement between two runs of one seat."""
    p_o = 1.0 - flip_rate
    p_e = reject_rate**2 + (1.0 - reject_rate) ** 2
    if abs(1.0 - p_e) < 1e-12:          # a seat that never varies has no chance baseline
        return None
    return (p_o - p_e) / (1.0 - p_e)


def seat_stability(judgments, label: str | None = None, ci: float = 0.90,
                   reps: int = 2000, seed: int = 17) -> dict[str, SeatStability]:
    """Per-seat test-retest stability over cases judged more than once.

    `label="defect"` bounds sensitivity; `label="clean"` bounds specificity;
    `None` pools every labeled and unlabeled case the seat repeated.
    """
    out: dict[str, SeatStability] = {}
    for seat, cases in repeat_groups(judgments, label).items():
        counts = _counts(cases)
        discordant, pairs = _pooled_flip(counts)
        if pairs == 0:
            out[seat] = SeatStability(seat, 0, sum(len(v) for v in cases.values()),
                                      0, 0, 0.0, 0.0, 1.0, 0.0, None, observed=False)
            continue
        n_runs = sum(n for n, _ in counts)
        flip = discordant / pairs
        reject_rate = sum(r for _, r in counts) / n_runs
        lo, hi = _bootstrap_flip(counts, ci, reps, seed)
        out[seat] = SeatStability(
            seat=seat, n_cases=len(counts), n_runs=n_runs, n_pairs=pairs,
            n_discordant=discordant, flip_rate=flip, flip_lo=lo, flip_hi=hi,
            reject_rate=reject_rate, kappa=_kappa(flip, reject_rate))
    return out


def stability_ceilings(judgments, ci: float = 0.90,
                       generous: bool = False) -> dict[str, tuple[float, float]]:
    """Per-seat `(sensitivity_ceiling, specificity_ceiling)` from repeat-run flip rates.

    The accuracy ceiling `½ + √(1-2f)/2` is the most a seat flipping `f` of its
    re-runs can score; sensitivity is bounded by its flip rate on defect cases and
    specificity by its flip rate on clean cases. Feeding this to
    `reliability.posterior_defect_prob(..., ceilings=...)` caps how much any single
    verdict from a flaky seat can move the posterior — a seat cannot contribute more
    evidence than its self-consistency permits.

    `generous=False` (default) uses the point-estimate ceiling; `generous=True` uses
    the ceiling at the *lowest* plausible flip rate (the seat's best case), which
    only bites when even the seat's most favourable consistency can't support the
    claimed accuracy. Seats with no repeated cases (nothing measured) are omitted,
    leaving them uncapped.
    """
    judgments = list(judgments)
    sens = seat_stability(judgments, label="defect", ci=ci)
    spec = seat_stability(judgments, label="clean", ci=ci)

    def ceil_of(st) -> float:
        if st is None or not st.observed:
            return 1.0
        return st.ceiling_generous if generous else st.ceiling

    out: dict[str, tuple[float, float]] = {}
    for seat in set(sens) | set(spec):
        sc, tc = ceil_of(sens.get(seat)), ceil_of(spec.get(seat))
        if sc < 1.0 or tc < 1.0:          # omit fully-uncapped seats — no measured repeats
            out[seat] = (sc, tc)
    return out


# ---- Cross-checking the reliability model against the ceiling ----------------

@dataclass
class CeilingViolation:
    seat: str
    metric: str                   # "sensitivity" | "specificity"
    reported: float               # posterior mean from reliability.py
    reported_lo: float            # its lower credible bound
    ceiling: float                # always the most generous ceiling (from flip_lo)
    flip_rate: float
    flip_lo: float
    strict: bool                  # compared against the lower credible bound

    @property
    def claim(self) -> float:
        """The accuracy figure this tier held the seat to."""
        return self.reported_lo if self.strict else self.reported

    @property
    def overclaim(self) -> float:
        return self.claim - self.ceiling


def ceiling_violations(judgments, ci: float = 0.90,
                       strict: bool = True) -> list[CeilingViolation]:
    """Seats whose *measured* accuracy exceeds what their consistency permits.

    Both sides are given the benefit of the doubt: the seat's reported accuracy is
    taken at its **lower** credible bound, and compared against the ceiling implied
    by the **lowest** plausible flip rate. A violation that survives that is not an
    estimation artifact.

    What a violation actually means. On one fixed case distribution `s ≤ ceiling(f)`
    is a theorem, so it cannot be broken by a real reviewer. But `reliability.py`
    estimates accuracy over *every* labeled row, while the flip rate is only
    measurable on the subset of cases that were re-run. A violation says those two
    supports disagree: the seat scores well on the cases it was benchmarked on once,
    and behaves like a coin on the cases it was asked twice. Either the re-run cases
    are harder than the benchmark, or the single-pass accuracy was luck. Both make
    the single-pass number the one to distrust — and both are worth an alarm in a
    system that reports that number as an assurance level.

    Two tiers, differing only in which accuracy estimate the seat is held to. Both
    compare against `ceiling_generous`, the highest ceiling the flip data permits,
    so neither can be produced by a noisy flip rate.

    - `strict=True` holds the seat to its accuracy's **lower** credible bound. Even
      the most pessimistic reading of its accuracy clears the most generous ceiling,
      so the contradiction cannot be estimation noise on either side. The CLI fails
      closed on this.
    - `strict=False` holds it to the **posterior mean** — the number the system
      actually reports and aggregates into the trust score. An advisory, not a hard
      failure: reported accuracy can outrun the ceiling because the benchmark cases
      were easier than the re-run ones. Worth saying out loud all the same, because
      it means the seat's apparent competence lives in cases nobody asked it twice.
    """
    from .reliability import estimate_reliability

    # Iterated three times below; a generator would silently look like an empty log
    # from the second pass on, and report no violations. Fail closed on that.
    judgments = list(judgments)
    model = estimate_reliability(judgments, ci=ci)
    found: list[CeilingViolation] = []
    for label, metric, mean_attr, lo_attr in (
            ("defect", "sensitivity", "sens_mean", "sens_lo"),
            ("clean", "specificity", "spec_mean", "spec_lo")):
        for seat, st in seat_stability(judgments, label=label, ci=ci).items():
            r = model.get(seat)
            if not st.observed or r is None:
                continue
            claim = getattr(r, lo_attr) if strict else getattr(r, mean_attr)
            if claim > st.ceiling_generous + 1e-9:
                found.append(CeilingViolation(
                    seat=seat, metric=metric, reported=getattr(r, mean_attr),
                    reported_lo=getattr(r, lo_attr), ceiling=st.ceiling_generous,
                    flip_rate=st.flip_rate, flip_lo=st.flip_lo, strict=strict))
    return sorted(found, key=lambda v: -v.overclaim)


def format_stability(judgments, ci: float = 0.90) -> str:
    judgments = list(judgments)          # re-scanned per label and per tier
    stats = seat_stability(judgments, ci=ci)
    repeated = {s: st for s, st in stats.items() if st.observed}
    lines = ["", "=== Repeat-run stability (test-retest) ==="]
    if not repeated:
        lines.append("  (no case was judged twice by the same seat — "
                     "re-run --bench or --canary to collect repeats)")
        if stats:
            lines.append(f"  {len(stats)} seat(s) present, all single-pass.")
        return "\n".join(lines)

    lines.append("  seat          flip rate [90% CI]   pairs   kappa   ceiling   runs@5%")
    skewed = False
    for st in sorted(repeated.values(), key=lambda s: -s.flip_rate):
        if st.kappa is None:
            kap = "  n/a"
        elif st.kappa_informative:
            kap = f"{st.kappa:5.2f}"
        else:
            kap = f"{st.kappa:5.2f}~"
            skewed = True
        runs = "never" if st.repeats_for_5pct is None else f"{st.repeats_for_5pct:>5}"
        flag = "  (!) f>1/2: indistinguishable from chance" if not st.feasible else ""
        pad = "" if kap.endswith("~") else " "     # the marker eats one column
        lines.append(
            f"  {st.seat:<12}  {st.flip_rate*100:3.0f}% "
            f"[{st.flip_lo*100:3.0f}-{st.flip_hi*100:3.0f}]"
            f"      {st.n_pairs:>4}   {kap}{pad}   {st.ceiling*100:3.0f}%"
            f"     {runs}{flag}")
    lines.append("  (a seat cannot be more accurate than it is self-consistent: "
                 "|s - 1/2| <= sqrt(1-2f)/2)")
    lines.append("  (runs@5% = self-repeats whose majority vote flips <=5% of the time)")
    if skewed:
        lines.append("  (~ = marginals too skewed for kappa to carry information: a seat "
                     "that rejects 95%")
        lines.append("     of the time agrees with itself 90% of the time by chance, so "
                     "kappa collapses")
        lines.append("     toward zero however consistent it is. Read the flip rate, "
                     "not the kappa.)")

    strict = ceiling_violations(judgments, ci=ci, strict=True)
    advisory = [v for v in ceiling_violations(judgments, ci=ci, strict=False)
                if not any(s.seat == v.seat and s.metric == v.metric for s in strict)]

    lines.append("")
    if strict:
        lines.append("  !! Reported accuracy exceeds the consistency ceiling:")
        for v in strict:
            lines.append(
                f"     {v.seat:<12} {v.metric:<11} reported {v.reported*100:3.0f}% "
                f"(lower bound {v.reported_lo*100:3.0f}%) but flips {v.flip_rate*100:3.0f}% "
                f"of re-runs -> ceiling {v.ceiling*100:3.0f}%  "
                f"[+{v.overclaim*100:.1f} pts]")
        lines.append("     The single-pass accuracy is the number to distrust.")
    if advisory:
        lines.append("  Note: reported accuracy is above even the most generous ceiling "
                     "the flip data allows,")
        lines.append("  though its lower credible bound is not:")
        for v in advisory:
            lines.append(
                f"     {v.seat:<12} {v.metric:<11} reported {v.reported*100:3.0f}% vs "
                f"ceiling {v.ceiling*100:3.0f}% from a {v.flip_rate*100:3.0f}% flip rate "
                f"[+{v.overclaim*100:.1f} pts]")
        lines.append("     This seat's apparent competence lives in cases it was never "
                     "asked twice.")
    if not strict and not advisory:
        lines.append("  No seat's reported accuracy exceeds its consistency ceiling.")
    return "\n".join(lines)
