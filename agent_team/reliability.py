"""Reviewer reliability modeling and Bayesian trust aggregation.

The research question this answers: *can we measure an LLM reviewer's reliability
quantitatively, and use it instead of a naive vote?*

We treat each council seat as a noisy binary sensor of a latent fact — "this
change contains a defect (should be rejected)." From labeled judgments (gold
cases, canaries, and mutants are all ground-truth-labeled by construction) we
estimate two quantities per seat, with uncertainty:

    sensitivity  s = P(seat says reject | the change IS defective)   [defect recall]
    specificity  t = P(seat says approve | the change is clean)      [true-approve]

Each is a Beta-Binomial posterior (conjugate Beta prior + observed counts), so we
report not just a point estimate but an exact credible interval — a reviewer
judged on three cases and one judged on three hundred are not equally trustworthy,
and the interval width says so.

Then, instead of majority voting, we aggregate calibrated: assuming seats err
independently given the latent fact (the whole heterogeneous-jury design targets
exactly this decorrelation), the posterior odds of a defect are the prior odds
times the product of each seat's likelihood ratio. A seat with s = t = 0.5
contributes a likelihood ratio of 1 — it is ignored, automatically. A seat with a
long, clean track record dominates. The output is a **trust score**,
P(change is correct) = 1 - P(defect | votes), that is grounded in measured
reliability rather than one-agent-one-vote.

Everything here is deterministic, dependency-free numerics (a self-contained
regularized incomplete beta for the credible intervals) and is unit-tested
against hand-computed values.
"""

import math
from collections import defaultdict
from dataclasses import dataclass

# ---- Self-contained regularized incomplete beta (for exact credible intervals)

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes)."""
    MAXIT, EPS, FPMIN = 300, 1e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) ∈ [0, 1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_ppf(p: float, a: float, b: float) -> float:
    """Inverse CDF (quantile) of Beta(a, b) via bisection on betainc."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---- Reliability model -------------------------------------------------------

@dataclass
class SeatReliability:
    seat: str
    sens_mean: float          # P(reject | defect)
    sens_lo: float
    sens_hi: float
    n_defect: int             # labeled-defect judgments this seat was scored on
    spec_mean: float          # P(approve | clean)
    spec_lo: float
    spec_hi: float
    n_clean: int


def estimate_reliability(judgments, prior_a: float = 0.5, prior_b: float = 0.5,
                         ci: float = 0.90) -> dict[str, SeatReliability]:
    """Per-seat sensitivity/specificity posteriors from labeled judgments.

    `judgments` is an iterable of dicts with at least `seat`, `verdict`
    ("approve"|"reject"), and `label` ("defect"|"clean"). Unlabeled judgments are
    ignored (kept in the log for later, once an outcome is known). Prior defaults
    to Jeffreys Beta(0.5, 0.5).
    """
    agg: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fn": 0, "tn": 0, "fp": 0})
    for j in judgments:
        seat, verdict, label = j.get("seat"), j.get("verdict"), j.get("label")
        if not seat or verdict not in ("approve", "reject") or label not in ("defect", "clean"):
            continue
        a = agg[seat]
        if label == "defect":
            a["tp" if verdict == "reject" else "fn"] += 1
        else:
            a["tn" if verdict == "approve" else "fp"] += 1

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    model: dict[str, SeatReliability] = {}
    for seat, a in agg.items():
        sa, sb = prior_a + a["tp"], prior_b + a["fn"]   # sensitivity Beta
        ta, tb = prior_a + a["tn"], prior_b + a["fp"]   # specificity Beta
        model[seat] = SeatReliability(
            seat=seat,
            sens_mean=sa / (sa + sb),
            sens_lo=beta_ppf(lo_q, sa, sb), sens_hi=beta_ppf(hi_q, sa, sb),
            n_defect=a["tp"] + a["fn"],
            spec_mean=ta / (ta + tb),
            spec_lo=beta_ppf(lo_q, ta, tb), spec_hi=beta_ppf(hi_q, ta, tb),
            n_clean=a["tn"] + a["fp"],
        )
    return model


# ---- Bayesian trust aggregation ---------------------------------------------

def posterior_defect_prob(votes, model: dict[str, SeatReliability],
                          base_rate: float = 0.5, conservative: bool = False,
                          default_sens: float = 0.5, default_spec: float = 0.5,
                          eps: float = 1e-3) -> float:
    """P(defect | votes), combining seats as conditionally-independent sensors.

    `votes` is an iterable of (seat, is_reject). Seats absent from the model use
    the uninformative default (0.5/0.5 → a likelihood ratio of 1, i.e. no
    influence). `conservative=True` uses each seat's *lower* credible bound —
    a fail-closed posture that refuses to over-trust a thinly-evidenced seat.
    """
    log_odds = math.log(base_rate / (1 - base_rate))
    for seat, is_reject in votes:
        r = model.get(seat)
        if r is None:
            s, t = default_sens, default_spec
        elif conservative:
            s, t = r.sens_lo, r.spec_lo
        else:
            s, t = r.sens_mean, r.spec_mean
        s = min(max(s, eps), 1 - eps)
        t = min(max(t, eps), 1 - eps)
        lr = s / (1 - t) if is_reject else (1 - s) / t
        log_odds += math.log(lr)
    odds = math.exp(log_odds)
    return odds / (1 + odds)


def trust_score(votes, model, base_rate: float = 0.5, conservative: bool = False) -> float:
    """P(the change is correct) = 1 - P(defect | votes)."""
    return 1.0 - posterior_defect_prob(votes, model, base_rate, conservative)


def format_reliability(model: dict[str, SeatReliability]) -> str:
    if not model:
        return "\n=== Reviewer reliability ===\n(no labeled judgments yet)"
    lines = ["", "=== Reviewer reliability (Bayesian, from labeled judgments) ===",
             "  seat          sensitivity[90% CI] (n_def)   specificity[90% CI] (n_clean)"]

    def fmt(m, lo, hi, n):
        return f"{m*100:3.0f}% [{lo*100:3.0f}-{hi*100:3.0f}] (n={n})"

    # Worst defect-catchers first — that's who to re-route or retire.
    for s in sorted(model.values(), key=lambda r: r.sens_mean):
        lines.append(f"  {s.seat:<12}  {fmt(s.sens_mean, s.sens_lo, s.sens_hi, s.n_defect):<26}"
                     f"  {fmt(s.spec_mean, s.spec_lo, s.spec_hi, s.n_clean)}")
    lines.append("  (sensitivity = catches real defects; specificity = doesn't cry wolf; "
                 "wide CI = too few judgments to trust yet)")
    return "\n".join(lines)
