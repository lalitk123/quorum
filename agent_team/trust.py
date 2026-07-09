"""Formal trust score (idea #8) — the assurance layer as one legible number.

"Passed review" is a bit; trust is a quantity with reasons. This composes every
assurance signal Quorum produces into a single 0–100 score with a fully itemized
breakdown, so an approval reads like:

    Trust 88 / 100 (high)
      + council posterior (P correct 94%)     +56
      + deterministic checks passed           +12
      + clean-sandbox reproduction            +8
      + new tests have teeth                   +8
      + mutation score 82%                     +8
      - HIGH-risk surface (extra caution)      -7
      + reviewer track record (n≈40)           +5

This is the concrete form of the "AI software *assurance* layer" thesis (idea
#10): the coding worker is interchangeable; the product is this measurable,
explainable trust verdict aggregating deterministic checks, a calibrated council
posterior, reproduction, mutation, and risk. One component — the council
posterior — is a calibrated probability (`reliability.py`); the rest is a
transparent, tunable weighted tally, and the breakdown never hides which is which.

Deterministic and unit-tested.
"""

from dataclasses import dataclass, field

# Tunable weights. The council posterior is the anchor (up to +60); everything
# else adjusts around it. A failed deterministic check craters the score on
# purpose — fail-closed everywhere.
DEFAULT_WEIGHTS = {
    "posterior_scale": 60.0,
    "posterior_unknown_base": 30.0,
    "checks_pass": 12.0,
    "checks_fail": -40.0,
    "tamper_each": -8.0,
    "tamper_cap": 3,
    "sandbox_reproduced": 8.0,
    "sandbox_repro_fail": -30.0,
    "sandbox_teeth": 8.0,
    "sandbox_teeth_fail": -10.0,
    "mutation_scale": 10.0,
    "risk_high": -7.0,
    "risk_low": 2.0,
    "evidence_bonus": 5.0,
    "evidence_saturate_n": 30.0,
}


@dataclass
class AssuranceInputs:
    council_posterior_correct: float | None = None   # P(correct), [0,1], from reliability
    checks_passed: bool | None = None
    tamper_findings: int = 0
    sandbox_reproduced: bool | None = None
    sandbox_teeth: bool | None = None
    mutation_score: float | None = None
    risk_level: str = "medium"
    reviewer_evidence_n: float | None = None         # mean labeled judgments backing seats


@dataclass
class TrustComponent:
    name: str
    delta: float
    detail: str = ""


@dataclass
class TrustReport:
    score: float
    grade: str
    components: list[TrustComponent] = field(default_factory=list)


def _grade(score: float) -> str:
    if score >= 85:
        return "high"
    if score >= 70:
        return "moderate"
    if score >= 50:
        return "low"
    return "block"


def assess_trust(inp: AssuranceInputs, weights: dict | None = None) -> TrustReport:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    comps: list[TrustComponent] = []

    if inp.council_posterior_correct is not None:
        p = min(max(inp.council_posterior_correct, 0.0), 1.0)
        comps.append(TrustComponent("council posterior", p * w["posterior_scale"],
                                    f"P(correct) {p:.0%}"))
    else:
        comps.append(TrustComponent("council posterior", w["posterior_unknown_base"],
                                    "no reliability model yet (neutral prior)"))

    if inp.checks_passed is True:
        comps.append(TrustComponent("deterministic checks", w["checks_pass"], "passed"))
    elif inp.checks_passed is False:
        comps.append(TrustComponent("deterministic checks", w["checks_fail"], "FAILED"))

    if inp.tamper_findings:
        n = min(inp.tamper_findings, int(w["tamper_cap"]))
        comps.append(TrustComponent("tamper findings", n * w["tamper_each"],
                                    f"{inp.tamper_findings} finding(s)"))

    if inp.sandbox_reproduced is True:
        comps.append(TrustComponent("clean-sandbox reproduction", w["sandbox_reproduced"],
                                    "tests pass from a clean checkout"))
    elif inp.sandbox_reproduced is False:
        comps.append(TrustComponent("clean-sandbox reproduction", w["sandbox_repro_fail"],
                                    "FAILED in a clean checkout"))

    if inp.sandbox_teeth is True:
        comps.append(TrustComponent("test teeth", w["sandbox_teeth"], "new tests fail without the code"))
    elif inp.sandbox_teeth is False:
        comps.append(TrustComponent("test teeth", w["sandbox_teeth_fail"], "new tests pass without the code"))

    if inp.mutation_score is not None:
        ms = min(max(inp.mutation_score, 0.0), 1.0)
        comps.append(TrustComponent("mutation score", ms * w["mutation_scale"], f"{ms:.0%} killed"))

    if inp.risk_level == "high":
        comps.append(TrustComponent("risk surface", w["risk_high"], "HIGH — extra caution"))
    elif inp.risk_level == "low":
        comps.append(TrustComponent("risk surface", w["risk_low"], "LOW"))

    if inp.reviewer_evidence_n is not None:
        frac = min(1.0, inp.reviewer_evidence_n / w["evidence_saturate_n"])
        comps.append(TrustComponent("reviewer track record", frac * w["evidence_bonus"],
                                    f"n≈{inp.reviewer_evidence_n:.0f} labeled judgments"))

    score = min(100.0, max(0.0, sum(c.delta for c in comps)))
    return TrustReport(score=score, grade=_grade(score), components=comps)


def format_trust(report: TrustReport) -> str:
    lines = [f"Trust {report.score:.0f} / 100 ({report.grade})"]
    for c in sorted(report.components, key=lambda x: x.delta, reverse=True):
        sign = "+" if c.delta >= 0 else "-"
        lines.append(f"  {sign} {c.name:<28} {sign}{abs(c.delta):>4.0f}"
                     + (f"   {c.detail}" if c.detail else ""))
    return "\n".join(lines)
