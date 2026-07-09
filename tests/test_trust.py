"""Formal trust-score tests. Deterministic composite; each signal moves the
score in the expected direction and the total is clamped to [0, 100]."""

import pytest

from agent_team.trust import AssuranceInputs, assess_trust


def test_strong_change_scores_high():
    r = assess_trust(AssuranceInputs(
        council_posterior_correct=0.95, checks_passed=True, tamper_findings=0,
        sandbox_reproduced=True, sandbox_teeth=True, mutation_score=0.9,
        risk_level="medium", reviewer_evidence_n=40))
    assert r.grade == "high" and r.score >= 85


def test_failing_checks_crater_trust():
    strong = assess_trust(AssuranceInputs(council_posterior_correct=0.9, checks_passed=True))
    failed = assess_trust(AssuranceInputs(council_posterior_correct=0.9, checks_passed=False))
    assert failed.score < strong.score
    assert failed.grade in ("low", "block")


def test_higher_posterior_raises_score():
    lo = assess_trust(AssuranceInputs(council_posterior_correct=0.3, checks_passed=True))
    hi = assess_trust(AssuranceInputs(council_posterior_correct=0.9, checks_passed=True))
    assert hi.score > lo.score


def test_tamper_and_no_teeth_lower_score():
    base = AssuranceInputs(council_posterior_correct=0.8, checks_passed=True,
                           sandbox_reproduced=True, sandbox_teeth=True)
    worse = AssuranceInputs(council_posterior_correct=0.8, checks_passed=True,
                            sandbox_reproduced=True, sandbox_teeth=False, tamper_findings=2)
    assert assess_trust(worse).score < assess_trust(base).score


def test_score_is_clamped_to_unit_range():
    z = assess_trust(AssuranceInputs(council_posterior_correct=0.0, checks_passed=False,
                                     tamper_findings=9, sandbox_reproduced=False,
                                     sandbox_teeth=False, risk_level="high"))
    assert z.score == 0.0 and z.grade == "block"
    top = assess_trust(AssuranceInputs(council_posterior_correct=1.0, checks_passed=True,
                                       sandbox_reproduced=True, sandbox_teeth=True,
                                       mutation_score=1.0, risk_level="low",
                                       reviewer_evidence_n=100))
    assert top.score <= 100.0


def test_unknown_posterior_uses_neutral_prior_component():
    r = assess_trust(AssuranceInputs(council_posterior_correct=None, checks_passed=True))
    names = {c.name for c in r.components}
    assert "council posterior" in names
    posterior = next(c for c in r.components if c.name == "council posterior")
    assert "neutral prior" in posterior.detail


def test_high_risk_penalizes_relative_to_low():
    hi = assess_trust(AssuranceInputs(council_posterior_correct=0.8, risk_level="high"))
    lo = assess_trust(AssuranceInputs(council_posterior_correct=0.8, risk_level="low"))
    assert lo.score > hi.score
