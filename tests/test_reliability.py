"""Reviewer reliability + Bayesian trust aggregation tests.

Deterministic, no API. The probability results are hand-computable, which is the
point — the aggregation is auditable arithmetic, not a black box.
"""

import pytest

from agent_team import judgments
from agent_team.reliability import (
    beta_ppf, betainc, estimate_reliability, posterior_defect_prob, trust_score,
)


# ---- Numerics: regularized incomplete beta -----------------------------------

def test_betainc_uniform_is_identity():
    # I_x(1,1) = x
    for x in (0.1, 0.37, 0.9):
        assert betainc(1.0, 1.0, x) == pytest.approx(x, abs=1e-9)


def test_betainc_known_integer_value():
    # I_0.5(2,3) = (C(4,2)+C(4,3)+C(4,4)) / 2^4 = 11/16
    assert betainc(2.0, 3.0, 0.5) == pytest.approx(11 / 16, abs=1e-9)


def test_betainc_symmetry_relation():
    # I_x(a,b) = 1 - I_{1-x}(b,a)
    assert betainc(3.0, 5.0, 0.4) == pytest.approx(1 - betainc(5.0, 3.0, 0.6), abs=1e-9)


def test_beta_ppf_inverts_betainc():
    a, b, q = 4.0, 7.0, 0.9
    x = beta_ppf(q, a, b)
    assert betainc(a, b, x) == pytest.approx(q, abs=1e-6)


# ---- Reliability estimation --------------------------------------------------

def _j(seat, verdict, label):
    return {"seat": seat, "verdict": verdict, "label": label}


def test_oracle_seat_scores_high_coin_seat_scores_half():
    js = []
    for _ in range(10):
        js += [_j("oracle", "reject", "defect"), _j("oracle", "approve", "clean")]
    # coin: rejects half the defects and half the cleans
    for i in range(10):
        js.append(_j("coin", "reject" if i % 2 else "approve", "defect"))
        js.append(_j("coin", "reject" if i % 2 else "approve", "clean"))
    model = estimate_reliability(js)

    assert model["oracle"].sens_mean > 0.9 and model["oracle"].spec_mean > 0.9
    assert model["coin"].sens_mean == pytest.approx(0.5, abs=0.05)
    assert model["coin"].spec_mean == pytest.approx(0.5, abs=0.05)


def test_credible_interval_is_ordered_and_narrows_with_data():
    thin = estimate_reliability([_j("a", "reject", "defect")])["a"]
    assert thin.sens_lo < thin.sens_mean < thin.sens_hi
    thick = estimate_reliability(
        [_j("a", "reject", "defect")] * 200)["a"]
    # more agreeing evidence -> tighter interval
    assert (thick.sens_hi - thick.sens_lo) < (thin.sens_hi - thin.sens_lo)


def test_unlabeled_and_malformed_rows_ignored():
    model = estimate_reliability([
        _j("a", "reject", "defect"),
        {"seat": "a", "verdict": "reject", "label": None},   # unlabeled
        {"seat": "a", "verdict": "maybe", "label": "defect"},  # bad verdict
    ])
    assert model["a"].n_defect == 1


# ---- Bayesian aggregation (hand-verified) ------------------------------------

def _model(**seats):
    from agent_team.reliability import SeatReliability
    return {name: SeatReliability(name, s, s, s, 50, t, t, t, 50)
            for name, (s, t) in seats.items()}


def test_single_reliable_reject_drives_defect_probability():
    m = _model(sec=(0.9, 0.9))
    # LR = 0.9/0.1 = 9, prior odds 1 -> P(defect) = 9/10
    p = posterior_defect_prob([("sec", True)], m, base_rate=0.5)
    assert p == pytest.approx(0.9, abs=1e-6)


def test_single_reliable_approve_lowers_defect_probability():
    m = _model(sec=(0.9, 0.9))
    p = posterior_defect_prob([("sec", False)], m, base_rate=0.5)
    assert p == pytest.approx(0.1, abs=1e-6)


def test_coin_flip_seat_has_no_influence():
    m = _model(coin=(0.5, 0.5))
    assert posterior_defect_prob([("coin", True)], m, base_rate=0.5) == pytest.approx(0.5)
    assert posterior_defect_prob([("coin", False)], m, base_rate=0.3) == pytest.approx(0.3)


def test_independent_reliable_rejects_compound():
    m = _model(a=(0.9, 0.9), b=(0.9, 0.9))
    # odds = 1 * 9 * 9 = 81 -> P = 81/82
    p = posterior_defect_prob([("a", True), ("b", True)], m, base_rate=0.5)
    assert p == pytest.approx(81 / 82, abs=1e-6)


def test_unknown_seat_uses_uninformative_default():
    assert posterior_defect_prob([("stranger", True)], {}, base_rate=0.4) == pytest.approx(0.4)


def test_trust_score_is_complement():
    m = _model(sec=(0.9, 0.9))
    votes = [("sec", True)]
    assert trust_score(votes, m) == pytest.approx(1 - posterior_defect_prob(votes, m))


def test_conservative_uses_lower_bounds():
    from agent_team.reliability import SeatReliability
    # wide interval: mean high but lower bound low
    m = {"x": SeatReliability("x", sens_mean=0.9, sens_lo=0.4, sens_hi=0.99, n_defect=2,
                              spec_mean=0.9, spec_lo=0.4, spec_hi=0.99, n_clean=2)}
    optimistic = posterior_defect_prob([("x", True)], m, conservative=False)
    cautious = posterior_defect_prob([("x", True)], m, conservative=True)
    # a reject from a thinly-evidenced seat is weaker evidence when we're cautious
    assert cautious < optimistic


# ---- Ceiling-capped likelihood ratio (spends the stability measurement) ------

def test_ceiling_caps_a_strong_seats_reject_evidence():
    m = _model(sec=(0.9, 0.9))
    # Uncapped a reject drives P(defect) to 0.9 (LR = 0.9/0.1 = 9).
    uncapped = posterior_defect_prob([("sec", True)], m, base_rate=0.5)
    # A ceiling of 0.6 on both caps s,t to 0.6: LR = 0.6/0.4 = 1.5 -> 1.5/2.5 = 0.6.
    capped = posterior_defect_prob([("sec", True)], m, base_rate=0.5,
                                   ceilings={"sec": (0.6, 0.6)})
    assert uncapped == pytest.approx(0.9, abs=1e-6)
    assert capped == pytest.approx(0.6, abs=1e-6)
    assert capped < uncapped


def test_ceiling_at_half_removes_all_influence():
    # A seat whose consistency ceiling is 1/2 is a coin: its likelihood ratio is 1,
    # so the posterior is left at the base rate whichever way it votes.
    m = _model(sec=(0.95, 0.95))
    ceilings = {"sec": (0.5, 0.5)}
    assert posterior_defect_prob([("sec", True)], m, base_rate=0.3,
                                 ceilings=ceilings) == pytest.approx(0.3)
    assert posterior_defect_prob([("sec", False)], m, base_rate=0.7,
                                 ceilings=ceilings) == pytest.approx(0.7)


def test_ceiling_above_measured_accuracy_does_not_change_anything():
    m = _model(sec=(0.9, 0.9))
    base = posterior_defect_prob([("sec", True)], m, base_rate=0.5)
    # A ceiling that sits above the measured accuracy never bites (min keeps it).
    loose = posterior_defect_prob([("sec", True)], m, base_rate=0.5,
                                  ceilings={"sec": (0.99, 0.99)})
    assert loose == pytest.approx(base)


def test_seat_absent_from_ceilings_is_left_uncapped():
    m = _model(a=(0.9, 0.9), b=(0.9, 0.9))
    # Only 'a' is capped; 'b' behaves exactly as it would with no ceilings at all.
    with_ceil = posterior_defect_prob([("a", True), ("b", True)], m, base_rate=0.5,
                                      ceilings={"a": (0.5, 0.5)})
    b_only = posterior_defect_prob([("b", True)], m, base_rate=0.5)
    # 'a' contributes nothing (ceiling 1/2), so the pair equals 'b' alone.
    assert with_ceil == pytest.approx(b_only)


# ---- Judgment log round-trip -------------------------------------------------

def test_judgment_log_roundtrip(tmp_path):
    log = str(tmp_path / "j.jsonl")
    rows = [judgments.Judgment("security", "reject", "defect", source="gold"),
            judgments.Judgment("qa", "approve", "clean", source="gold")]
    judgments.append_judgments(log, rows)
    judgments.append_judgments(log, [{"seat": "arch", "verdict": "reject", "label": "defect"}])
    loaded = judgments.load_judgments(log)
    assert len(loaded) == 3
    assert loaded[0]["seat"] == "security" and loaded[0]["label"] == "defect"
    model = estimate_reliability(loaded)
    assert set(model) == {"security", "qa", "arch"}


def test_load_missing_log_is_empty(tmp_path):
    assert judgments.load_judgments(str(tmp_path / "nope.jsonl")) == []
