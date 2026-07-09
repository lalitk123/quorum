"""Independence audit: the assumption under the Bayesian council and the selector.

These tests pin the three ways a two-seat panel can relate — clones, independent,
complementary — and check that each estimator says the right thing about each.
"""

import math

from agent_team.independence import (
    correlation_discount, design_effect, effective_seats, empirical_detection,
    independence_gap, make_empirical_detector, mean_phi, pairwise_dependence,
)
from agent_team.reliability import estimate_reliability, posterior_defect_prob
from agent_team.selection import select_reviewers, team_detection


def _rows(pattern: dict[str, list[bool]], label="defect", prefix="c"):
    """`{seat: [correct_on_case_0, ...]}` -> judgment rows."""
    out = []
    for seat, hits in pattern.items():
        for i, correct in enumerate(hits):
            if label == "defect":
                verdict = "reject" if correct else "approve"
            else:
                verdict = "approve" if correct else "reject"
            out.append({"seat": seat, "verdict": verdict, "label": label,
                        "source": "gold", "case_id": f"{prefix}{i}"})
    return out


# A panel that actually carries information: 7/8 sensitivity *and* 7/8 specificity,
# so each seat's likelihood ratio is far from 1 and the discount has something to bite.
INFORMATIVE = (
    _rows({"a": [1, 1, 1, 1, 1, 1, 1, 0], "b": [1, 1, 1, 1, 1, 1, 1, 0]}, "defect", "d")
    + _rows({"a": [1, 1, 1, 1, 1, 1, 1, 0], "b": [1, 1, 1, 1, 1, 1, 1, 0]}, "clean", "n")
)


# Two seats, 8 defect cases, each with sensitivity 1/2 — but arranged three ways.
CLONES = _rows({"a": [1, 1, 1, 1, 0, 0, 0, 0], "b": [1, 1, 1, 1, 0, 0, 0, 0]})
COMPLEMENTARY = _rows({"a": [1, 1, 1, 1, 0, 0, 0, 0], "b": [0, 0, 0, 0, 1, 1, 1, 1]})
INDEPENDENT = _rows({"a": [1, 1, 1, 1, 0, 0, 0, 0], "b": [1, 1, 0, 0, 1, 1, 0, 0]})


def test_all_three_panels_have_identical_marginal_sensitivities():
    # The whole point: marginals cannot distinguish these panels. Only the joint can.
    for rows in (CLONES, COMPLEMENTARY, INDEPENDENT):
        model = estimate_reliability(rows)
        assert math.isclose(model["a"].sens_mean, 0.5, abs_tol=0.02)
        assert math.isclose(model["b"].sens_mean, 0.5, abs_tol=0.02)
        # ...and the independence model therefore predicts the same 75% for all three.
        assert math.isclose(team_detection(["a", "b"], model), 0.75, abs_tol=0.02)


def test_clones_detect_no_better_than_one_seat():
    est = empirical_detection(CLONES, ["a", "b"])
    assert est.n_cases == 8 and est.n_detected == 4
    assert math.isclose(est.mean, 0.5, abs_tol=0.06)     # NOT the 75% independence claims


def test_complementary_seats_detect_everything():
    est = empirical_detection(COMPLEMENTARY, ["a", "b"])
    assert est.n_detected == 8
    assert est.mean > 0.9


def test_independent_seats_match_the_independence_prediction():
    est = empirical_detection(INDEPENDENT, ["a", "b"])
    assert est.n_detected == 6                            # 6/8 = 0.75, exactly 1-(1/2)(1/2)
    assert math.isclose(est.mean, 0.75, abs_tol=0.06)


def test_independence_gap_sign_flags_the_unsafe_direction():
    # Clones: independence is optimistic -> positive inflation -> selector under-provisions.
    assert independence_gap(CLONES, ["a", "b"]).inflation > 0.15
    # Complementary: independence is pessimistic -> negative inflation (safe direction).
    assert independence_gap(COMPLEMENTARY, ["a", "b"]).inflation < -0.15
    # Truly independent: the model is right.
    assert abs(independence_gap(INDEPENDENT, ["a", "b"]).inflation) < 0.06


def test_phi_separates_the_three_panels():
    (clone,) = pairwise_dependence(CLONES, ["a", "b"])
    (comp,) = pairwise_dependence(COMPLEMENTARY, ["a", "b"])
    (indep,) = pairwise_dependence(INDEPENDENT, ["a", "b"])
    assert math.isclose(clone.phi, 1.0, abs_tol=1e-9)
    assert math.isclose(comp.phi, -1.0, abs_tol=1e-9)
    assert math.isclose(indep.phi, 0.0, abs_tol=1e-9)


def test_clone_blind_spots_overlap_more_than_independence_predicts():
    (clone,) = pairwise_dependence(CLONES, ["a", "b"])
    assert math.isclose(clone.both_miss, 0.5)               # they fail together, always
    assert math.isclose(clone.both_miss_independent, 0.25)  # independence says half as often
    assert clone.excess_joint_miss > 0.2
    # And b never covers for a: when a misses, b misses too.
    assert clone.conditional_coverage == 0.0


def test_complementary_seat_always_covers_for_the_other():
    (comp,) = pairwise_dependence(COMPLEMENTARY, ["a", "b"])
    assert comp.conditional_coverage == 1.0
    assert comp.both_miss == 0.0


def test_conditional_coverage_is_none_when_a_seat_never_misses():
    rows = _rows({"a": [1, 1, 1, 1], "b": [1, 0, 1, 0]})
    (p,) = pairwise_dependence(rows, ["a", "b"])
    assert p.conditional_coverage is None       # a never misses -> nothing to cover for
    assert p.phi is None                        # a is constant -> correlation undefined


def test_mean_phi_clamps_complementarity_to_zero_for_fail_closed():
    # A negative φ would *increase* effective seats and inflate confidence. Refuse.
    assert mean_phi(COMPLEMENTARY, ["a", "b"]) == 0.0
    assert math.isclose(mean_phi(CLONES, ["a", "b"]), 1.0)


def test_design_effect_and_effective_seats():
    assert design_effect(1, 0.9) == 1.0                      # a lone seat cannot self-correlate
    assert math.isclose(design_effect(5, 0.0), 1.0)          # independent -> no inflation
    assert math.isclose(effective_seats(5, 0.0), 5.0)
    assert math.isclose(design_effect(5, 1.0), 5.0)          # perfect clones...
    assert math.isclose(effective_seats(5, 1.0), 1.0)        # ...are worth one seat
    assert math.isclose(effective_seats(4, 1 / 3), 2.0)      # deff = 1 + (1/3)(3) = 2


def test_correlation_discount_is_identity_for_independent_seats():
    assert math.isclose(correlation_discount(INDEPENDENT, ["a", "b"]), 1.0, abs_tol=1e-9)
    assert math.isclose(correlation_discount(CLONES, ["a", "b"]), 0.5, abs_tol=1e-9)


def test_a_coin_flipper_seat_contributes_nothing_with_or_without_a_discount():
    # INDEPENDENT has no clean cases, so specificity falls back to the 0.5 prior:
    # likelihood ratio 1, no evidence either way. Discounting nothing changes nothing.
    model = estimate_reliability(INDEPENDENT)
    votes = [("a", True), ("b", True)]
    assert math.isclose(posterior_defect_prob(votes, model, base_rate=0.5), 0.5, abs_tol=1e-9)
    assert math.isclose(posterior_defect_prob(votes, model, discount=0.5), 0.5, abs_tol=1e-9)


def test_discount_pulls_the_posterior_back_toward_the_base_rate():
    model = estimate_reliability(INFORMATIVE)
    votes = [("a", True), ("b", True)]           # both seats reject
    full = posterior_defect_prob(votes, model, base_rate=0.5)
    halved = posterior_defect_prob(votes, model, base_rate=0.5, discount=0.5)
    assert full > halved > 0.5                   # still leans defect, but less confidently
    # Correlated approvals must not stack into false comfort either.
    approve = [("a", False), ("b", False)]
    assert posterior_defect_prob(approve, model, discount=0.5) > \
           posterior_defect_prob(approve, model, discount=1.0)


def test_discount_of_zero_ignores_the_panel_entirely():
    model = estimate_reliability(INFORMATIVE)
    votes = [("a", True), ("b", True)]
    assert math.isclose(posterior_defect_prob(votes, model, base_rate=0.3, discount=0.0),
                        0.3, abs_tol=1e-9)


def test_discount_of_one_is_exactly_the_old_behavior():
    model = estimate_reliability(INFORMATIVE)
    votes = [("a", True), ("b", False)]
    assert posterior_defect_prob(votes, model) == \
           posterior_defect_prob(votes, model, discount=1.0)


def test_clone_panel_is_discounted_to_a_single_seats_worth_of_evidence():
    """The end-to-end correction: two clones rejecting should persuade us exactly as
    much as one seat rejecting, not twice as much."""
    clones = (_rows({"a": [1, 1, 1, 1, 1, 1, 1, 0], "b": [1, 1, 1, 1, 1, 1, 1, 0]}, "defect", "d")
              + _rows({"a": [1, 1, 1, 1, 1, 1, 1, 0], "b": [1, 1, 1, 1, 1, 1, 1, 0]}, "clean", "n"))
    model = estimate_reliability(clones)
    disc = correlation_discount(clones, ["a", "b"])
    assert math.isclose(disc, 0.5, abs_tol=1e-9)          # 2 clones -> 1 effective seat

    both = posterior_defect_prob([("a", True), ("b", True)], model, discount=disc)
    one = posterior_defect_prob([("a", True)], model, discount=1.0)
    assert math.isclose(both, one, abs_tol=1e-9)


# ---- The payoff: the selector stops buying clones ----------------------------

def test_selector_refuses_to_count_a_clone_as_added_coverage():
    model = estimate_reliability(CLONES)
    # Independence model: two 50% seats "reach" 75%, so it happily buys both.
    naive = select_reviewers(model, target_detection=0.7)
    assert len(naive.seats) == 2 and naive.detection > 0.7

    # Measured joint: the clone adds nothing, so the target is unreachable and the
    # selector stops rather than pretending a second copy of seat `a` bought coverage.
    measured = select_reviewers(model, target_detection=0.7,
                                joint=make_empirical_detector(CLONES, min_cases=3))
    assert len(measured.seats) == 1
    assert measured.detection < 0.7
    assert measured.steps[0]["estimator"] == "measured"


def test_selector_still_buys_a_genuinely_complementary_second_seat():
    model = estimate_reliability(COMPLEMENTARY)
    sel = select_reviewers(model, target_detection=0.9,
                           joint=make_empirical_detector(COMPLEMENTARY, min_cases=3))
    assert sorted(sel.seats) == ["a", "b"]
    assert sel.detection > 0.9


def test_selector_falls_back_to_independence_when_joint_data_is_thin():
    thin = _rows({"a": [1, 0], "b": [1, 0]})          # only 2 joint cases
    model = estimate_reliability(thin)
    sel = select_reviewers(model, target_detection=0.9,
                           joint=make_empirical_detector(thin, min_cases=3))
    assert sel.steps and all(st["estimator"] == "independence" for st in sel.steps)


def test_empirical_detection_needs_a_defined_joint():
    # Seats that never reviewed the same case have no joint distribution.
    disjoint = [{"seat": "a", "verdict": "reject", "label": "defect", "case_id": "c1"},
                {"seat": "b", "verdict": "reject", "label": "defect", "case_id": "c2"}]
    est = empirical_detection(disjoint, ["a", "b"])
    assert est.observed is False and est.n_cases == 0
    # ...but each alone is measurable.
    assert empirical_detection(disjoint, ["a"]).n_cases == 1


def test_rows_without_case_id_are_ignored_rather_than_faking_a_joint():
    rows = [{"seat": "a", "verdict": "reject", "label": "defect"},
            {"seat": "b", "verdict": "approve", "label": "defect"}]
    assert empirical_detection(rows, ["a", "b"]).observed is False
