"""Test-retest stability: the flip rate, the accuracy ceiling it implies, and
the cross-check against the single-pass reliability model.

Every expected value below is either hand-computed from the closed form
`f = 2p(1-p)` or derived from a seeded simulation, never copied from a run.
"""

import random

import pytest

from agent_team import stability
from agent_team.stability import (
    accuracy_ceiling,
    ceiling_violations,
    repeat_groups,
    runs_for_stability,
    seat_stability,
    stability_ceilings,
)


def _rows(seat, case, verdicts, label="defect"):
    return [{"seat": seat, "case_id": case, "verdict": v, "label": label}
            for v in verdicts]


R, A = "reject", "approve"


# ---- The closed form ---------------------------------------------------------

@pytest.mark.parametrize("p", [0.5, 0.6, 0.75, 0.8, 0.9, 0.99, 1.0])
def test_accuracy_ceiling_inverts_2p_times_1_minus_p(p):
    """ceiling(2p(1-p)) == p for any competent seat: the map is an exact inverse."""
    assert accuracy_ceiling(2 * p * (1 - p)) == pytest.approx(p, abs=1e-12)


def test_accuracy_ceiling_hand_computed():
    assert accuracy_ceiling(0.0) == 1.0            # never flips -> no bound below 1
    assert accuracy_ceiling(0.18) == pytest.approx(0.9)   # 2(.9)(.1) = .18
    assert accuracy_ceiling(0.32) == pytest.approx(0.8)   # 2(.8)(.2) = .32
    assert accuracy_ceiling(0.5) == pytest.approx(0.5)    # a coin flip


def test_a_flip_rate_above_one_half_clamps_to_chance_rather_than_raising():
    """No p produces f > 1/2; an over-half sample is noise, and the seat is a coin."""
    assert accuracy_ceiling(0.67) == 0.5
    assert accuracy_ceiling(1.0) == 0.5


def test_ceiling_is_monotone_decreasing_in_the_flip_rate():
    ceilings = [accuracy_ceiling(f / 20) for f in range(0, 11)]
    assert ceilings == sorted(ceilings, reverse=True)


# ---- The estimator, hand-computed --------------------------------------------

def test_flip_rate_kappa_and_reject_rate_are_hand_computable():
    """3 cases x 2 runs: [R,R], [R,A], [A,A].

    discordant pairs = 0 + 1 + 0 = 1;  total pairs = 3  ->  f = 1/3
    reject rate = 3/6 = 1/2
    Po = 1 - 1/3 = 2/3;  Pe = .5^2 + .5^2 = 1/2;  kappa = (2/3 - 1/2)/(1/2) = 1/3
    """
    rows = (_rows("a", "c1", [R, R]) + _rows("a", "c2", [R, A])
            + _rows("a", "c3", [A, A]))
    st = seat_stability(rows)["a"]
    assert st.n_cases == 3 and st.n_runs == 6 and st.n_pairs == 3
    assert st.n_discordant == 1
    assert st.flip_rate == pytest.approx(1 / 3)
    assert st.reject_rate == pytest.approx(0.5)
    assert st.kappa == pytest.approx(1 / 3)


def test_one_case_run_four_times_uses_all_six_pairs():
    """n=4, r=2 -> discordant = r(n-r) = 4 out of C(4,2) = 6."""
    st = seat_stability(_rows("a", "c1", [R, R, A, A]))["a"]
    assert st.n_pairs == 6 and st.n_discordant == 4
    assert st.flip_rate == pytest.approx(2 / 3)


def test_a_single_realization_may_exceed_the_one_half_bound():
    """f > 1/2 is a legitimate *sample*; the ceiling clamps instead of pretending."""
    st = seat_stability(_rows("a", "c1", [R, R, A, A]))["a"]
    assert st.flip_rate > 0.5
    assert st.feasible is False
    assert st.ceiling == 0.5


def test_a_perfectly_consistent_seat_has_no_ceiling_below_one():
    st = seat_stability(_rows("a", "c1", [R, R, R]) + _rows("a", "c2", [A, A, A]))["a"]
    assert st.flip_rate == 0.0
    assert st.ceiling == 1.0
    assert st.repeats_for_5pct == 1          # already stable; one run is enough


def test_a_seat_that_never_varies_has_no_chance_baseline_so_kappa_is_none():
    st = seat_stability(_rows("a", "c1", [R, R]) + _rows("a", "c2", [R, R]))["a"]
    assert st.reject_rate == 1.0
    assert st.kappa is None                  # Pe == 1; (Po-Pe)/(1-Pe) undefined


# ---- The kappa paradox -------------------------------------------------------

def test_kappa_goes_negative_for_a_highly_consistent_but_skewed_seat():
    """The paradox: a seat that flips only 10% of re-runs can score kappa < 0.

    Rejecting 95% of the time means two runs agree 90.5% of the time by chance
    alone, so an observed 90% agreement scores *below* chance. Kappa is measuring
    the marginal, not the consistency. This is why the ceiling, not kappa, is the
    metric this module reports on.
    """
    rows = _simulate("a", lambda i: 0.95, n_cases=300, runs=4, seed=3)
    st = seat_stability(rows)["a"]

    assert st.flip_rate == pytest.approx(2 * 0.95 * 0.05, abs=0.03)   # ~0.095: consistent
    assert st.ceiling == pytest.approx(0.95, abs=0.03)                # and accurate
    assert st.chance_agreement > 0.85         # two runs mostly agree by chance alone
    assert st.kappa < 0.2                     # yet kappa is uninformative...
    assert st.kappa_informative is False      # ...and the report says so


def test_kappa_is_informative_when_the_marginals_are_balanced():
    rows = _simulate("a", lambda i: 0.5, n_cases=300, runs=4, seed=3)
    st = seat_stability(rows)["a"]
    assert st.chance_agreement == pytest.approx(0.5, abs=0.05)
    assert st.kappa_informative is True


def test_a_genuinely_flaky_seat_has_low_kappa_for_the_right_reason():
    """Balanced marginals, so its low kappa reflects flakiness, not skew."""
    rows = _simulate("a", lambda i: 0.5, n_cases=300, runs=4, seed=9)
    st = seat_stability(rows)["a"]
    assert st.kappa == pytest.approx(0.0, abs=0.1)
    assert st.kappa_informative is True
    # sqrt(1-2f) has an infinite derivative at f = 1/2, so the ceiling of a coin
    # flip is only ever pinned loosely from a finite sample. It stays near chance.
    assert st.ceiling < 0.65


# ---- What the log must contain ----------------------------------------------

def test_single_pass_seats_are_marked_unobserved_not_perfectly_stable():
    """One run per case yields zero pairs. That is 'unknown', not 'flip rate 0'."""
    st = seat_stability(_rows("a", "c1", [R]) + _rows("a", "c2", [A]))["a"]
    assert st.observed is False
    assert st.n_pairs == 0


def test_rows_without_a_case_id_cannot_be_recognized_as_repeats():
    rows = [{"seat": "a", "verdict": R, "label": "defect"},
            {"seat": "a", "verdict": A, "label": "defect"}]
    assert seat_stability(rows) == {}


def test_repeat_groups_filters_by_label():
    rows = _rows("a", "c1", [R, A], "defect") + _rows("a", "c2", [A, A], "clean")
    assert set(repeat_groups(rows, "defect")["a"]) == {"c1"}
    assert set(repeat_groups(rows, "clean")["a"]) == {"c2"}
    assert set(repeat_groups(rows, None)["a"]) == {"c1", "c2"}


def test_the_bootstrap_interval_is_deterministic_under_a_fixed_seed():
    rows = sum((_rows("a", f"c{i}", [R, A] if i % 3 else [R, R]) for i in range(9)), [])
    a = seat_stability(rows)["a"]
    b = seat_stability(rows)["a"]
    assert (a.flip_lo, a.flip_hi) == (b.flip_lo, b.flip_hi)
    assert a.flip_lo <= a.flip_rate <= a.flip_hi


def test_fewer_than_two_repeated_cases_yields_a_maximally_generous_ceiling():
    """One case cannot bound the spread, so the ceiling must not accuse anyone."""
    st = seat_stability(_rows("a", "c1", [R, A]))["a"]
    assert (st.flip_lo, st.flip_hi) == (0.0, 1.0)
    assert st.ceiling_generous == 1.0


# ---- The bound, verified by simulation ---------------------------------------

def _simulate(seat, per_case_p, n_cases, runs, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(n_cases):
        p = per_case_p(i)
        verdicts = [R if rng.random() < p else A for _ in range(runs)]
        rows += _rows(seat, f"c{i}", verdicts)
    return rows


def test_for_a_constant_p_seat_the_bound_is_tight():
    """p = 0.8 everywhere -> f -> 2(.8)(.2) = .32 and ceiling -> .8 == sensitivity.

    When every case shares one p, Jensen holds with equality and the ceiling is
    exactly the seat's sensitivity: the bound gives away nothing.
    """
    rows = _simulate("a", lambda i: 0.8, n_cases=400, runs=4, seed=11)
    st = seat_stability(rows)["a"]
    assert st.flip_rate == pytest.approx(0.32, abs=0.03)
    assert st.ceiling == pytest.approx(0.8, abs=0.03)
    assert st.ceiling == pytest.approx(st.reject_rate, abs=0.03)


def test_when_p_varies_across_cases_the_ceiling_is_valid_but_slack():
    """Half the cases p=1.0, half p=0.6  ->  s = 0.8, but f = (0 + .48)/2 = .24.

    E[p^2] > (E[p])^2 strictly, so f < 2s(1-s) = .32 and the ceiling rises above
    the true sensitivity. The bound must still *hold* (ceiling >= s).
    """
    rows = _simulate("a", lambda i: 1.0 if i % 2 else 0.6, n_cases=400, runs=4, seed=23)
    st = seat_stability(rows)["a"]
    s = st.reject_rate
    assert s == pytest.approx(0.8, abs=0.03)
    assert st.flip_rate == pytest.approx(0.24, abs=0.03)
    assert st.flip_rate < 2 * s * (1 - s)          # strict: p is not constant
    assert st.ceiling > s                          # valid, and slack
    assert st.ceiling == pytest.approx(accuracy_ceiling(0.24), abs=0.03)


def test_the_estimator_is_unbiased_regardless_of_how_many_runs_per_case():
    """E[r(n-r)/C(n,2)] = 2p(1-p) for every n, so 2 runs and 6 runs must agree."""
    two = seat_stability(_simulate("a", lambda i: 0.75, 900, 2, seed=5))["a"]
    six = seat_stability(_simulate("a", lambda i: 0.75, 300, 6, seed=5))["a"]
    expected = 2 * 0.75 * 0.25                     # = 0.375
    assert two.flip_rate == pytest.approx(expected, abs=0.04)
    assert six.flip_rate == pytest.approx(expected, abs=0.04)


# ---- Turning flakiness into a price ------------------------------------------

def test_runs_for_stability_hand_computed():
    """f = .18 -> p = .9.  k=1 flips .18;  k=3 flips 2(.972)(.028) = .0544 > .05;
    k=5 flips 2(.98144)(.01856) = .0364 <= .05.  So 5 repeats.
    """
    assert runs_for_stability(0.18, target_flip=0.05) == 5
    assert runs_for_stability(0.0, target_flip=0.05) == 1


def test_more_repeats_are_needed_as_a_seat_gets_flakier():
    ks = [runs_for_stability(f, 0.05) for f in (0.02, 0.10, 0.18, 0.30)]
    assert all(k is not None for k in ks)
    assert ks == sorted(ks)


def test_repeating_a_coin_flip_never_makes_it_a_reviewer():
    """The majority of k fair coins is still a fair coin."""
    assert runs_for_stability(0.5, target_flip=0.05) is None
    assert runs_for_stability(0.5, target_flip=0.05, max_k=999) is None


def test_a_coin_flip_seat_is_stable_only_against_a_coin_flip_target():
    assert runs_for_stability(0.5, target_flip=0.5) == 1


# ---- Cross-checking the reliability model ------------------------------------

def test_no_violation_when_accuracy_and_consistency_agree():
    """A seat that always rejects defects: sensitivity ~1, flip rate 0, ceiling 1."""
    rows = sum((_rows("a", f"c{i}", [R, R]) for i in range(10)), [])
    assert ceiling_violations(rows, strict=True) == []
    assert ceiling_violations(rows, strict=False) == []


def test_violation_fires_when_single_pass_accuracy_outruns_consistency():
    """40 single-pass defects it rejects, plus 10 re-run defects it flips every time.

    reliability.py sees 50 rejects / 10 approves -> sensitivity ~83%.
    stability sees a seat that contradicts itself on every re-run -> ceiling 50%.
    The two cannot describe the same reviewer on the same case distribution.
    """
    single = sum((_rows("a", f"s{i}", [R]) for i in range(40)), [])
    flipped = sum((_rows("a", f"r{i}", [R, A]) for i in range(10)), [])

    violations = ceiling_violations(single + flipped)
    assert len(violations) == 1
    v = violations[0]
    assert v.seat == "a" and v.metric == "sensitivity"
    assert v.strict is True
    assert v.ceiling == 0.5                # flips on every pair -> clamped
    assert v.claim == v.reported_lo        # strict holds it to the pessimistic side
    assert v.reported_lo > 0.5             # even that clears the ceiling
    assert v.overclaim > 0.0


def test_violation_compares_the_lower_bound_against_the_most_generous_ceiling():
    """Both sides get the benefit of the doubt, so a fired alarm is not noise."""
    single = sum((_rows("a", f"s{i}", [R]) for i in range(40)), [])
    flipped = sum((_rows("a", f"r{i}", [R, A]) for i in range(10)), [])
    rows = single + flipped

    from agent_team.reliability import estimate_reliability
    r = estimate_reliability(rows)["a"]
    st = seat_stability(rows, label="defect")["a"]

    v = ceiling_violations(rows)[0]
    assert v.reported_lo == pytest.approx(r.sens_lo)
    assert v.reported_lo < r.sens_mean               # the pessimistic side of accuracy
    assert v.ceiling == pytest.approx(st.ceiling_generous)
    assert st.ceiling_generous >= st.ceiling         # the optimistic side of the ceiling


def test_both_tiers_hold_the_seat_to_the_same_generous_ceiling():
    """The tiers differ only in the accuracy estimate, never in the ceiling.

    Otherwise the advisory would fire on well-behaved seats whose point ceiling
    happens to dip a fraction below their reported accuracy — noise, not signal.

    The flip rate here varies across cases, so the point ceiling (0.72) and the
    generous ceiling (0.82) genuinely differ; a tier that reached for the point
    ceiling would be caught.
    """
    single = sum((_rows("a", f"s{i}", [R]) for i in range(40)), [])
    mostly = sum((_rows("a", f"m{i}", [R, R, R, A]) for i in range(8)), [])
    always = sum((_rows("a", f"c{i}", [R, R, R, R]) for i in range(2)), [])
    rows = single + mostly + always

    st = seat_stability(rows, label="defect")["a"]
    assert st.ceiling < st.ceiling_generous          # the test can tell them apart

    for strict in (True, False):
        v = ceiling_violations(rows, strict=strict)[0]
        assert v.ceiling == pytest.approx(st.ceiling_generous)
        assert v.claim == (v.reported_lo if strict else v.reported)


def test_the_advisory_tier_catches_a_seat_the_strict_tier_lets_pass():
    """The realistic trap: strong on the single-pass benchmark, a coin on re-runs.

    Its posterior-mean sensitivity clears the ceiling, but its lower credible bound
    does not — so this warns rather than failing the build.
    """
    single = sum((_rows("a", f"s{i}", [R]) for i in range(8)), [])
    coin = sum((_rows("a", f"r{i}", [R, A, R, A]) for i in range(10)), [])
    rows = single + coin

    strict = ceiling_violations(rows, strict=True)
    advisory = ceiling_violations(rows, strict=False)

    assert strict == []                              # not beyond doubt
    assert [v.seat for v in advisory] == ["a"]       # but worth saying
    assert advisory[0].ceiling == 0.5                # a coin, on every re-run
    assert advisory[0].reported > advisory[0].ceiling
    assert advisory[0].reported_lo <= advisory[0].ceiling


def test_a_consistent_seat_triggers_neither_tier():
    """Guards the regression where a 10%-flip seat drew a +0.7pt advisory.

    A constant-p seat sits *on* its own ceiling (Jensen is tight), so the advisory
    can only stay quiet because it compares against the generous ceiling rather
    than the point estimate. That headroom is the credible interval's width.
    """
    rows = _simulate("a", lambda i: 0.9, n_cases=40, runs=4, seed=31)
    st = seat_stability(rows, label="defect")["a"]
    assert st.ceiling_generous > st.ceiling          # the headroom exists

    assert ceiling_violations(rows, strict=True) == []
    assert ceiling_violations(rows, strict=False) == []


def test_specificity_is_bounded_by_consistency_on_clean_cases_too():
    single = sum((_rows("a", f"s{i}", [A], "clean") for i in range(40)), [])
    flipped = sum((_rows("a", f"r{i}", [A, R], "clean") for i in range(10)), [])
    violations = ceiling_violations(single + flipped)
    assert [v.metric for v in violations] == ["specificity"]


def test_a_seat_with_no_repeats_is_never_accused():
    rows = sum((_rows("a", f"s{i}", [R]) for i in range(40)), [])
    assert ceiling_violations(rows) == []


def test_a_generator_of_judgments_is_not_silently_exhausted():
    """The rows are scanned once per label and once per tier; a one-shot iterator
    would look like an empty log after the first pass and clear every seat."""
    single = sum((_rows("a", f"s{i}", [R]) for i in range(40)), [])
    flipped = sum((_rows("a", f"r{i}", [R, A]) for i in range(10)), [])

    from_list = ceiling_violations(single + flipped)
    from_generator = ceiling_violations(iter(single + flipped))
    assert from_generator == from_list
    assert from_generator != []


# ---- Reporting ---------------------------------------------------------------

def test_format_stability_says_so_when_there_are_no_repeats():
    out = stability.format_stability(_rows("a", "c1", [R]))
    assert "no case was judged twice" in out


def test_format_stability_reports_the_violation_and_the_ceiling():
    single = sum((_rows("a", f"s{i}", [R]) for i in range(40)), [])
    flipped = sum((_rows("a", f"r{i}", [R, A]) for i in range(10)), [])
    out = stability.format_stability(single + flipped)
    assert "exceeds the consistency ceiling" in out
    assert "sensitivity" in out
    assert "distrust" in out


def test_format_stability_is_quiet_when_every_seat_is_within_its_ceiling():
    rows = sum((_rows("a", f"c{i}", [R, R]) for i in range(10)), [])
    out = stability.format_stability(rows)
    assert "No seat's reported accuracy exceeds its consistency ceiling." in out


def test_format_stability_explains_kappa_only_when_a_seat_is_skewed():
    skewed = stability.format_stability(_simulate("a", lambda i: 0.95, 300, 4, seed=3))
    assert "Read the flip rate, not the kappa." in skewed

    balanced = stability.format_stability(_simulate("a", lambda i: 0.5, 300, 4, seed=3))
    assert "Read the flip rate, not the kappa." not in balanced


def test_format_stability_marks_a_seat_that_cannot_be_told_from_chance():
    """Deterministic: n=4, r=2 on every case gives f = 4/6 > 1/2 with no sampling."""
    rows = sum((_rows("a", f"c{i}", [R, R, A, A]) for i in range(10)), [])
    out = stability.format_stability(rows)
    assert "indistinguishable from chance" in out
    assert "never" in out                     # no number of repeats rescues it


# ---- Ceilings consumed by the aggregation (spends the stability measurement) --

def test_stability_ceilings_bound_sensitivity_by_the_defect_flip_rate():
    # A seat with real defect-case flips but no clean repeats: capped on
    # sensitivity, left uncapped on specificity.
    rows = sum((_rows("a", f"c{i}", [R, R, R, R, A], "defect") for i in range(6)), [])
    st = seat_stability(rows, label="defect")["a"]
    ceilings = stability_ceilings(rows)
    assert "a" in ceilings
    sens_ceiling, spec_ceiling = ceilings["a"]
    assert sens_ceiling == pytest.approx(accuracy_ceiling(st.flip_rate))
    assert sens_ceiling < 1.0
    assert spec_ceiling == 1.0                # no clean repeats -> no specificity cap


def test_a_perfectly_consistent_seat_is_omitted_and_left_uncapped():
    # Never flips on either label -> ceiling 1.0 on both -> nothing to cap.
    rows = (_rows("a", "c1", [R, R, R], "defect")
            + _rows("a", "c2", [A, A, A], "clean"))
    assert "a" not in stability_ceilings(rows)


def test_generous_ceilings_are_at_least_as_high_as_point_estimate_ceilings():
    rows = sum((_rows("a", f"c{i}", [R, R, A]) for i in range(8)), [])
    point = stability_ceilings(rows)["a"][0]
    generous = stability_ceilings(rows, generous=True)["a"][0]
    assert generous >= point - 1e-12
