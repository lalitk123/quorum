"""Unit tests for the benchmark metric math and gold-set loader (no API)."""

import pytest

from agent_team.bench import CaseResult, cohen_kappa, compute_metrics, load_gold
from agent_team.models import Verdict


def _case(cid, expected_approve, decided_approve, verdicts=(), cost=0.0, error=""):
    return CaseResult(cid, expected_approve, decided_approve,
                      list(verdicts), cost, error)


def test_confusion_matrix_counts():
    results = [
        _case("good1", True, True),      # true approve
        _case("good2", True, False),     # false reject
        _case("bad1", False, False),     # true reject (defect caught)
        _case("bad2", False, True),      # false approve (defect shipped)
        _case("bad3", False, False),     # true reject
    ]
    m = compute_metrics(results)
    g = m["gate"]
    assert m["n_good"] == 2 and m["n_bad"] == 3
    assert g["true_approve"] == 1 and g["false_reject"] == 1
    assert g["true_reject"] == 2 and g["false_approve"] == 1
    assert g["defect_recall"] == 2 / 3          # 2 of 3 defects caught
    assert g["false_reject_rate"] == 1 / 2       # 1 of 2 good diffs blocked


def test_per_seat_detection_rates():
    # security seat catches both bad cases; product seat misses one.
    results = [
        _case("bad1", False, False, verdicts=[
            Verdict("security", False, "sqli"), Verdict("product", True, "fine")]),
        _case("bad2", False, False, verdicts=[
            Verdict("security", False, "sqli"), Verdict("product", False, "scope")]),
        _case("good1", True, True, verdicts=[
            Verdict("security", True, "ok"), Verdict("product", True, "ok")]),
    ]
    m = compute_metrics(results)
    assert m["seats"]["security"]["defect_recall"] == 1.0      # caught both
    assert m["seats"]["product"]["defect_recall"] == 0.5       # caught one
    assert m["seats"]["security"]["false_reject_rate"] == 0.0  # didn't block good


def test_errored_cases_excluded_from_scoring_but_costed():
    results = [
        _case("bad1", False, False, cost=0.10),
        _case("boom", False, False, cost=0.02, error="reviewer unreachable"),
    ]
    m = compute_metrics(results)
    assert m["n_scored"] == 1 and m["n_errored"] == 1
    assert m["cost_usd_total"] == pytest.approx(0.12)  # errored cost still counted


def test_empty_classes_give_none_not_zerodiv():
    m = compute_metrics([_case("good1", True, True)])   # no bad cases
    assert m["gate"]["defect_recall"] is None
    assert m["gate"]["false_reject_rate"] == 0.0


# ---- Cohen's kappa -----------------------------------------------------------

def test_kappa_perfect_agreement_is_one():
    pairs = [(True, True), (False, False), (True, True), (False, False)]
    assert cohen_kappa(pairs) == pytest.approx(1.0)


def test_kappa_chance_agreement_is_zero():
    # A and B each say True half the time, independently -> κ ≈ 0.
    pairs = [(True, True), (True, False), (False, True), (False, False)]
    assert cohen_kappa(pairs) == pytest.approx(0.0)


def test_kappa_undefined_only_when_both_raters_constant_same_class():
    # both raters always approve -> chance agreement is already total -> undefined.
    assert cohen_kappa([(True, True), (True, True)]) is None
    assert cohen_kappa([]) is None
    # only one rater constant -> still defined (here, no better than chance).
    assert cohen_kappa([(True, True), (False, True)]) == pytest.approx(0.0)


def test_gate_and_seat_kappa_vs_human():
    # gate perfectly matches labels; one seat perfectly matches, one is inverted.
    results = [
        _case("g", True, True, verdicts=[Verdict("s_good", True, "x"),
                                         Verdict("s_bad", False, "x")]),
        _case("b", False, False, verdicts=[Verdict("s_good", False, "x"),
                                           Verdict("s_bad", True, "x")]),
    ]
    m = compute_metrics(results)
    assert m["gate"]["kappa_vs_human"] == pytest.approx(1.0)
    assert m["seats"]["s_good"]["kappa_vs_human"] == pytest.approx(1.0)
    assert m["seats"]["s_bad"]["kappa_vs_human"] == pytest.approx(-1.0)


def test_inter_seat_kappa_flags_correlated_seats():
    # two seats that always vote identically -> mean pairwise κ = 1.0.
    results = [
        _case("g", True, True, verdicts=[Verdict("a", True, "x"), Verdict("b", True, "x")]),
        _case("b", False, False, verdicts=[Verdict("a", False, "x"), Verdict("b", False, "x")]),
    ]
    m = compute_metrics(results)
    assert m["inter_seat"]["mean_pairwise_kappa"] == pytest.approx(1.0)
    assert m["inter_seat"]["pairwise_kappa"]["a~b"] == pytest.approx(1.0)


def test_load_gold_reads_shipped_examples():
    cases = load_gold("bench/gold")
    by_id = {c.id: c for c in cases}
    assert by_id["G1"].expected_approve is True
    assert by_id["B1"].expected_approve is False
    assert "SELECT" in by_id["B1"].diff
    assert by_id["B2"].workspace_src is not None
