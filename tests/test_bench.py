"""Unit tests for the benchmark metric math and gold-set loader (no API)."""

import pytest

from agent_team.bench import CaseResult, compute_metrics, load_gold
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


def test_load_gold_reads_shipped_examples():
    cases = load_gold("bench/gold")
    by_id = {c.id: c for c in cases}
    assert by_id["G1"].expected_approve is True
    assert by_id["B1"].expected_approve is False
    assert "SELECT" in by_id["B1"].diff
    assert by_id["B2"].workspace_src is not None
