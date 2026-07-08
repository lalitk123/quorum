"""Adaptive reviewer selection (RQ2) tests. Deterministic; detection math is
hand-verifiable under the independence model P = 1 - Π(1 - sensitivity)."""

import pytest

from agent_team.reliability import SeatReliability
from agent_team.selection import (
    for_condition, recall_per_dollar, select_reviewers, team_detection,
)


def _seat(name, sens):
    return SeatReliability(name, sens, sens, sens, 50, 0.9, 0.9, 0.9, 50)


def _model(**seats):
    return {name: _seat(name, s) for name, s in seats.items()}


# ---- detection math ----------------------------------------------------------

def test_team_detection_independence_formula():
    m = _model(a=0.8, b=0.8)
    # 1 - 0.2*0.2 = 0.96
    assert team_detection(["a", "b"], m) == pytest.approx(0.96)


def test_team_detection_empty_is_zero():
    assert team_detection([], _model(a=0.9)) == 0.0


# ---- greedy selection --------------------------------------------------------

def test_selects_until_target_met():
    m = _model(strong=0.9, mid=0.6, weak=0.5)
    sel = select_reviewers(m, target_detection=0.95)
    # strong alone = 0.90; + mid -> 1 - 0.1*0.4 = 0.96 >= 0.95
    assert sel.seats == ["strong", "mid"]
    assert sel.detection == pytest.approx(0.96)


def test_stops_when_target_already_reached():
    m = _model(strong=0.97, mid=0.6)
    sel = select_reviewers(m, target_detection=0.95)
    assert sel.seats == ["strong"]


def test_cost_awareness_prefers_two_cheap_over_one_star():
    m = _model(star=0.9, cheapA=0.7, cheapB=0.7)
    costs = {"star": 10.0, "cheapA": 1.0, "cheapB": 1.0}
    sel = select_reviewers(m, costs=costs, target_detection=0.9)
    # cheapA (0.7) then cheapB -> 1 - 0.3*0.3 = 0.91 >= 0.9, total cost 2 < 10
    assert set(sel.seats) == {"cheapA", "cheapB"}
    assert sel.cost == pytest.approx(2.0)
    assert "star" not in sel.seats


def test_budget_caps_the_council():
    m = _model(a=0.8, b=0.8, c=0.8)
    sel = select_reviewers(m, costs={"a": 1, "b": 1, "c": 1},
                           target_detection=0.999, budget=2.0)
    assert len(sel.seats) == 2 and sel.cost <= 2.0


def test_unreachable_target_returns_best_effort():
    m = _model(a=0.5, b=0.5)
    sel = select_reviewers(m, target_detection=0.99)
    assert set(sel.seats) == {"a", "b"}
    assert sel.detection == pytest.approx(0.75)     # 1 - 0.5*0.5


def test_steps_form_an_audit_trail():
    m = _model(strong=0.9, mid=0.6)
    sel = select_reviewers(m, target_detection=0.95)
    assert [s["seat"] for s in sel.steps] == sel.seats
    assert sel.steps[0]["detection"] == pytest.approx(0.9)


# ---- conditioning + ranking --------------------------------------------------

def test_for_condition_filters_by_condition_keys():
    js = [
        {"seat": "a", "verdict": "reject", "label": "defect", "condition": {"risk": "high"}},
        {"seat": "a", "verdict": "approve", "label": "defect", "condition": {"risk": "low"}},
    ]
    high = for_condition(js, risk="high")
    assert len(high) == 1 and high[0]["condition"]["risk"] == "high"


def test_recall_per_dollar_ranks_by_sensitivity_over_cost():
    m = _model(a=0.9, b=0.6)
    ranked = recall_per_dollar(m, costs={"a": 3.0, "b": 1.0})
    # b: 0.6/1 = 0.6 beats a: 0.9/3 = 0.3
    assert ranked[0][0] == "b"
