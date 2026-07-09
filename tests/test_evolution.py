"""Reviewer-evolution tests (idea #6). Deterministic model-vs-model comparison."""

from agent_team.evolution import recommend_evolution


def _rows(seat, model, n_defect, catches, n_clean, true_approves):
    """n_defect defect cases (catches of them rejected); n_clean clean cases."""
    rows = []
    for i in range(n_defect):
        rows.append({"seat": seat, "model": model, "label": "defect",
                     "verdict": "reject" if i < catches else "approve"})
    for i in range(n_clean):
        rows.append({"seat": seat, "model": model, "label": "clean",
                     "verdict": "approve" if i < true_approves else "reject"})
    return rows


def test_recommends_swap_when_challenger_clearly_better():
    rows = []
    # incumbent (more data) catches only half the defects; challenger catches all.
    rows += _rows("security", "old", n_defect=20, catches=10, n_clean=20, true_approves=18)
    rows += _rows("security", "new", n_defect=15, catches=15, n_clean=15, true_approves=14)
    ev = recommend_evolution(rows, min_n=10, margin=0.1)["security"]
    assert ev.incumbent == "old"
    assert ev.swap is True and ev.recommended == "new"


def test_no_swap_without_enough_evidence():
    rows = []
    rows += _rows("qa", "old", n_defect=20, catches=10, n_clean=20, true_approves=18)
    rows += _rows("qa", "new", n_defect=3, catches=3, n_clean=3, true_approves=3)   # tiny
    ev = recommend_evolution(rows, min_n=10, margin=0.1)["qa"]
    assert ev.swap is False
    assert "insufficient evidence" in ev.reason or ev.recommended == "old"


def test_no_swap_when_incumbent_already_best():
    rows = []
    rows += _rows("arch", "old", n_defect=20, catches=19, n_clean=20, true_approves=19)
    rows += _rows("arch", "new", n_defect=15, catches=8, n_clean=15, true_approves=8)
    ev = recommend_evolution(rows, min_n=10, margin=0.1)["arch"]
    assert ev.swap is False and ev.recommended == "old"


def test_marginal_lead_is_held_not_swapped():
    rows = []
    rows += _rows("product", "old", n_defect=20, catches=14, n_clean=20, true_approves=16)
    rows += _rows("product", "new", n_defect=15, catches=11, n_clean=15, true_approves=12)
    ev = recommend_evolution(rows, min_n=10, margin=0.25)["product"]  # demand a big margin
    assert ev.swap is False


def test_untagged_judgments_produce_no_recommendation():
    rows = [{"seat": "x", "label": "defect", "verdict": "reject"}]   # no model field
    assert recommend_evolution(rows) == {}
