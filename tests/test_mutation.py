"""Mutation generator tests. Deterministic; the council evaluation is not tested
here (it needs the API) but the labeled diffs it consumes are."""

from agent_team.mutation import Mutant, generate_mutants, mutation_score


def _ops(mutants):
    return {m.operator for m in mutants}


def test_force_true_on_if_condition():
    muts = generate_mutants("    if authenticated:\n        grant()\n", "auth.py")
    ft = [m for m in muts if m.operator == "force-true"]
    assert ft and ft[0].mutated_line.strip() == "if True:"
    assert "if authenticated:" in ft[0].diff and "+    if True:" in ft[0].diff


def test_comparison_and_boolean_swaps():
    muts = generate_mutants("    return a == b and c\n", "m.py")
    ops = _ops(muts)
    assert "eq->ne" in ops and "and->or" in ops
    ne = next(m for m in muts if m.operator == "eq->ne")
    assert "a != b" in ne.mutated_line


def test_drop_not_removes_guard_negation():
    muts = generate_mutants("        if not is_admin:\n", "p.py")
    dn = [m for m in muts if m.operator == "drop-not"]
    assert dn and dn[0].mutated_line.strip() == "if is_admin:"


def test_boolean_literal_swap():
    muts = generate_mutants("    allowed = True\n", "c.py")
    tf = [m for m in muts if m.operator == "true->false"]
    assert tf and "allowed = False" in tf[0].mutated_line


def test_spaced_swaps_do_not_corrupt_shift_or_arrow():
    # ' < ' requires surrounding spaces, so '<<' and '->' are left alone.
    muts = generate_mutants("    x = a << 2\n    y: int = 0\n", "s.py")
    assert all("<<" not in m.mutated_line or m.mutated_line == m.original_line
               for m in muts)


def test_each_mutant_is_single_site_and_diffs_are_git_style():
    muts = generate_mutants("    return x == y\n", "m.py")
    for m in muts:
        assert m.diff.startswith("diff --git a/m.py b/m.py\n")
        assert m.mutated_line != m.original_line


def test_max_mutants_cap():
    src = "\n".join("    a == b and c or d" for _ in range(50))
    assert len(generate_mutants(src, "big.py", max_mutants=10)) == 10


def test_no_mutants_from_plain_prose():
    assert generate_mutants("the quick brown fox\n", "notes.txt") == []


# ---- Mutation score math -----------------------------------------------------

class _R:
    def __init__(self, approved, error=""):
        self.decision_approved = approved
        self.error = error


def test_mutation_score_counts_kills():
    # killed = rejected (fault caught); survived = approved.
    results = [_R(False), _R(False), _R(True), _R(False, error="boom")]
    s = mutation_score(results)
    assert s["n"] == 3 and s["killed"] == 2 and s["survived"] == 1
    assert s["mutation_score"] == 2 / 3
