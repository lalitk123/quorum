"""Evidence-graph tests (idea #4): normalization is fail-closed, rendering is clean."""

from agent_team.evidence import normalize_claims, render_graph


def test_normalize_keeps_valid_claims():
    raw = [{"statement": "SQL is string-formatted", "evidence": ["search.py:12"],
            "stance": "supports_reject", "counter": "the input is an int elsewhere"}]
    claims = normalize_claims(raw)
    assert len(claims) == 1
    c = claims[0]
    assert c["statement"] == "SQL is string-formatted"
    assert c["evidence"] == ["search.py:12"]
    assert c["stance"] == "supports_reject" and c["counter"]


def test_normalize_drops_claims_without_a_statement():
    raw = [{"evidence": ["a.py:1"]}, {"statement": "   "}, "not a dict", 42]
    assert normalize_claims(raw) == []


def test_normalize_defaults_bad_stance_to_neutral():
    claims = normalize_claims([{"statement": "x", "stance": "vibes"}])
    assert claims[0]["stance"] == "neutral"


def test_normalize_non_list_is_empty():
    assert normalize_claims(None) == []
    assert normalize_claims({"statement": "x"}) == []


def test_render_graph_includes_evidence_and_counter():
    claims = normalize_claims([{"statement": "auth bypass", "evidence": ["auth.py:9"],
                                "stance": "supports_reject", "counter": "gated upstream"}])
    md = render_graph(claims, indent="  ")
    assert "auth bypass" in md
    assert "evidence: auth.py:9" in md
    assert "counterargument: gated upstream" in md
    assert "reject" in md


def test_render_empty_is_blank():
    assert render_graph([]) == ""
