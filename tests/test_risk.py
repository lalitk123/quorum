"""Risk classifier + graduated-policy tests. Deterministic, no API.

Verifies the blast-radius classification, its HIGH-biased precedence, and that
the policy only ever tightens the gate and always keeps merge human.
"""

from agent_team.risk import (
    RiskAssessment, RiskLevel, classify, effective_push_rule, merge_disposition,
)


def _diff(path, added="+    x = 1", new=True):
    header = (f"diff --git a/{path} b/{path}\n"
              + ("new file mode 100644\n" if new else f"--- a/{path}\n+++ b/{path}\n"))
    return header + f"@@ -0,0 +1 @@\n{added}\n"


# ---- Classification by path --------------------------------------------------

def test_docs_only_is_low():
    assert classify(_diff("README.md", "+docs")).level == RiskLevel.LOW
    assert classify(_diff("docs/guide.rst", "+text")).level == RiskLevel.LOW


def test_lockfile_is_low():
    assert classify(_diff("poetry.lock", "+dep")).level == RiskLevel.LOW
    assert classify(_diff("package-lock.json", "+x")).level == RiskLevel.LOW


def test_ordinary_source_is_medium():
    assert classify(_diff("src/widgets.py")).level == RiskLevel.MEDIUM


def test_auth_path_is_high():
    a = classify(_diff("src/auth/session.py"))
    assert a.level == RiskLevel.HIGH
    assert any("auth" in r for r in a.reasons)


def test_migration_and_sql_are_high():
    assert classify(_diff("db/migrations/0007_add.py")).level == RiskLevel.HIGH
    assert classify(_diff("schema/users.sql")).level == RiskLevel.HIGH


def test_ci_and_infra_are_high():
    assert classify(_diff(".github/workflows/ci.yml", "+run: true")).level == RiskLevel.HIGH
    assert classify(_diff("infra/main.tf", "+resource")).level == RiskLevel.HIGH


# ---- Precedence (HIGH-biased) ------------------------------------------------

def test_high_wins_over_low_when_mixed():
    diff = _diff("README.md", "+docs") + _diff("src/oauth.py", "+token")
    assert classify(diff).level == RiskLevel.HIGH


def test_source_plus_docs_is_medium_not_low():
    diff = _diff("README.md", "+docs") + _diff("src/util.py", "+def f(): ...")
    assert classify(diff).level == RiskLevel.MEDIUM


# ---- Content-based HIGH ------------------------------------------------------

def test_dangerous_content_forces_high_even_in_ordinary_path():
    diff = _diff("src/util.py", "+    data = pickle.loads(blob)")
    a = classify(diff)
    assert a.level == RiskLevel.HIGH
    assert any("pickle.loads" in r for r in a.reasons)


def test_verify_false_and_eval_flagged():
    assert classify(_diff("src/client.py", "+    requests.get(u, verify=False)")).level == RiskLevel.HIGH
    assert classify(_diff("src/run.py", "+    eval(user_input)")).level == RiskLevel.HIGH


def test_removed_dangerous_line_does_not_trigger():
    # a '-' line removing eval() should not be read as introducing it.
    diff = ("diff --git a/src/run.py b/src/run.py\n--- a/src/run.py\n+++ b/src/run.py\n"
            "@@ -1 +1 @@\n-    eval(x)\n+    safe(x)\n")
    assert classify(diff).level == RiskLevel.MEDIUM


def test_unparseable_diff_is_medium_never_low():
    assert classify("").level == RiskLevel.MEDIUM


# ---- Policy: tighten only, merge always human --------------------------------

def test_high_forces_unanimous_even_when_configured_majority():
    assert effective_push_rule(RiskLevel.HIGH, "majority") == "unanimous"


def test_low_and_medium_keep_configured_rule_never_weakened():
    assert effective_push_rule(RiskLevel.LOW, "unanimous") == "unanimous"
    assert effective_push_rule(RiskLevel.MEDIUM, "unanimous") == "unanimous"
    assert effective_push_rule(RiskLevel.LOW, "majority") == "majority"


def test_disposition_is_always_human():
    for level in RiskLevel:
        assert "human" in merge_disposition(level).lower()
    assert "never auto-eligible" in merge_disposition(RiskLevel.HIGH)
