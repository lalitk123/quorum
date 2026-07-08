"""Unit tests for the deterministic, no-API-key logic of the system.

These are exactly the pieces that must never regress silently: the fail-closed
gates, budget circuit breaker, tamper scan, verdict parsing, seat routing, and
the decomposer's dependency validation. Run with: pytest -q
"""

import pytest

import asyncio

from agent_team.checks import classify_changed_files, tamper_scan
from agent_team import council
from agent_team.council import _outcome_decided, _review_one
from agent_team.decomposer import _parse_tickets
from agent_team.models import CouncilDecision, Verdict
from agent_team.reviewers import SeatSpec, _seat_cost, resolve_seat
from agent_team.runner import Budget, BudgetExceeded, extract_json_block


# ---- Budget circuit breaker --------------------------------------------------

def test_budget_trips_fail_closed():
    b = Budget(1.0, "run")
    b.add(0.6)
    b.precheck()          # under cap, fine
    b.add(0.6)            # now $1.20
    with pytest.raises(BudgetExceeded):
        b.precheck()


def test_budget_none_never_trips():
    b = Budget(None, "unlimited")
    b.add(10_000)
    b.precheck()          # no cap -> no raise


# ---- Tamper scan -------------------------------------------------------------

def test_tamper_scan_flags_gaming_surfaces():
    diff = (
        "diff --git a/tests/test_auth.py b/tests/test_auth.py\n"
        "--- a/tests/test_auth.py\n+++ b/tests/test_auth.py\n"
        "-    assert user.is_admin is False\n"
        "+    pass\n"
        "+@pytest.mark.skip(reason=\"flaky\")\n"
        "diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml\n"
        "-      run: pytest\n+      run: true\n"
    )
    findings = tamper_scan(diff)
    joined = "\n".join(findings)
    assert "removes an assertion" in joined
    assert "skip/xfail" in joined
    assert "CI/harness" in joined


def test_tamper_scan_ignores_non_test_assertions():
    diff = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n+++ b/src/app.py\n"
        "-    assert cfg is not None\n+    cfg = load()\n"
    )
    assert tamper_scan(diff) == []


# ---- Diff file classification (clean-sandbox verifier) -----------------------

def test_classify_changed_files_flags_new_deleted_and_tests():
    diff = (
        "diff --git a/src/feature.py b/src/feature.py\n"
        "new file mode 100644\n+def f(): return 1\n"
        "diff --git a/src/old.py b/src/old.py\n"
        "deleted file mode 100644\n-def g(): ...\n"
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/tests/test_app.py b/tests/test_app.py\n"
        "new file mode 100644\n+def test_it(): assert True\n"
    )
    by_path = {c.path: c for c in classify_changed_files(diff)}
    assert by_path["src/feature.py"].is_new and not by_path["src/feature.py"].is_test
    assert by_path["src/old.py"].is_deleted
    assert not by_path["src/app.py"].is_new and not by_path["src/app.py"].is_deleted
    assert by_path["tests/test_app.py"].is_test and by_path["tests/test_app.py"].is_new


# ---- Verdict JSON extraction -------------------------------------------------

def test_extract_json_takes_last_object():
    text = 'thinking...\n{"verdict":"reject"}\nfinal:\n{"verdict":"approve","evidence":["a.py:1"]}'
    assert extract_json_block(text)["verdict"] == "approve"


def test_extract_json_handles_fenced_block():
    text = "```json\n{\"verdict\": \"approve\", \"evidence\": [\"x.py:9\"]}\n```"
    assert extract_json_block(text)["evidence"] == ["x.py:9"]


# ---- Fail-closed reviewer verdicts ------------------------------------------

class _StubSpec(SeatSpec):
    pass


async def _run_review(monkeypatch, returned_text):
    async def fake_run_reviewer(spec, system_prompt, material, cwd, budgets):
        return returned_text
    monkeypatch.setattr("agent_team.council.run_reviewer", fake_run_reviewer)
    spec = SeatSpec("claude", "sonnet", has_file_tools=True)
    return await _review_one("qa", "persona", "material", "/tmp", spec, ())


@pytest.mark.asyncio
async def test_approve_without_evidence_becomes_reject(monkeypatch):
    v = await _run_review(monkeypatch, '{"verdict":"approve","evidence":[]}')
    assert v.approve is False
    assert "no evidence" in v.reasoning


@pytest.mark.asyncio
async def test_unparseable_verdict_becomes_reject(monkeypatch):
    v = await _run_review(monkeypatch, "I think it's probably fine, ship it")
    assert v.approve is False


@pytest.mark.asyncio
async def test_approve_with_evidence_passes(monkeypatch):
    v = await _run_review(
        monkeypatch,
        '{"verdict":"approve","reasoning":"ok","evidence":["app.py:3 checked bounds"]}',
    )
    assert v.approve is True
    assert v.evidence == ["app.py:3 checked bounds"]


# ---- Aggregation -------------------------------------------------------------

def _decision(*approvals):
    verdicts = [Verdict(f"r{i}", ok, "why") for i, ok in enumerate(approvals)]
    return verdicts


def test_feedback_only_includes_rejections_with_evidence():
    d = CouncilDecision(approved=False, verdicts=[
        Verdict("architect", True, "fine"),
        Verdict("security", False, "unsafe deserialize", evidence=["p.py:9 pickle.loads"]),
    ])
    fb = d.feedback()
    assert "architect" not in fb
    assert "security" in fb and "p.py:9" in fb


# ---- Seat routing (heterogeneous jury) --------------------------------------

def test_resolve_seat_defaults_to_claude():
    s = resolve_seat(None, "sonnet")
    assert s.backend == "claude" and s.model == "sonnet" and s.has_file_tools


def test_resolve_seat_bare_string_is_claude_model():
    s = resolve_seat("opus", "sonnet")
    assert s.backend == "claude" and s.model == "opus"


def test_resolve_seat_openai_spec():
    s = resolve_seat(
        {"backend": "openai", "model": "gpt-5.1", "api_key_env": "OPENROUTER_API_KEY",
         "base_url": "https://openrouter.ai/api/v1", "input_price_per_mtok": 2.5,
         "output_price_per_mtok": 10.0},
        "sonnet",
    )
    assert s.backend == "openai" and s.has_file_tools is False
    assert s.api_key_env == "OPENROUTER_API_KEY"


def test_resolve_seat_openai_requires_model():
    with pytest.raises(ValueError):
        resolve_seat({"backend": "openai"}, "sonnet")


def test_resolve_seat_rejects_unknown_backend():
    with pytest.raises(ValueError):
        resolve_seat({"backend": "telepathy", "model": "x"}, "sonnet")


def test_seat_cost_none_without_prices():
    spec = SeatSpec("openai", "m", has_file_tools=False)
    assert _seat_cost(spec, {"prompt_tokens": 1000, "completion_tokens": 1000}) is None


def test_seat_cost_computes_from_prices():
    spec = SeatSpec("openai", "m", has_file_tools=False,
                    input_price_per_mtok=2.0, output_price_per_mtok=10.0)
    cost = _seat_cost(spec, {"prompt_tokens": 1_000_000, "completion_tokens": 500_000})
    assert cost == pytest.approx(2.0 + 5.0)


# ---- Decomposer dependency validation ---------------------------------------

def test_parse_tickets_rejects_cycles():
    text = ('{"tickets":['
            '{"id":"A","title":"a","description":"d","depends_on":["B"]},'
            '{"id":"B","title":"b","description":"d","depends_on":["A"]}]}')
    result = _parse_tickets(text)
    assert isinstance(result, str) and "cycle" in result.lower()


def test_parse_tickets_rejects_unknown_dependency():
    text = ('{"tickets":['
            '{"id":"A","title":"a","description":"d","depends_on":["Z"]}]}')
    result = _parse_tickets(text)
    assert isinstance(result, str) and "unknown" in result.lower()


def test_parse_tickets_accepts_valid_dag():
    text = ('{"tickets":['
            '{"id":"A","title":"a","description":"d","depends_on":[]},'
            '{"id":"B","title":"b","description":"d","depends_on":["A"]}]}')
    tickets = _parse_tickets(text)
    assert not isinstance(tickets, str)
    assert [t.id for t in tickets] == ["A", "B"]
    assert tickets[1].depends_on == ["A"]


# ---- Early-exit voting (lossless) -------------------------------------------

def test_outcome_decided_unanimous_short_circuits_on_reject():
    # one reject with seats still pending already forces a reject.
    assert _outcome_decided(approvals=2, rejections=1, pending=2, total=5,
                            rule="unanimous") is False
    # all-approve only resolves once nothing is pending.
    assert _outcome_decided(4, 0, 1, 5, "unanimous") is None
    assert _outcome_decided(5, 0, 0, 5, "unanimous") is True


def test_outcome_decided_majority_resolves_both_ways_early():
    # 3/5 approvals is a strict majority regardless of the last two votes.
    assert _outcome_decided(3, 0, 2, 5, "majority") is True
    # 3 rejections make a 5-seat majority unreachable (max 2 approvals left).
    assert _outcome_decided(0, 3, 2, 5, "majority") is False
    # 2 approvals, 1 reject, 2 pending: still open.
    assert _outcome_decided(2, 1, 2, 5, "majority") is None


def test_outcome_decided_rejects_unknown_rule():
    with pytest.raises(ValueError):
        _outcome_decided(1, 0, 0, 1, "vibes")


@pytest.mark.asyncio
async def test_convene_early_exit_cancels_pending_seats(monkeypatch):
    """A fast reject on a unanimous gate must cancel the slow seats' sessions."""
    completed = []

    async def fake_review_one(name, persona, material, cwd, spec, budgets):
        # 'security' rejects immediately; everyone else is slow.
        if name == "security":
            completed.append(name)
            return Verdict(name, approve=False, reasoning="sqli", evidence=["a:1"])
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            raise
        completed.append(name)
        return Verdict(name, approve=True, reasoning="ok", evidence=["a:1"])

    monkeypatch.setattr(council, "_review_one", fake_review_one)
    decision = await council.convene("m", "/tmp", "sonnet", "unanimous",
                                     early_exit=True)
    assert decision.approved is False
    # the slow seats were cancelled, not awaited to completion.
    assert completed == ["security"]
    assert [v.reviewer for v in decision.verdicts] == ["security"]
