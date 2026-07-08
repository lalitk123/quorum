"""Deterministic tests for the canary monitor — the scheduling cadence and the
verify-the-verifier alert logic. No API: the council review itself is stubbed.
"""

import pytest

from agent_team.canary import CanaryMonitor, CanaryResult, evaluate_canary
from agent_team.models import Verdict


def _verdicts(**seats):
    return [Verdict(name, approve, "why") for name, approve in seats.items()]


# ---- Alert logic -------------------------------------------------------------

def test_gate_approving_known_bad_is_critical():
    v = _verdicts(architect=True, security=True, qa=True, product=True, adversary=True)
    r = evaluate_canary("sql-injection", gate_approved=True, verdicts=v, min_dissent=2)
    assert r.critical is True
    assert any("CRITICAL" in a and "sql-injection" in a for a in r.alerts)
    assert r.approving_seats == ["adversary", "architect", "product", "qa", "security"]


def test_healthy_rejection_raises_no_alert():
    # gate rejected, plenty of dissent -> nothing to report.
    v = _verdicts(architect=False, security=False, qa=False, product=True, adversary=False)
    r = evaluate_canary("sql-injection", gate_approved=False, verdicts=v, min_dissent=2)
    assert r.critical is False
    assert not any("WARNING" in a for a in r.alerts)
    # one seat was fooled; that is still recorded for routing.
    assert r.approving_seats == ["product"]


def test_dissent_collapse_warns_even_when_gate_holds():
    # gate technically rejected (unanimous not met) but only one seat dissented.
    v = _verdicts(architect=True, security=False, qa=True, product=True, adversary=True)
    r = evaluate_canary("gamed-test", gate_approved=False, verdicts=v, min_dissent=2)
    assert r.critical is False
    assert any("WARNING" in a and "dissent collapsed" in a for a in r.alerts)


def test_no_dissent_warning_when_gate_already_critical():
    v = _verdicts(architect=True, security=True, qa=True, product=True, adversary=True)
    r = evaluate_canary("x", gate_approved=True, verdicts=v, min_dissent=2)
    # a CRITICAL supersedes the dissent warning; don't double-report.
    assert not any("WARNING" in a for a in r.alerts)


# ---- Scheduling cadence ------------------------------------------------------

@pytest.mark.asyncio
async def test_monitor_fires_every_n_reviews():
    cases = [type("C", (), {"id": "bad1"})(), type("C", (), {"id": "bad2"})()]
    mon = CanaryMonitor(cases, sample_every=3, min_dissent=2, budget_cap=None)
    fired = [await mon._claim_due_case() for _ in range(6)]
    ids = [c.id if c else None for c in fired]
    # only the 3rd and 6th tick fire, rotating through the case list.
    assert ids == [None, None, "bad1", None, None, "bad2"]


@pytest.mark.asyncio
async def test_monitor_disabled_without_cases():
    assert CanaryMonitor([], sample_every=3, min_dissent=2, budget_cap=None).enabled is False
    assert CanaryMonitor([1], sample_every=0, min_dissent=2, budget_cap=None).enabled is False


@pytest.mark.asyncio
async def test_run_if_due_records_and_logs(monkeypatch):
    case = type("C", (), {"id": "sql-injection"})()
    mon = CanaryMonitor([case], sample_every=1, min_dissent=2, budget_cap=None)

    async def fake_review(self, c, cfg, budgets):
        return evaluate_canary(c.id, gate_approved=True,
                               verdicts=_verdicts(security=True, qa=True), min_dissent=2)

    monkeypatch.setattr(CanaryMonitor, "_review", fake_review)
    await mon.run_if_due({}, budgets=())
    assert len(mon.results) == 1 and mon.results[0].critical
