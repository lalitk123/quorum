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


# ---- Baseline-relative (seat-specific) regression alerts ---------------------

from agent_team.canary import baseline_catch_rates, seat_catch_counts   # noqa: E402


def _dj(seat, case, verdict):
    return {"seat": seat, "case_id": case, "verdict": verdict, "label": "defect"}


def test_seat_catch_counts_are_case_scoped():
    rows = [_dj("security", "sql", "reject"), _dj("security", "sql", "reject"),
            _dj("security", "xss", "approve"), _dj("qa", "sql", "approve")]
    assert seat_catch_counts(rows, "sql") == {"security": (2, 2), "qa": (0, 1)}
    assert seat_catch_counts(rows)["security"] == (2, 3)   # pooled over both cases


def test_baseline_prefers_case_history_then_falls_back_to_overall():
    rows = ([_dj("security", "sql", "reject")] * 4        # strong on this case
            + [_dj("qa", "other", "reject")] * 4)         # no 'sql' history, strong overall
    base = baseline_catch_rates(rows, "sql", min_n=3)
    assert base["security"] == pytest.approx(1.0)          # case-specific
    assert base["qa"] == pytest.approx(1.0)                # fell back to overall
    # A seat with too little history either way gets no baseline (no regression claim).
    assert "newbie" not in baseline_catch_rates(
        rows + [_dj("newbie", "sql", "approve")], "sql", min_n=3)


def test_a_reliable_seat_that_now_misses_is_flagged_a_regression():
    v = _verdicts(architect=False, security=True, qa=False, product=False, adversary=False)
    # security caught this canary 100% before but approved it now.
    r = evaluate_canary("sql-injection", gate_approved=False, verdicts=v, min_dissent=2,
                        baselines={"security": 1.0}, baseline_floor=0.75)
    assert r.regressed_seats == ["security"]
    assert any("REGRESSION" in a and "security" in a for a in r.alerts)
    # The panel still rejected with plenty of dissent, so absent the baseline there
    # would have been no warning at all.
    assert not any("WARNING" in a for a in r.alerts)


def test_a_seat_below_its_baseline_floor_is_not_flagged():
    v = _verdicts(architect=False, security=True, qa=False, product=False, adversary=False)
    # security only caught this 50% historically -> approving now is within noise.
    r = evaluate_canary("sql-injection", gate_approved=False, verdicts=v, min_dissent=2,
                        baselines={"security": 0.5}, baseline_floor=0.75)
    assert r.regressed_seats == []
    assert not any("REGRESSION" in a for a in r.alerts)


def test_monitor_computes_baselines_from_the_log_before_appending(tmp_path):
    import asyncio
    from agent_team import bench as bench_mod

    log = str(tmp_path / "j.jsonl")
    from agent_team import judgments
    # History: security reliably caught 'sql', then it approves it on this review.
    judgments.append_judgments(log, [_dj("security", "sql", "reject") for _ in range(4)])

    case = bench_mod.GoldCase(id="sql", title="t", description="d",
                              expected_approve=False, diff="", note="", workspace_src=None)

    class _R:
        error = ""
        decision_approved = False
        cost_usd = 0.0
        verdicts = _verdicts(architect=False, security=True, qa=False,
                             product=False, adversary=False)

    async def fake_run_case(c, cfg, cap):
        return _R()

    mon = CanaryMonitor([case], sample_every=1, min_dissent=2, budget_cap=None,
                        judgment_log=log, baseline_floor=0.75, baseline_min_n=3)
    orig = bench_mod._run_case
    bench_mod._run_case = fake_run_case
    try:
        result = asyncio.run(mon._review(case, {}, ()))
    finally:
        bench_mod._run_case = orig
    assert result.regressed_seats == ["security"]
