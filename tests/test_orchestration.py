"""End-to-end tests for the orchestration seam against real (throwaway) git repos.

Fully deterministic — no API keys, no model calls. The two model-driven steps
(`worker.run_agent`, which drafts the plan and writes the implementation, and
`council.convene`, which returns the panel verdict) are stubbed; everything else
runs for real: worktree creation, commits, the deterministic check gate, risk
classification, trust scoring, the dry-run receipt path, file-overlap locks, and
dependency gating. This covers the control flow that no other test exercises.
"""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent_team import council, orchestrator, worker
from agent_team.models import CouncilDecision, Outcome, Ticket, TicketResult, Verdict
from agent_team.orchestrator import FileLocks

SEATS = list(council.REVIEWERS)


def test_escalation_marks_partial_panel_and_names_unpolled_seats(tmp_path):
    """With early-exit voting, an escalation's panel is partial. The report must
    label it as partial and name the seats that never voted — a human triaging it
    must not read a cancelled seat as a seat that raised no objection."""
    ws = SimpleNamespace(path=tmp_path)
    ticket = Ticket("T1", "add feature", "desc")
    decision = CouncilDecision(approved=False, verdicts=[
        Verdict("security", approve=False, reasoning="sqli", evidence=["app.py:1"])])
    worker._write_escalation(ws, ticket, "diff rejected", "the plan", decision)

    text = (tmp_path / "ESCALATION.md").read_text()
    assert f"1 of {len(council.REVIEWERS)} seats polled" in text
    assert "not polled" in text
    for seat in council.REVIEWERS:
        if seat != "security":
            assert seat in text            # every unpolled seat is named


def test_escalation_full_panel_has_no_partial_note(tmp_path):
    """A complete panel keeps the plain 'final round' heading and no partial note."""
    ws = SimpleNamespace(path=tmp_path)
    ticket = Ticket("T1", "add feature", "desc")
    decision = CouncilDecision(approved=False, verdicts=[
        Verdict(s, approve=(s != "adversary"), reasoning="ok", evidence=["app.py:1"])
        for s in council.REVIEWERS])
    worker._write_escalation(ws, ticket, "diff rejected", "the plan", decision)

    text = (tmp_path / "ESCALATION.md").read_text()
    assert "final round" in text
    assert "not polled" not in text


def _git(repo, *args):
    proc = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _repo(tmp_path):
    """A minimal git repo on branch `master` with one committed file."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "master")
    _git(repo, "config", "user.email", "t@t.test")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "app.py").write_text("VALUE = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _cfg(repo, **over):
    cfg = {
        "repo_path": str(repo),
        "base_branch": "master",
        "max_review_rounds": 2,
        "worker_model": "sonnet",
        "council_model": "sonnet",
        "council_models": {},
        "plan_approval": "majority",
        "push_approval": "unanimous",
        "dry_run": True,
        "early_exit_voting": False,
        "checks": [],
        "verify": {},
        "autonomy": {"enabled": True},
        "budget": {},
        "reliability": {},
    }
    cfg.update(over)
    return cfg


def _approve(seats=SEATS):
    return CouncilDecision(
        approved=True,
        verdicts=[Verdict(s, True, "looks correct", ["app.py:1"]) for s in seats],
    )


def _reject(seats=SEATS):
    return CouncilDecision(
        approved=False,
        verdicts=[Verdict(s, False, "unconvinced", ["app.py:1"]) for s in seats],
    )


def _install_agent_stub(monkeypatch, plan_files=("feature.py",)):
    """Stub the two model-driven calls in the worker.

    The planner returns a plan ending in the required JSON file list; the
    implementer actually writes a source file into the worktree so a real diff
    exists for the gate and the council to see.
    """
    async def fake_run_agent(prompt, options, budgets=()):
        import json
        from pathlib import Path
        if "Write" in (options.allowed_tools or []):        # implementer session
            (Path(options.cwd) / "feature.py").write_text("def feature():\n    return 42\n")
            return "implemented"
        files_json = json.dumps({"files": list(plan_files)})  # planner session
        return f"Approach: add a feature.\nFiles to modify: {', '.join(plan_files)}\n{files_json}"

    monkeypatch.setattr(worker, "run_agent", fake_run_agent)


@pytest.mark.asyncio
async def test_dry_run_approval_happy_path(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    _install_agent_stub(monkeypatch)

    async def fake_convene(material, **kw):
        return _approve()
    monkeypatch.setattr(council, "convene", fake_convene)

    ticket = Ticket("T1", "Add a feature", "Add a small feature to app.py")
    result = await worker.run_ticket(ticket, _cfg(repo), FileLocks())

    assert result.outcome is Outcome.APPROVED_DRY_RUN
    assert result.branch == "agent/T1"
    assert result.trust_score is not None
    assert result.disposition                      # human-merge recommendation present
    # dry_run keeps the worktree on disk for inspection.
    assert (repo / ".worktrees" / "T1").exists()
    # The implementer's file was committed on the ticket branch.
    assert "feature.py" in _git(repo / ".worktrees" / "T1", "ls-files")


@pytest.mark.asyncio
async def test_plan_rejection_escalates(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    _install_agent_stub(monkeypatch)

    async def always_reject(material, **kw):
        return _reject()
    monkeypatch.setattr(council, "convene", always_reject)

    ticket = Ticket("T1", "Add a feature", "Add a small feature to app.py")
    result = await worker.run_ticket(ticket, _cfg(repo), FileLocks())

    assert result.outcome is Outcome.ESCALATED_PLAN
    # The worktree is kept with a root-cause report for a human.
    assert (repo / ".worktrees" / "T1" / "ESCALATION.md").exists()


@pytest.mark.asyncio
async def test_diff_rejection_escalates_after_plan_approval(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    _install_agent_stub(monkeypatch)

    async def approve_plan_reject_diff(material, **kw):
        # The plan review material announces itself; the diff review does too.
        return _reject() if "Review this DIFF" in material else _approve()
    monkeypatch.setattr(council, "convene", approve_plan_reject_diff)

    ticket = Ticket("T1", "Add a feature", "Add a small feature to app.py")
    result = await worker.run_ticket(ticket, _cfg(repo), FileLocks())

    assert result.outcome is Outcome.ESCALATED_REVIEW
    assert result.review_rounds == _cfg(repo)["max_review_rounds"]
    assert (repo / ".worktrees" / "T1" / "ESCALATION.md").exists()


@pytest.mark.asyncio
async def test_orchestrator_blocks_dependents_of_a_failed_ticket(monkeypatch):
    """A failed dependency blocks its dependent instead of building on air."""
    cfg = {
        "backlog": {"source": "file", "file": "unused"},
        "max_concurrent_workers": 2,
        "budget": {},
    }
    tickets = [
        Ticket("T1", "root", "the dependency"),
        Ticket("T2", "leaf", "depends on T1", depends_on=["T1"]),
    ]
    monkeypatch.setattr(orchestrator, "_load_tickets",
                        lambda cfg, run_budget: _async(tickets))
    monkeypatch.setattr(orchestrator.canary_mod, "build_monitor", lambda cfg: None)

    async def fake_run_ticket(ticket, *a, **k):
        # T1 fails; T2 should never run (it is blocked before taking a slot).
        assert ticket.id != "T2", "dependent ran despite a failed dependency"
        return TicketResult(ticket, Outcome.FAILED, detail="boom")
    monkeypatch.setattr(orchestrator, "run_ticket", fake_run_ticket)

    results = {r.ticket.id: r for r in await orchestrator.run(cfg)}
    assert results["T1"].outcome is Outcome.FAILED
    assert results["T2"].outcome is Outcome.BLOCKED


async def _async(value):
    return value
