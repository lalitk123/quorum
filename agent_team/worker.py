"""A Worker agent's lifecycle for one ticket:

worktree setup (branching from dependency branches when the ticket has deps)
  -> plan -> council plan review (majority)
  -> implement -> deterministic check gate (tests/linters + tamper scan)
  -> council pre-push review (unanimous)
  -> push branch + open PR (stacked on the dependency branch when applicable)

Everything runs under hard budget caps; a tripped cap escalates fail-closed.
Escalated tickets keep their worktree plus an ESCALATION.md root-cause report.
"""

import logging
from datetime import datetime, timezone

from claude_agent_sdk import ClaudeAgentOptions

from . import council, receipts, sandbox
from .checks import run_checks, tamper_scan
from .models import CouncilDecision, Outcome, Ticket, TicketResult
from .runner import AgentRunError, Budget, BudgetExceeded, extract_json_block, run_agent
from .workspace import GitError, Workspace

log = logging.getLogger("worker")

PLANNER_PROMPT = (
    "You are a software engineer planning a change. Explore the codebase "
    "read-only and produce an implementation plan with these sections: "
    "Approach, Files to modify, Test plan, Risks. Be concrete — name real "
    "files and functions. Do not write any code yet.\n"
    "End your response with exactly one JSON object listing every file you "
    'expect to create or modify:\n{"files": ["path/one.py", "path/two.py"]}'
)

IMPLEMENTER_PROMPT = (
    "You are a software engineer implementing an approved plan in this working "
    "directory. Follow the plan; if reality forces a deviation, keep it minimal "
    "and note it. Run the project's tests (or at minimum the tests you added) "
    "and make them pass before finishing. Never weaken, delete, or skip "
    "existing tests to make them pass — fix the code instead. Do not run any "
    "git commands — committing and pushing are handled outside this session."
)


def _ticket_block(ticket: Ticket) -> str:
    return f"## Ticket {ticket.id}: {ticket.title}\n\n{ticket.description}"


async def _emit_receipt(cfg: dict, ticket: Ticket, ws: Workspace, commit_sha: str,
                        decision: CouncilDecision, action: str, pr_url: str = "") -> None:
    """Best-effort signed audit receipt for a trust-boundary action.

    Never fatal: a signing/IO failure is logged but must not fail an otherwise
    approved ticket. Chaining + append is synchronous so concurrent workers on
    the single event loop can't interleave and fork the hash chain.
    """
    audit = cfg.get("audit") or {}
    if not audit.get("enabled"):
        return
    try:
        key = receipts.load_or_create_key(
            audit.get("signing_key_file", ".agent_team/audit_key.pem"))
        receipt = receipts.Receipt(
            action=action, ticket_id=ticket.id, title=ticket.title,
            branch=ws.branch, base=ws.pr_base, commit_sha=commit_sha,
            timestamp=datetime.now(timezone.utc).isoformat(),
            council_rule=cfg["push_approval"],
            approvals=sum(v.approve for v in decision.verdicts),
            seats=len(decision.verdicts),
            verdicts=[{"reviewer": v.reviewer, "approve": v.approve}
                      for v in decision.verdicts],
            pr_url=pr_url,
        )
        receipts.emit(receipt, key,
                      audit.get("receipts_file", ".agent_team/receipts.jsonl"))
        log.info("[%s] signed audit receipt emitted (%s)", ticket.id, action)
    except Exception as exc:
        log.warning("[%s] could not emit audit receipt: %s", ticket.id, exc)


def _write_escalation(ws: Workspace, ticket: Ticket, reason: str,
                      plan_text: str = "", decision: CouncilDecision | None = None) -> None:
    """Root-cause report for humans, written into the preserved worktree."""
    lines = [f"# Escalation: {ticket.id} — {ticket.title}", "",
             f"**Reason:** {reason}", ""]
    if decision:
        lines.append("## Council verdicts (final round)")
        for v in decision.verdicts:
            mark = "approve" if v.approve else "REJECT"
            lines.append(f"- **{v.reviewer}** ({mark}): {v.reasoning}")
            lines.extend(f"  - {e}" for e in v.evidence)
        lines.append("")
    if plan_text:
        lines += ["## Last plan", plan_text, ""]
    try:
        (ws.path / "ESCALATION.md").write_text("\n".join(lines))
    except OSError:
        log.warning("[%s] could not write ESCALATION.md", ticket.id)


async def run_ticket(ticket: Ticket, cfg: dict, locks,
                     dep_branches: list[str] | None = None,
                     run_budget: Budget | None = None,
                     canary=None) -> TicketResult:
    max_rounds = int(cfg["max_review_rounds"])
    worker_model = cfg["worker_model"]
    council_model = cfg["council_model"]
    council_models = cfg.get("council_models") or {}
    check_commands = cfg.get("checks") or []
    verify_cfg = cfg.get("verify") or {}
    early_exit = bool(cfg.get("early_exit_voting", False))

    budget_cfg = cfg.get("budget") or {}
    ticket_budget = Budget(budget_cfg.get("per_ticket_usd"), f"ticket {ticket.id}")
    budgets = tuple(b for b in (ticket_budget, run_budget) if b)

    # ---- Phase 0: isolated worktree. Dependent tickets start from their
    # dependencies' branches so the code they build on is present.
    ws = Workspace(cfg["repo_path"], cfg["base_branch"], ticket.id,
                   start_refs=dep_branches or None)
    try:
        await ws.create()
    except GitError as exc:
        return TicketResult(ticket, Outcome.BLOCKED,
                            detail=f"could not integrate dependency branches: {exc}")

    plan_text = ""
    try:
        # ---- Phase 1: plan inside the worktree (read-only), until the council
        # approves or we run out of rounds.
        plan_options = ClaudeAgentOptions(
            system_prompt=PLANNER_PROMPT,
            allowed_tools=["Read", "Grep", "Glob"],
            permission_mode="dontAsk",
            cwd=str(ws.path),
            model=worker_model,
            setting_sources=["project"],
        )

        feedback = ""
        plan_rounds = 0
        decision: CouncilDecision | None = None
        for plan_rounds in range(1, max_rounds + 1):
            prompt = _ticket_block(ticket)
            if feedback:
                prompt += (
                    "\n\nYour previous plan was rejected by the review council. "
                    f"Address every objection:\n{feedback}\n\nPrevious plan:\n{plan_text}"
                )
            try:
                plan_text = await run_agent(prompt, plan_options, budgets=budgets)
            except AgentRunError as exc:
                return TicketResult(ticket, Outcome.FAILED, worktree=str(ws.path),
                                    branch=ws.branch, detail=f"planning failed: {exc}")

            log.info("[%s] plan round %d -> council", ticket.id, plan_rounds)
            decision = await council.convene(
                material=(
                    "Review this implementation PLAN (no code exists yet).\n\n"
                    f"{_ticket_block(ticket)}\n\n## Proposed plan\n{plan_text}"
                ),
                cwd=str(ws.path), model=council_model, rule=cfg["plan_approval"],
                models=council_models, budgets=budgets, early_exit=early_exit,
            )
            if decision.approved:
                break
            feedback = decision.feedback()
            log.info("[%s] plan rejected:\n%s", ticket.id, feedback)
        else:
            _write_escalation(ws, ticket, "council never approved the plan",
                              plan_text, decision)
            return TicketResult(ticket, Outcome.ESCALATED_PLAN, worktree=str(ws.path),
                                branch=ws.branch, detail=feedback, plan_rounds=plan_rounds)

        # ---- Claim the files this plan touches so parallel workers don't collide.
        parsed = extract_json_block(plan_text)
        files = [f for f in (parsed or {}).get("files", []) if isinstance(f, str)]
        if not files:
            log.warning("[%s] plan declared no file list; skipping overlap locking", ticket.id)
        await locks.acquire(ticket.id, files)

        try:
            # ---- Phase 2: implement, then pass the deterministic check gate,
            # then loop through pre-push council review until unanimous
            # approval or escalation.
            impl_options = ClaudeAgentOptions(
                system_prompt=IMPLEMENTER_PROMPT,
                allowed_tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob"],
                permission_mode="acceptEdits",
                cwd=str(ws.path),
                model=worker_model,
                setting_sources=["project"],
            )

            impl_prompt = f"{_ticket_block(ticket)}\n\n## Approved plan\n{plan_text}"
            review_rounds = 0
            decision = None
            last_gate_failure = ""
            for review_rounds in range(1, max_rounds + 1):
                try:
                    await run_agent(impl_prompt, impl_options, budgets=budgets)
                except AgentRunError as exc:
                    return TicketResult(ticket, Outcome.FAILED, branch=ws.branch,
                                        worktree=str(ws.path), plan_rounds=plan_rounds,
                                        detail=f"implementation failed: {exc}")

                committed = await ws.commit_all(
                    f"{ticket.id}: {ticket.title} (round {review_rounds})")
                diff = await ws.diff_for_review()
                if not committed and not diff.strip():
                    return TicketResult(ticket, Outcome.FAILED, branch=ws.branch,
                                        worktree=str(ws.path), plan_rounds=plan_rounds,
                                        detail="agent finished without producing changes")

                # Deterministic gate: don't spend reviewer tokens on a diff
                # that fails its own tests or linters.
                check_results = await run_checks(check_commands, str(ws.path))
                failed = [(cmd, out) for cmd, ok, out in check_results if not ok]
                if failed:
                    last_gate_failure = "\n\n".join(
                        f"$ {cmd}\n{out}" for cmd, out in failed)
                    log.info("[%s] deterministic checks failed (round %d)",
                             ticket.id, review_rounds)
                    impl_prompt = (
                        f"{_ticket_block(ticket)}\n\nThe deterministic check gate "
                        "failed — fix the code so every check passes. Do not "
                        "weaken or skip the checks themselves.\n\n"
                        f"{last_gate_failure}"
                    )
                    continue

                tamper_findings = tamper_scan(diff)
                tamper_section = (
                    "\n\n## Automated tamper scan\n"
                    + ("\n".join(f"- {f}" for f in tamper_findings)
                       + "\nTreat any finding above that the diff does not "
                         "convincingly justify as grounds for rejection."
                       if tamper_findings else "No findings.")
                )

                # Clean-sandbox verification: re-run the committed tests in a
                # fresh worktree (catches reliance on local state) plus a teeth
                # check on the new tests. A reproduction failure is a hard gate
                # failure (back to the worker); teeth findings inform the council.
                sandbox_section = ""
                if verify_cfg.get("enabled"):
                    test_command = verify_cfg.get("test_command")
                    if not test_command:
                        log.warning("[%s] verify.enabled but no verify.test_command; "
                                    "skipping sandbox verification", ticket.id)
                    else:
                        report = await sandbox.verify_in_sandbox(
                            cfg["repo_path"], ws.branch, ws.review_base, diff,
                            test_command)
                        if report.hard_failure:
                            last_gate_failure = report.hard_failure
                            log.info("[%s] clean-sandbox verification failed (round %d)",
                                     ticket.id, review_rounds)
                            impl_prompt = (
                                f"{_ticket_block(ticket)}\n\nYour committed change "
                                "failed independent verification. Fix the code so it "
                                "passes from a clean checkout; do not weaken tests.\n\n"
                                f"{report.hard_failure}")
                            continue
                        if report.error:
                            log.warning("[%s] sandbox verification errored: %s",
                                        ticket.id, report.error)
                        if report.findings:
                            sandbox_section = (
                                "\n\n## Clean-sandbox verification\n"
                                + "\n".join(f"- {f}" for f in report.findings)
                                + "\nTreat a teeth-check failure the diff does not "
                                  "convincingly justify as grounds for rejection.")

                log.info("[%s] review round %d -> council", ticket.id, review_rounds)
                decision = await council.convene(
                    material=(
                        "Review this DIFF before it is pushed to GitHub. The working "
                        "directory contains the full changed codebase, and the "
                        "deterministic check gate (tests/linters) already passed.\n\n"
                        f"{_ticket_block(ticket)}\n\n## Approved plan\n{plan_text}"
                        f"{tamper_section}{sandbox_section}"
                        f"\n\n## Diff\n```diff\n{diff}\n```"
                    ),
                    cwd=str(ws.path), model=council_model, rule=cfg["push_approval"],
                    models=council_models, budgets=budgets, early_exit=early_exit,
                )
                # Verify the verifier: periodically slip a known-bad canary into
                # the live stream and alert if the council waves it through.
                # Charged to the run budget only — it is not this ticket's work.
                if canary is not None:
                    await canary.run_if_due(
                        cfg, budgets=tuple(b for b in (run_budget,) if b))
                if decision.approved:
                    break
                impl_prompt = (
                    f"{_ticket_block(ticket)}\n\nThe review council rejected your "
                    f"current changes. Revise the code in place to address every "
                    f"objection:\n{decision.feedback()}"
                )
                log.info("[%s] diff rejected:\n%s", ticket.id, decision.feedback())
            else:
                reason = ("council never approved the diff" if decision
                          else "deterministic checks kept failing")
                detail = decision.feedback() if decision else last_gate_failure[:500]
                _write_escalation(ws, ticket, reason, plan_text, decision)
                return TicketResult(ticket, Outcome.ESCALATED_REVIEW, branch=ws.branch,
                                    worktree=str(ws.path), plan_rounds=plan_rounds,
                                    review_rounds=review_rounds, detail=detail)

            # ---- Phase 3: publish. Every action at the trust boundary — a
            # dry-run approval or a real push/PR — gets a signed audit receipt.
            approved_sha = await ws.head_sha()
            if cfg.get("dry_run", True):
                await _emit_receipt(cfg, ticket, ws, approved_sha, decision, "approved")
                return TicketResult(ticket, Outcome.APPROVED_DRY_RUN, branch=ws.branch,
                                    worktree=str(ws.path), plan_rounds=plan_rounds,
                                    review_rounds=review_rounds,
                                    detail="dry_run enabled — branch ready, not pushed")

            approvals = "\n".join(
                f"- **{v.reviewer}**: {v.reasoning}" for v in decision.verdicts
            )
            pr_url = await ws.push_and_open_pr(
                title=f"{ticket.id}: {ticket.title}",
                body=(f"{ticket.description}\n\n## Plan\n{plan_text}\n\n"
                      f"## Council review\n{approvals}"),
            )
            await _emit_receipt(cfg, ticket, ws, approved_sha, decision, "pr_opened", pr_url)
            await ws.remove(keep_branch=True)
            return TicketResult(ticket, Outcome.PR_OPENED, branch=ws.branch,
                                plan_rounds=plan_rounds, review_rounds=review_rounds,
                                detail=pr_url)
        finally:
            await locks.release(ticket.id)
    except BudgetExceeded as exc:
        log.warning("[%s] %s", ticket.id, exc)
        _write_escalation(ws, ticket, str(exc), plan_text)
        return TicketResult(ticket, Outcome.BUDGET_EXCEEDED, branch=ws.branch,
                            worktree=str(ws.path), detail=str(exc))
