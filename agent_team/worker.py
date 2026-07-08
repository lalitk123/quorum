"""A Worker agent's lifecycle for one ticket:

worktree setup (branching from dependency branches when the ticket has deps)
  -> plan -> council plan review (majority)
  -> implement -> council pre-push review (unanimous)
  -> push branch + open PR (stacked on the dependency branch when applicable)
"""

import logging

from claude_agent_sdk import ClaudeAgentOptions

from . import council
from .models import Outcome, Ticket, TicketResult
from .runner import AgentRunError, extract_json_block, run_agent
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
    "and make them pass before finishing. Do not run any git commands — "
    "committing and pushing are handled outside this session."
)


def _ticket_block(ticket: Ticket) -> str:
    return f"## Ticket {ticket.id}: {ticket.title}\n\n{ticket.description}"


async def run_ticket(ticket: Ticket, cfg: dict, locks,
                     dep_branches: list[str] | None = None) -> TicketResult:
    max_rounds = int(cfg["max_review_rounds"])
    worker_model = cfg["worker_model"]
    council_model = cfg["council_model"]

    # ---- Phase 0: isolated worktree. Dependent tickets start from their
    # dependencies' branches so the code they build on is present.
    ws = Workspace(cfg["repo_path"], cfg["base_branch"], ticket.id,
                   start_refs=dep_branches or None)
    try:
        await ws.create()
    except GitError as exc:
        return TicketResult(ticket, Outcome.BLOCKED,
                            detail=f"could not integrate dependency branches: {exc}")

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

    plan_text = ""
    feedback = ""
    plan_rounds = 0
    for plan_rounds in range(1, max_rounds + 1):
        prompt = _ticket_block(ticket)
        if feedback:
            prompt += (
                "\n\nYour previous plan was rejected by the review council. "
                f"Address every objection:\n{feedback}\n\nPrevious plan:\n{plan_text}"
            )
        try:
            plan_text = await run_agent(prompt, plan_options)
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
        )
        if decision.approved:
            break
        feedback = decision.feedback()
        log.info("[%s] plan rejected:\n%s", ticket.id, feedback)
    else:
        return TicketResult(ticket, Outcome.ESCALATED_PLAN, worktree=str(ws.path),
                            branch=ws.branch, detail=feedback, plan_rounds=plan_rounds)

    # ---- Claim the files this plan touches so parallel workers don't collide.
    parsed = extract_json_block(plan_text)
    files = [f for f in (parsed or {}).get("files", []) if isinstance(f, str)]
    if not files:
        log.warning("[%s] plan declared no file list; skipping overlap locking", ticket.id)
    await locks.acquire(ticket.id, files)

    try:
        # ---- Phase 2: implement, then loop through pre-push council review
        # until unanimous approval or escalation.
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
        for review_rounds in range(1, max_rounds + 1):
            try:
                await run_agent(impl_prompt, impl_options)
            except AgentRunError as exc:
                return TicketResult(ticket, Outcome.FAILED, branch=ws.branch,
                                    worktree=str(ws.path), plan_rounds=plan_rounds,
                                    detail=f"implementation failed: {exc}")

            committed = await ws.commit_all(f"{ticket.id}: {ticket.title} (round {review_rounds})")
            diff = await ws.diff_for_review()
            if not committed and not diff.strip():
                return TicketResult(ticket, Outcome.FAILED, branch=ws.branch,
                                    worktree=str(ws.path), plan_rounds=plan_rounds,
                                    detail="agent finished without producing changes")

            log.info("[%s] review round %d -> council", ticket.id, review_rounds)
            decision = await council.convene(
                material=(
                    "Review this DIFF before it is pushed to GitHub. The working "
                    "directory contains the full changed codebase.\n\n"
                    f"{_ticket_block(ticket)}\n\n## Approved plan\n{plan_text}"
                    f"\n\n## Diff\n```diff\n{diff}\n```"
                ),
                cwd=str(ws.path), model=council_model, rule=cfg["push_approval"],
            )
            if decision.approved:
                break
            impl_prompt = (
                f"{_ticket_block(ticket)}\n\nThe review council rejected your "
                f"current changes. Revise the code in place to address every "
                f"objection:\n{decision.feedback()}"
            )
            log.info("[%s] diff rejected:\n%s", ticket.id, decision.feedback())
        else:
            return TicketResult(ticket, Outcome.ESCALATED_REVIEW, branch=ws.branch,
                                worktree=str(ws.path), plan_rounds=plan_rounds,
                                review_rounds=review_rounds,
                                detail=decision.feedback())

        # ---- Phase 3: publish.
        if cfg.get("dry_run", True):
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
        await ws.remove(keep_branch=True)
        return TicketResult(ticket, Outcome.PR_OPENED, branch=ws.branch,
                            plan_rounds=plan_rounds, review_rounds=review_rounds,
                            detail=pr_url)
    finally:
        await locks.release(ticket.id)
