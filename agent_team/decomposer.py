"""Goal decomposition: turn a high-level goal + acceptance criteria into a
ticket backlog, reviewed by the council before any worker starts.

A bad breakdown poisons every downstream ticket, so the council gets a veto
here too (same majority rule and revision loop as per-ticket plans). If the
council never approves, the whole run aborts rather than building on a plan
nobody endorsed.
"""

import logging

from claude_agent_sdk import ClaudeAgentOptions

from . import council
from .models import Ticket
from .runner import extract_json_block, run_agent

log = logging.getLogger("decomposer")

DECOMPOSER_PROMPT = (
    "You are a tech lead breaking a project goal into tickets for a team of "
    "engineers who work in parallel and cannot talk to each other. Explore the "
    "codebase read-only before deciding the breakdown. Rules:\n"
    "- Every ticket must be a self-contained brief: what to build, where it "
    "lives, and concrete acceptance criteria. The engineer sees ONLY the "
    "ticket text, never this goal document.\n"
    "- Minimize dependencies. Prefer independent tickets that own disjoint "
    "files. Use depends_on only when a ticket genuinely builds on another "
    "ticket's code (e.g. scaffolding before features).\n"
    "- No umbrella tickets like 'integrate and test everything'.\n"
    "End your response with exactly one JSON object:\n"
    '{"tickets": [{"id": "T1", "title": "...", "description": "...", '
    '"depends_on": []}]}'
)


class DecompositionRejected(RuntimeError):
    pass


def _parse_tickets(text: str) -> list[Ticket] | str:
    """Returns tickets, or an error string usable as revision feedback."""
    parsed = extract_json_block(text)
    if not parsed or not isinstance(parsed.get("tickets"), list) or not parsed["tickets"]:
        return "Response must end with the required JSON object containing a non-empty 'tickets' list."
    tickets = []
    for raw in parsed["tickets"]:
        try:
            tickets.append(Ticket(
                id=str(raw["id"]), title=str(raw["title"]),
                description=str(raw["description"]),
                depends_on=[str(d) for d in raw.get("depends_on", [])],
            ))
        except (KeyError, TypeError) as exc:
            return f"Every ticket needs id, title, description: {exc}"

    ids = [t.id for t in tickets]
    if len(set(ids)) != len(ids):
        return "Ticket ids must be unique."
    known = set(ids)
    for t in tickets:
        for dep in t.depends_on:
            if dep not in known:
                return f"Ticket {t.id} depends on unknown ticket {dep}."

    # Kahn's algorithm: reject dependency cycles.
    remaining = {t.id: set(t.depends_on) for t in tickets}
    while remaining:
        free = [tid for tid, deps in remaining.items() if not deps]
        if not free:
            return f"Dependency cycle among tickets: {sorted(remaining)}"
        for tid in free:
            del remaining[tid]
        for deps in remaining.values():
            deps.difference_update(free)
    return tickets


async def decompose(cfg: dict, goal: str) -> list[Ticket]:
    options = ClaudeAgentOptions(
        system_prompt=DECOMPOSER_PROMPT,
        allowed_tools=["Read", "Grep", "Glob"],
        permission_mode="dontAsk",
        cwd=cfg["repo_path"],
        model=cfg["worker_model"],
        setting_sources=["project"],
    )

    breakdown = ""
    feedback = ""
    for round_ in range(1, int(cfg["max_review_rounds"]) + 1):
        prompt = f"## Project goal\n\n{goal}"
        if feedback:
            prompt += (
                "\n\nYour previous breakdown was rejected. Address every "
                f"objection:\n{feedback}\n\nPrevious breakdown:\n{breakdown}"
            )
        breakdown = await run_agent(prompt, options)

        tickets = _parse_tickets(breakdown)
        if isinstance(tickets, str):
            feedback = tickets
            log.info("breakdown round %d malformed: %s", round_, feedback)
            continue

        log.info("breakdown round %d: %d ticket(s) -> council", round_, len(tickets))
        decision = await council.convene(
            material=(
                "Review this BREAKDOWN of a project goal into engineering "
                "tickets. Judge whether the tickets fully cover the goal and "
                "its acceptance criteria, are self-contained enough for an "
                "engineer who sees nothing else, and declare correct "
                f"dependencies.\n\n## Goal\n{goal}\n\n## Breakdown\n{breakdown}"
            ),
            cwd=cfg["repo_path"], model=cfg["council_model"],
            rule=cfg["plan_approval"],
        )
        if decision.approved:
            return tickets
        feedback = decision.feedback()
        log.info("breakdown rejected:\n%s", feedback)

    raise DecompositionRejected(
        f"council never approved the goal breakdown; last objections:\n{feedback}"
    )
