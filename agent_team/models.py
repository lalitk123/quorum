import re
from dataclasses import dataclass, field
from enum import Enum

# A ticket id becomes a filesystem path (`<repo>/.worktrees/<id>`) and a git ref
# (`agent/<id>`), so it is an injection sink, not a label. Ids reach it from a
# model's decomposition output and from tasks.yaml — neither is trustworthy
# enough to interpolate into a path unchecked. Today a traversal attempt happens
# to die at `git worktree add`, because git's ref rules reject `..`, `//`, and
# leading/trailing slashes; that is git defending us by accident, not a decision
# this codebase made. An allowlist makes it a decision.
TICKET_ID_PATTERN = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def invalid_ticket_id(ticket_id: str) -> str:
    """Return a human/model-readable reason the id is unusable, else ""."""
    if not ticket_id:
        return "ticket id is empty"
    if not TICKET_ID_PATTERN.match(ticket_id):
        return (f"ticket id {ticket_id!r} is not allowed — use only letters, digits, "
                "dot, dash, or underscore (max 64 chars, starting alphanumeric). "
                "It becomes a directory name and a git branch name.")
    if ticket_id in (".", "..") or ticket_id.startswith("."):
        return f"ticket id {ticket_id!r} is not allowed — it must not start with a dot."
    return ""


@dataclass
class Ticket:
    id: str
    title: str
    description: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    reviewer: str
    approve: bool
    reasoning: str
    evidence: list[str] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)   # evidence graph (idea #4)


@dataclass
class CouncilDecision:
    approved: bool
    verdicts: list[Verdict]

    def feedback(self) -> str:
        """Consolidated objections from rejecting reviewers, fed back to the worker."""
        lines = []
        for v in self.verdicts:
            if v.approve:
                continue
            lines.append(f"[{v.reviewer}] {v.reasoning}")
            lines.extend(f"  evidence: {e}" for e in v.evidence)
        return "\n".join(lines)


class Outcome(str, Enum):
    PR_OPENED = "pr_opened"
    APPROVED_DRY_RUN = "approved_dry_run"
    ESCALATED_PLAN = "escalated_plan"        # council never approved the plan
    ESCALATED_REVIEW = "escalated_review"    # council never approved the diff
    BLOCKED = "blocked"                      # dependency failed or couldn't be integrated
    BUDGET_EXCEEDED = "budget_exceeded"      # hard cost ceiling tripped (fail closed)
    FAILED = "failed"                        # agent error / no changes produced


@dataclass
class TicketResult:
    ticket: Ticket
    outcome: Outcome
    detail: str = ""
    branch: str | None = None
    worktree: str | None = None
    plan_rounds: int = 0
    review_rounds: int = 0
    risk_level: str = ""
    disposition: str = ""      # human-facing merge recommendation (merge stays human)
    trust_score: float | None = None
    trust_grade: str = ""
