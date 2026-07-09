from dataclasses import dataclass, field
from enum import Enum


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
