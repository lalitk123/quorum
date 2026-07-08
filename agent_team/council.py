"""The Council: a fixed panel of specialist reviewers plus a red-team seat.

Each reviewer is an independent, read-only agent session with its own fresh
context — they cannot see each other's opinions, so votes are uncorrelated.
All seats run in parallel; the aggregation rule (majority vs unanimous) is
plain code, not model judgment.

Verdicts must be grounded: every reviewer cites the files/lines it actually
inspected and runs a self-falsification pass before finalizing (per Anthropic
Code Review's verification step, which held incorrect findings under 1%).
An approval with no cited evidence is treated as a rejection — fail closed
against "optimistic verdicts" that approve without inspecting anything.
"""

import asyncio

from .models import CouncilDecision, Verdict
from .reviewers import SeatSpec, resolve_seat, run_reviewer
from .runner import AgentRunError, Budget, extract_json_block

REVIEWERS: dict[str, str] = {
    "architect": (
        "You are the architecture reviewer on an engineering review council. "
        "Judge soundness of approach, fit with the existing codebase, scope creep, "
        "and over-engineering. A simple correct solution beats a clever one."
    ),
    "security": (
        "You are the security reviewer on an engineering review council. "
        "Judge the security implications: injection, authz/authn gaps, secrets "
        "handling, unsafe deserialization, and newly exposed surface area."
    ),
    "qa": (
        "You are the QA reviewer on an engineering review council. "
        "Judge the verification story: are the right cases tested, are edge cases "
        "covered, would the tests actually catch a regression here?"
    ),
    "product": (
        "You are the product/scope reviewer on an engineering review council. "
        "Judge whether the work actually solves the stated ticket — no more, no "
        "less. Flag missing acceptance criteria and unrequested extras."
    ),
    "adversary": (
        "You are the red-team reviewer on an engineering review council. Your "
        "sole job is to break this change: construct a concrete input, call "
        "sequence, or scenario under which it fails, can be abused, or games "
        "its own verification — weakened or deleted tests, trivially-passing "
        "assertions, hidden logic, and edits to CI or the test harness are all "
        "findings. If you cannot construct a plausible break after genuinely "
        "trying, approve."
    ),
}

_EVIDENCE_WITH_TOOLS = (
    "You may read files in the working directory to inform your judgment. "
    "Ground every claim in evidence: cite the specific files (with line numbers "
    "where possible) you actually inspected."
)
_EVIDENCE_NO_TOOLS = (
    "You do not have file access — judge from the diff and plan provided above. "
    "Ground every claim in evidence: cite file:line references drawn from the "
    "diff for anything you flag or verify."
)
_VERDICT_TAIL = (
    " Before finalizing, attempt to DISPROVE your own conclusion — re-read the "
    "cited code and drop any finding that does not survive scrutiny; report only "
    "findings that do.\n"
    "End your response with exactly one JSON object:\n"
    '{"verdict": "approve" | "reject", "reasoning": "<one or two sentences>", '
    '"evidence": ["path/to/file.py:42 — what you verified or found there"]}\n'
    "Reject only for concrete, actionable problems — state precisely what must "
    "change. Do not reject for style preferences. An approval with an empty "
    "evidence list is discarded, so always cite what you checked."
)


def _verdict_instructions(has_file_tools: bool) -> str:
    lead = _EVIDENCE_WITH_TOOLS if has_file_tools else _EVIDENCE_NO_TOOLS
    return "\n\n" + lead + _VERDICT_TAIL


async def _review_one(
    name: str, persona: str, material: str, cwd: str, spec: SeatSpec,
    budgets: tuple[Budget, ...],
) -> Verdict:
    system_prompt = persona + _verdict_instructions(spec.has_file_tools)
    try:
        text = await run_reviewer(spec, system_prompt, material, cwd, budgets)
    except AgentRunError as exc:
        return Verdict(reviewer=name, approve=False, reasoning=f"reviewer errored: {exc}")

    parsed = extract_json_block(text)
    if not parsed or parsed.get("verdict") not in ("approve", "reject"):
        # Fail closed: an unparseable review never waves work through.
        return Verdict(reviewer=name, approve=False,
                       reasoning=f"unparseable verdict: {text[-300:]}")

    evidence = [str(e) for e in parsed.get("evidence") or [] if str(e).strip()]
    approve = parsed["verdict"] == "approve"
    if approve and not evidence:
        # Fail closed against optimistic rubber-stamps: approval requires
        # evidence of actual inspection.
        return Verdict(reviewer=name, approve=False,
                       reasoning="approval cited no evidence of inspection (discarded fail-closed)")
    return Verdict(
        reviewer=name,
        approve=approve,
        reasoning=str(parsed.get("reasoning", "")),
        evidence=evidence,
    )


async def convene(material: str, cwd: str, model: str, rule: str,
                  models: dict[str, object] | None = None,
                  budgets: tuple[Budget, ...] = ()) -> CouncilDecision:
    """Run all reviewers in parallel and aggregate under the given rule.

    `models` optionally routes individual seats to different backends/models
    for a heterogeneous jury: a bare Claude model string, or a full backend
    spec (e.g. an OpenAI-compatible model). See `reviewers.resolve_seat`.
    """
    seat_config = models or {}
    verdicts = await asyncio.gather(*(
        _review_one(name, persona, material, cwd,
                    resolve_seat(seat_config.get(name), model), budgets)
        for name, persona in REVIEWERS.items()
    ))
    approvals = sum(v.approve for v in verdicts)
    if rule == "unanimous":
        approved = approvals == len(verdicts)
    elif rule == "majority":
        approved = approvals > len(verdicts) / 2
    else:
        raise ValueError(f"unknown approval rule: {rule}")
    return CouncilDecision(approved=approved, verdicts=list(verdicts))
