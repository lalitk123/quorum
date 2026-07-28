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

from . import evidence
from .models import CouncilDecision, Verdict
from .reviewers import SeatSpec, resolve_seat, run_reviewer
from .runner import Budget, BudgetExceeded, extract_json_block

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
        "Judge the verification story: would the tests actually catch a regression "
        "in this change, and is any risky path left unexercised? Reject only when a "
        "specific untested path is one this change could plausibly get wrong — name "
        "the concrete defect that would then ship undetected. A correct change whose "
        "own behavior is covered is an approval: the mere absence of exhaustive or "
        "hypothetical extra tests is not a defect, and \"more tests would be nice\" "
        "is never grounds to reject."
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
    '"evidence": ["path/to/file.py:42 — what you verified or found there"], '
    '"claims": [{"statement": "<a specific claim>", "evidence": ["file.py:42"], '
    '"stance": "supports_reject" | "supports_approve", '
    '"counter": "<the strongest argument against this claim>"}]}\n'
    "The `claims` evidence graph is optional but expected for any non-trivial "
    "finding: state each claim, the file:line it rests on, and the best "
    "counterargument to it. Reject only for concrete, actionable problems — state "
    "precisely what must change. Do not reject for style preferences. An approval "
    "with an empty evidence list is discarded, so always cite what you checked."
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
    except asyncio.CancelledError:
        raise                          # early-exit cancellation is not a seat failure
    except BudgetExceeded:
        raise                          # a tripped cap must abort the run, not vote
    except Exception as exc:
        # Fail closed at the *seat*, not the ticket. Any backend failure — an
        # AgentRunError, an SDK transport/decode error, an unforeseen bug — becomes
        # one rejecting vote with the cause recorded, so the panel still returns a
        # decision and the escalation report explains it. Letting it propagate would
        # abort the whole ticket and lose that report.
        return Verdict(reviewer=name, approve=False,
                       reasoning=f"reviewer errored ({type(exc).__name__}): {exc}")

    parsed = extract_json_block(text)
    if not parsed or parsed.get("verdict") not in ("approve", "reject"):
        # Fail closed: an unparseable review never waves work through.
        return Verdict(reviewer=name, approve=False,
                       reasoning=f"unparseable verdict: {text[-300:]}")

    cited = [str(e) for e in parsed.get("evidence") or [] if str(e).strip()]
    claims = evidence.normalize_claims(parsed.get("claims"))
    approve = parsed["verdict"] == "approve"
    if approve and not cited:
        # Fail closed against optimistic rubber-stamps: approval requires
        # evidence of actual inspection.
        return Verdict(reviewer=name, approve=False,
                       reasoning="approval cited no evidence of inspection (discarded fail-closed)")
    return Verdict(
        reviewer=name,
        approve=approve,
        reasoning=str(parsed.get("reasoning", "")),
        evidence=cited,
        claims=claims,
    )


def _outcome_decided(approvals: int, rejections: int, pending: int,
                     total: int, rule: str) -> bool | None:
    """Is the gate's verdict already forced by the votes counted so far?

    Returns the decided outcome (True/False) when the remaining `pending` seats
    cannot change it, else None. This is *lossless*: it only reports a decision
    the full panel would reach anyway — early exit skips votes that can't matter,
    it never overrides one. When pending == 0 it always resolves.
    """
    if rule == "unanimous":
        if rejections > 0:
            return False                      # one reject already fails unanimity
        if pending == 0:
            return approvals == total
        return None
    if rule == "majority":
        need = total // 2 + 1                  # strict majority (> total/2)
        if approvals >= need:
            return True
        if approvals + pending < need:
            return False                       # majority now unreachable
        return None
    raise ValueError(f"unknown approval rule: {rule}")


async def convene(material: str, cwd: str, model: str, rule: str,
                  models: dict[str, object] | None = None,
                  budgets: tuple[Budget, ...] = (),
                  early_exit: bool = False) -> CouncilDecision:
    """Run all reviewers in parallel and aggregate under the given rule.

    `models` optionally routes individual seats to different backends/models
    for a heterogeneous jury: a bare Claude model string, or a full backend
    spec (e.g. an OpenAI-compatible model). See `reviewers.resolve_seat`.

    With `early_exit`, reviewer sessions still in flight are cancelled the moment
    the outcome is forced (a reject on a unanimous gate, a reached majority) —
    saving their tokens. The decision is identical to the full run; only the set
    of collected verdicts (and thus the feedback detail) may be smaller.
    """
    seat_config = models or {}
    coros = [
        _review_one(name, persona, material, cwd,
                    resolve_seat(seat_config.get(name), model), budgets)
        for name, persona in REVIEWERS.items()
    ]
    total = len(coros)

    if not early_exit:
        verdicts = list(await asyncio.gather(*coros))
        approvals = sum(v.approve for v in verdicts)
        decided = _outcome_decided(approvals, total - approvals, 0, total, rule)
        return CouncilDecision(approved=bool(decided), verdicts=verdicts)

    tasks = [asyncio.create_task(c) for c in coros]
    verdicts, approvals, rejections = [], 0, 0
    approved: bool | None = None
    try:
        for fut in asyncio.as_completed(tasks):
            v = await fut
            verdicts.append(v)
            approvals += v.approve
            rejections += not v.approve
            pending = total - len(verdicts)
            approved = _outcome_decided(approvals, rejections, pending, total, rule)
            if approved is not None:
                break
    finally:
        pending_tasks = [t for t in tasks if not t.done()]
        for t in pending_tasks:
            t.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
    return CouncilDecision(approved=bool(approved), verdicts=verdicts)
