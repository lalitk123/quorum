"""The Council: a fixed panel of specialist reviewers.

Each reviewer is an independent, read-only agent session with its own fresh
context — they cannot see each other's opinions, so votes are uncorrelated.
All four run in parallel; the aggregation rule (majority vs unanimous) is
plain code, not model judgment.
"""

import asyncio

from claude_agent_sdk import ClaudeAgentOptions

from .models import CouncilDecision, Verdict
from .runner import AgentRunError, extract_json_block, run_agent

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
}

VERDICT_INSTRUCTIONS = (
    "\n\nYou may read files in the working directory to inform your judgment. "
    "End your response with exactly one JSON object:\n"
    '{"verdict": "approve" | "reject", "reasoning": "<one or two sentences>"}\n'
    "Reject only for concrete, actionable problems — state precisely what must "
    "change. Do not reject for style preferences."
)


async def _review_one(
    name: str, persona: str, material: str, cwd: str, model: str
) -> Verdict:
    options = ClaudeAgentOptions(
        system_prompt=persona + VERDICT_INSTRUCTIONS,
        allowed_tools=["Read", "Grep", "Glob"],
        permission_mode="dontAsk",
        cwd=cwd,
        model=model,
        max_turns=20,
        setting_sources=[],
    )
    try:
        text = await run_agent(material, options)
    except AgentRunError as exc:
        return Verdict(reviewer=name, approve=False, reasoning=f"reviewer errored: {exc}")

    parsed = extract_json_block(text)
    if not parsed or parsed.get("verdict") not in ("approve", "reject"):
        # Fail closed: an unparseable review never waves work through.
        return Verdict(reviewer=name, approve=False,
                       reasoning=f"unparseable verdict: {text[-300:]}")
    return Verdict(
        reviewer=name,
        approve=parsed["verdict"] == "approve",
        reasoning=str(parsed.get("reasoning", "")),
    )


async def convene(material: str, cwd: str, model: str, rule: str) -> CouncilDecision:
    """Run all reviewers in parallel and aggregate under the given rule."""
    verdicts = await asyncio.gather(*(
        _review_one(name, persona, material, cwd, model)
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
