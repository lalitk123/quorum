"""Evidence graphs (idea #4): make a verdict explainable, not just asserted.

A reviewer that writes "looks fine" is unfalsifiable. Instead each seat may back
its verdict with a small graph of **claims**, and each claim carries the file:line
evidence it rests on, the stance it supports (reject / approve), and — the part
that forces real reasoning — the strongest *counterargument* to itself. An
approval then isn't a rubber stamp; it's a structure a human (or another agent)
can walk: this claim, on this evidence, with this acknowledged weakness.

This module is the deterministic schema layer: normalize whatever the model
emitted into validated claims (dropping junk fail-closed), and render them as
Markdown for the escalation report and PR notes. The richer prompt lives in the
council; the parsing/validation/rendering here is unit-tested.
"""

from dataclasses import dataclass, field

_STANCES = {"supports_reject", "supports_approve", "neutral"}


@dataclass
class Claim:
    statement: str
    evidence: list[str] = field(default_factory=list)
    stance: str = "neutral"
    counter: str = ""


def normalize_claims(raw) -> list[dict]:
    """Coerce model output into validated claim dicts; drop anything malformed.

    Fail-closed on shape: a claim with no statement is discarded rather than
    guessed at. Returns plain dicts so verdicts stay JSON-serializable.
    """
    if not isinstance(raw, list):
        return []
    claims: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        statement = str(item.get("statement", "")).strip()
        if not statement:
            continue
        evidence = [str(e).strip() for e in (item.get("evidence") or [])
                    if str(e).strip()]
        stance = str(item.get("stance", "neutral")).strip()
        if stance not in _STANCES:
            stance = "neutral"
        claims.append(Claim(
            statement=statement, evidence=evidence, stance=stance,
            counter=str(item.get("counter", "")).strip(),
        ).__dict__)
    return claims


def render_graph(claims: list[dict], indent: str = "") -> str:
    """Render claims as a Markdown evidence graph."""
    if not claims:
        return ""
    arrow = {"supports_reject": "⛔ reject", "supports_approve": "✅ approve",
             "neutral": "• neutral"}
    lines = []
    for c in claims:
        stance = arrow.get(c.get("stance", "neutral"), "• neutral")
        lines.append(f"{indent}- [{stance}] {c.get('statement', '')}")
        for e in c.get("evidence") or []:
            lines.append(f"{indent}  - evidence: {e}")
        if c.get("counter"):
            lines.append(f"{indent}  - counterargument: {c['counter']}")
    return "\n".join(lines)
