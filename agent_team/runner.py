"""Thin wrapper around the Claude Agent SDK's query() for one-shot agent runs."""

import json
import re

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query


class AgentRunError(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    def __init__(self, label: str, limit: float, spent: float):
        super().__init__(f"{label} budget exhausted: ${spent:.2f} spent of ${limit:.2f} cap")


class Budget:
    """Hard USD ceiling on agent spend, checked before every agent call.

    Fail closed: once tripped, every further call under this budget raises
    instead of running. Parallel calls already in flight can overshoot by one
    round — this is a circuit breaker, not an accountant.
    """

    def __init__(self, limit_usd: float | None, label: str):
        self.limit = limit_usd
        self.label = label
        self.spent = 0.0

    def precheck(self) -> None:
        if self.limit is not None and self.spent >= self.limit:
            raise BudgetExceeded(self.label, self.limit, self.spent)

    def add(self, cost_usd: float | None) -> None:
        self.spent += cost_usd or 0.0


async def run_agent(prompt: str, options: ClaudeAgentOptions,
                    budgets: tuple[Budget, ...] = ()) -> str:
    """Run a single agent session to completion and return its final text."""
    for b in budgets:
        b.precheck()
    result: ResultMessage | None = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            result = message
    if result is None:
        raise AgentRunError("agent session produced no result message")
    for b in budgets:
        b.add(result.total_cost_usd)
    if result.is_error:
        raise AgentRunError(f"agent session failed ({result.subtype}): {result.errors}")
    return result.result or ""


def extract_json_block(text: str) -> dict | None:
    """Pull the last JSON object out of agent output (fenced block or bare)."""
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = fenced or re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
