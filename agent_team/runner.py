"""Thin wrapper around the Claude Agent SDK's query() for one-shot agent runs."""

import json
import re

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query


class AgentRunError(RuntimeError):
    pass


async def run_agent(prompt: str, options: ClaudeAgentOptions) -> str:
    """Run a single agent session to completion and return its final text."""
    result: ResultMessage | None = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            result = message
    if result is None:
        raise AgentRunError("agent session produced no result message")
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
