"""Pluggable reviewer backends for a heterogeneous council jury.

LLM errors are 60%+ correlated *within* a model family, so a council of five
identical Claude seats rubber-stamps far more than the five-seat design
implies. Routing seats to disjoint model families decorrelates their errors —
the single most evidence-backed reliability gain (PoLL, arXiv 2404.18796).

Two backends ship today:

- `claude`  — the Agent SDK session with live file tools (Read/Grep/Glob), so
  the reviewer can browse the working tree. This is the default for every seat.
- `openai`  — any OpenAI-compatible chat endpoint (OpenAI, DeepSeek, or GPT /
  Gemini / Qwen / Llama via OpenRouter). No live file access: the reviewer
  judges from the materialized diff + plan already embedded in the prompt, and
  cites evidence as file:line references drawn from that diff. One backend
  unlocks the whole cross-family jury via an OpenRouter base_url.

Cost accounting is honest across the split: the Claude backend reports real
per-session USD; the openai backend contributes USD to the budget only when
the seat spec declares per-Mtok prices (otherwise 0, but the pre-call budget
check still fires so an already-over-budget run won't spawn more reviewers).
"""

import asyncio
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from claude_agent_sdk import ClaudeAgentOptions

from .runner import AgentRunError, Budget, run_agent

HTTP_TIMEOUT_SECONDS = 180
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclass
class SeatSpec:
    backend: str                 # "claude" | "openai"
    model: str
    has_file_tools: bool
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float | None = None
    input_price_per_mtok: float | None = None
    output_price_per_mtok: float | None = None


def resolve_seat(spec: object, default_model: str) -> SeatSpec:
    """Normalize a per-seat config value into a SeatSpec.

    Accepts: None (Claude, default model), a bare model string (Claude), or a
    dict with a `backend` discriminator for richer routing.
    """
    if spec is None:
        return SeatSpec("claude", default_model, has_file_tools=True)
    if isinstance(spec, str):
        return SeatSpec("claude", spec, has_file_tools=True)
    if isinstance(spec, dict):
        backend = spec.get("backend", "claude")
        if backend == "claude":
            return SeatSpec("claude", spec.get("model", default_model),
                            has_file_tools=True)
        if backend in ("openai", "openai_compatible"):
            if "model" not in spec:
                raise ValueError("openai reviewer seat requires a 'model'")
            return SeatSpec(
                backend="openai",
                model=spec["model"],
                has_file_tools=False,
                base_url=spec.get("base_url", DEFAULT_OPENAI_BASE_URL),
                api_key_env=spec.get("api_key_env", "OPENAI_API_KEY"),
                temperature=spec.get("temperature"),
                input_price_per_mtok=spec.get("input_price_per_mtok"),
                output_price_per_mtok=spec.get("output_price_per_mtok"),
            )
        raise ValueError(f"unknown reviewer backend: {backend}")
    raise ValueError(f"invalid seat spec: {spec!r}")


def _seat_cost(spec: SeatSpec, usage: dict) -> float | None:
    if spec.input_price_per_mtok is None or spec.output_price_per_mtok is None:
        return None
    prompt = usage.get("prompt_tokens", 0) or 0
    completion = usage.get("completion_tokens", 0) or 0
    return (prompt / 1e6) * spec.input_price_per_mtok + \
           (completion / 1e6) * spec.output_price_per_mtok


def _post_chat_completion(spec: SeatSpec, system: str, user: str) -> tuple[str, dict]:
    api_key = os.environ.get(spec.api_key_env or "")
    if not api_key:
        raise AgentRunError(
            f"env var {spec.api_key_env} is unset — cannot reach {spec.model}")
    payload: dict = {
        "model": spec.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if spec.temperature is not None:
        payload["temperature"] = spec.temperature
    req = urllib.request.Request(
        (spec.base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise AgentRunError(f"{spec.model} HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise AgentRunError(f"{spec.model} unreachable: {exc}") from exc
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AgentRunError(f"{spec.model} returned no message: {data}") from exc
    return text or "", data.get("usage") or {}


async def run_reviewer(spec: SeatSpec, system_prompt: str, material: str,
                       cwd: str, budgets: tuple[Budget, ...]) -> str:
    """Dispatch one reviewer to its backend and return its raw text output."""
    if spec.backend == "claude":
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            allowed_tools=["Read", "Grep", "Glob"],
            permission_mode="dontAsk",
            cwd=cwd,
            model=spec.model,
            max_turns=20,
            setting_sources=[],
        )
        return await run_agent(material, options, budgets=budgets)

    # openai-compatible: no file tools, budget-checked around the HTTP call.
    for b in budgets:
        b.precheck()

    def _account(usage: dict) -> None:
        cost = _seat_cost(spec, usage)
        for b in budgets:
            b.add(cost)

    # Run the HTTP call as its own shielded task. If this reviewer is cancelled
    # mid-flight (early-exit voting forces the gate before this seat votes), the
    # request is already in flight and OpenAI bills it regardless — a running
    # thread is not cancellable. So we wait for it and record the cost before
    # letting the cancellation propagate; otherwise the per-ticket/per-run USD
    # caps silently under-count real OpenAI spend, which is exactly what they
    # exist to bound.
    call = asyncio.ensure_future(
        asyncio.to_thread(_post_chat_completion, spec, system_prompt, material))
    try:
        text, usage = await asyncio.shield(call)
    except asyncio.CancelledError:
        try:
            _, usage = await call
            _account(usage)
        finally:
            raise
    _account(usage)
    return text
