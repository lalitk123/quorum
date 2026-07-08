# multi-agent-threading

An AI engineering team built on the **Claude Agent SDK**: parallel worker
agents each take a ticket, and every worker must clear a **council of four
specialist reviewer agents** twice — once to approve its plan, and again to
approve its diff before anything is pushed to GitHub.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## How a ticket flows

```
goal.md ──> decomposer agent ──> council review of the breakdown ──┐
(optional: or hand-written tasks.yaml / GitHub issues) <───────────┘
                │
                ▼
backlog ──> worker: draft plan (read-only)
                │
                ▼
        council plan review  ── majority required ──┐ reject: revise (≤ N rounds)
                │ approve                           │
                ▼                                   │
        implement in isolated git worktree  <───────┘
                │
                ▼
        council pre-push review ── unanimous required ──┐ reject: fix (≤ N rounds)
                │ approve                               │
                ▼                                       │
        push branch + open PR  <────────────────────────┘
        (never straight to main; merge stays human)
```

The council — architect, security, QA, product/scope, and a red-team
adversary — runs as five **parallel, independent, read-only agent sessions**;
each cites file:line evidence, runs a self-falsification pass, and returns a
JSON verdict aggregated in plain code. Unparseable reviews, errored reviews,
and approvals with no cited evidence all count as rejections (fail closed).

Seats can run on **different model families** for a heterogeneous jury (the
highest-leverage reliability lever): each seat is a bare Claude model or an
OpenAI-compatible backend spec (GPT / Gemini / DeepSeek / Qwen, directly or
via OpenRouter). See `council_models` in `config.yaml`.

A **deterministic gate** runs before the diff council: configured test/linter
commands must pass in the worktree, and a tamper scan flags gamed tests
(deleted assertions, skip markers, CI edits). Hard per-ticket and per-run USD
budget caps trip fail-closed to escalation.

## Setup

```bash
# Requires Python 3.10+ (the Agent SDK does not support 3.9;
# on macOS: brew install python@3.12)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Prerequisites:

- **Anthropic auth** — `ANTHROPIC_API_KEY` in the environment, or an existing
  Claude Code login (the Agent SDK picks up either).
- **`gh` CLI authenticated** (`gh auth login`) — only needed when
  `dry_run: false` or `backlog.source: github`.
- A **target repo**: any local git clone with an `origin` remote. Point
  `repo_path` in `config.yaml` at it.

## Run

Autonomous goal mode is the default: write your goal + acceptance criteria in
`goal.md`, point `repo_path` at the target clone, and run —

```bash
python main.py --config config.yaml
```

To feed hand-written tickets or GitHub issues instead, set `backlog.source`
to `file` or `github` in `config.yaml`.

Run the unit tests for the deterministic fail-closed logic:

```bash
pytest -q
```

`dry_run: true` (the default) stops after council approval: the branch and
worktree exist locally under `<repo>/.worktrees/`, nothing is pushed. Flip it
to `false` to let approved workers push and open PRs.

Escalated tickets (council never approved within `max_review_rounds`) keep
their worktree on disk so a human can inspect what the agent and the council
disagreed about.

## Layout

## Fully autonomous mode

Set `backlog.source: goal` and write your goal + acceptance criteria in
`goal.md`. A decomposer agent explores the target repo and breaks the goal
into self-contained tickets with dependencies; the council must approve the
breakdown (majority) before any worker starts — if it never does, the run
aborts without touching the repo. Tickets with `depends_on` wait for their
dependencies, branch from the dependency's branch (so the code they build on
is present), and their PRs are stacked on that branch. A failed or escalated
dependency marks its dependents `blocked` instead of building on air.

`tasks.yaml` tickets may also declare `depends_on: [T1]` by hand.

| File | Role |
|---|---|
| `main.py` | CLI entry point |
| `agent_team/decomposer.py` | Goal → council-approved ticket backlog (goal mode) |
| `agent_team/orchestrator.py` | Parallel scheduling, file-overlap locks, summary |
| `agent_team/worker.py` | Per-ticket lifecycle: plan → review → implement → review → PR |
| `agent_team/council.py` | The five reviewer personas + vote aggregation |
| `agent_team/reviewers.py` | Pluggable reviewer backends (Claude + OpenAI-compatible) |
| `agent_team/checks.py` | Deterministic test/linter gate + diff tamper scan |
| `agent_team/workspace.py` | Git worktrees, commits, diffs, push/PR (deterministic code) |
| `agent_team/backlog.py` | tasks.yaml or GitHub Issues backlog |
| `agent_team/runner.py` | Thin `query()` wrapper + JSON extraction |
