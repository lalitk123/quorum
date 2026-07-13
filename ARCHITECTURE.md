# Quorum — Architecture Plan

## Goal

Replace a human engineering team with AI agents that work as independent
"engineers," running in parallel, each of which must clear a review gate with
a fixed panel of specialist reviewer agents ("the Council") twice per task:
once to approve the plan, once to approve the diff before it goes to GitHub.

## Actors

### Orchestrator (the "EM")

- Holds the backlog of work items (tickets/issues).
- Assigns one work item to one Worker Agent at a time; runs many Workers
  concurrently.
- Tracks each Worker's state machine (below) and detects file/module overlap
  between concurrently running Workers so two agents don't clobber the same
  area of the codebase.
- Is the escalation point when a Worker and the Council can't reach
  consensus after N revision rounds — hands the decision to the human user
  rather than looping forever.

### Worker Agents (the "engineers")

One per ticket, run in parallel, each in an isolated workspace (its own
branch/worktree). Lifecycle per ticket:

1. **Draft plan** — read the ticket, produce an implementation plan.
2. **Council plan review** — submit the plan; revise until the Council
   approves or the iteration cap is hit.
3. **Implement** — write the code in isolation.
4. **Verify** — run tests/build/lint against its own changes.
5. **Council pre-push review** — submit the diff (plus test results); revise
   until approved or the cap is hit.
6. **Push + open PR** — never push straight to main; opens a PR and reports
   status to the Orchestrator.

### The Council (fixed panel of specialist reviewers)

A constant roster of reviewer personas, each independently evaluating the
plan or diff from its own angle:

- **Architect** — soundness of approach, scope creep, alignment with the
  rest of the system, over-engineering.
- **Security** — security implications of the plan/diff.
- **QA / Test** — test coverage and verification strategy.
- **Product / Scope** — does this actually solve the stated ticket, is the
  scope right (not more, not less).
- **Adversary / Red-team** — actively tries to falsify the change: hunts for
  the failure mode the other seats missed and the way the tests could be gamed.

Each member returns approve/reject + reasoning. Two different consensus bars
are used depending on the checkpoint (see below) — cheap-and-fast for plan
review, strict for pre-push review, since a bad plan is cheap to redo but a
bad push is not.

## The Two Checkpoints

| Checkpoint | When | Bar | Purpose |
|---|---|---|---|
| Plan review | Before any code is written | Majority approve | Catch design flaws while they're still cheap to fix |
| Pre-push review | Before opening a PR | Unanimous approve | Last line of defense before the change becomes visible/shared |

## Parallelism & Isolation

- Each Worker gets its own branch/worktree — no shared mutable workspace.
- The Orchestrator serializes anything touching shared state (backlog
  board, merge to main).
- If two Workers' target files overlap, the Orchestrator either serializes
  them (second one waits) or flags it for a merge-conflict-aware review
  before either pushes.

## Failure / Escalation Loop

- Council rejection → Worker revises and resubmits.
- After N rounds (e.g. 3) without consensus at either checkpoint, the
  Orchestrator escalates to the human user instead of looping indefinitely.

## GitHub Interaction Policy

- Workers push feature branches and open PRs; they do not push to `main`.
- The pre-push Council review doubles as the PR's review notes/description.
- Final merge authority stays with the human user — pushing/opening PRs is
  visible-to-others territory, auto-merge is out of scope for v1.

## Decisions (v1, as built)

- **Substrate:** Claude Agent SDK (Python, `claude-agent-sdk`). Workers and
  council members are independent `query()` sessions; `asyncio` provides the
  parallelism.
- **Council implementation:** five parallel top-level agent sessions (not
  nested subagents) — each reviewer gets a fresh, isolated context and
  read-only tools, so votes are uncorrelated; aggregation (majority vs
  unanimous) is deterministic code. Unparseable/errored reviews, and approvals
  citing no file:line evidence, fail closed as rejections. Seats may run on
  different model families (a heterogeneous jury) to decorrelate errors.
- **Backlog:** three sources — `tasks.yaml`, open GitHub Issues via `gh`, or
  **goal mode**: a decomposer agent turns a high-level goal + acceptance
  criteria (`goal.md`) into a ticket backlog with dependencies, which the
  council must approve before any worker starts. Dependency cycles are
  rejected in code; dependent tickets branch from their dependency's branch
  and stack their PRs on it; a failed dependency blocks its dependents.
- **Consensus rules:** plan review = majority (3/5), pre-push = unanimous
  (5/5); both configurable, and risk-adjusted (a HIGH-risk change is forced to
  unanimous regardless of the configured bar).
- **Iteration cap:** `max_review_rounds: 3` per checkpoint, then escalate;
  escalated worktrees are kept on disk for human inspection.
- **Concurrency:** `max_concurrent_workers: 3` via semaphore; file-overlap
  locks serialize workers whose approved plans touch the same paths.
- **Branch naming:** `agent/<ticket-id>`; worktrees under
  `<repo>/.worktrees/<ticket-id>`.
- **Shared/remote state is code, not agents:** commits, diffs, push, and
  `gh pr create` are deterministic Python; agents only edit files. A
  `dry_run` flag (default on) stops before anything leaves the machine.
