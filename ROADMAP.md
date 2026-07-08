# Roadmap

Grounded in 2026 multi-agent research (heterogeneous juries beat single-model
councils; LLM judges alone catch a minority of code errors, hybrids roughly
double coverage; token spend explains ~80% of multi-agent performance
variance). Positioning: the differentiator is not "agents that write code" —
it is a **trustworthy, auditable, fail-closed review gate** with published
benchmarks of its own quality.

## Done

- Fail-closed council with five seats (architect, security, QA, product,
  **adversary/red-team**), parallel independent sessions, code-side vote
  aggregation.
- **Evidence-grounded verdicts**: every reviewer cites file:line evidence and
  runs a self-falsification pass; an approval with no evidence is discarded
  (guards against optimistic/echo verdicts).
- **Deterministic check gate** before the diff council: configurable
  test/linter commands must pass, plus a code-level tamper scan (CI/harness
  edits, deleted assertions, new skip markers) feeding the reviewers.
- **Clean-sandbox test verification** (`agent_team/sandbox.py`). After the
  checks pass, the worker's *committed* tests are re-run in a fresh git worktree
  at the branch HEAD — no untracked files, no worker residue — so a suite that
  passed only on the worker's dirty local state fails closed back to the worker.
  A lightweight mutation-style **teeth check** then reverts the change's source
  (keeping its new tests) and re-runs: if the suite still passes, the new tests
  never exercise the new behavior (vacuous/gamed), surfaced to the council as an
  advisory finding. Closes the reward-hacking surface the static tamper scan
  can't see. Deterministic, no API cost; covered by real-git integration tests.
- **Hard budget caps** (per-ticket / per-run USD, from real per-session API
  cost) tripping fail-closed to escalation with preserved worktrees.
- **Heterogeneous jury.** Pluggable reviewer backends (`agent_team/reviewers.py`):
  a `claude` backend with live file tools (default) and an `openai` backend for
  any OpenAI-compatible endpoint — GPT / Gemini / DeepSeek / Qwen directly or
  via OpenRouter, no new dependencies. Seats routed per-config decorrelate
  errors across model families (PoLL, arXiv 2404.18796). Cross-family spend
  folds into the USD budget when per-Mtok prices are declared.
- **Escalation reports**: ESCALATION.md root-cause file (rejecting verdicts +
  evidence) written into every preserved worktree.
- **Unit test suite** (`tests/`, `pytest -q`) covering the deterministic
  fail-closed logic: budget breaker, tamper scan, verdict parsing, evidence
  fail-closed rule, seat routing, decomposer DAG validation, bench metrics.
- **Council benchmark harness** (`agent_team/bench.py`, `python main.py --bench`):
  runs the council against a labeled gold set (`bench/gold/`) and reports the
  confusion matrix (defect recall, false-reject rate), per-seat detection
  rates, and cost per review. Ships with example good/bad cases (clean change,
  SQL injection, gamed test). The reproducible, publishable trust artifact.
- Goal → council-approved DAG decomposition; dependency-aware scheduling;
  stacked PRs; file-overlap locks; human-only merge.

## Next: reliability

- **Grow the gold set + tune from it.** The harness exists; now grow the corpus
  toward 30–200 labeled diffs from the real distribution, add swap-consistency
  and κ-vs-human measures, and use the per-seat numbers to weight/re-route
  seats and set aggregation thresholds empirically rather than by intuition.
- **Canary diffs.** Periodically inject known-bad diffs into the *live* review
  stream; alert when a seat approves one or its dissent rate collapses
  (sycophancy detection — "verify the verifier"). Reuses the bad gold cases.
- **Deeper mutation coverage.** The teeth check ships today as a single
  source-revert; extend toward per-file mutation and coverage deltas so a
  partially-tested change is scored, not just pass/fail.

## Next: scale

- Early-exit voting (stop at majority for plan gate; first hard reject for
  unanimous gate) once calibration shows the consensus score discriminates.
- Structured context objects between phases instead of full plan/diff text
  where possible; targeted retrieval for reviewers on large repos.
- Persist decomposer plan + per-ticket state to external memory for very
  large runs; scale reviewer depth with diff size (light pass on small diffs).
- Tune worker concurrency to human merge throughput (practical ceiling is
  review, not compute: ~4–8 concurrent worktrees).

## Next: autonomy

- **Graduated autonomy tiers**: notify-only → suggest → auto-merge low-risk
  classes (docs, lockfile bumps) after clean-track-record promotion; auth/
  migrations/security stay human-merge forever.
- **Daemon mode**: long-lived service watching the backlog, maintaining the
  DAG, dispatching as dependencies clear, governed by merge throughput.
- Bounded self-repair on escalation (auto-decompose repeatedly-failing
  tickets) — partial today via revision rounds + ESCALATION.md.
- Self-improvement of reviewer prompts/decomposer heuristics gated on the
  gold-set regression eval only (spec + regression oracle, never a raw metric).
- Signed (Ed25519) audit receipts for push/PR actions for regulated contexts.

## Product

- MIT license, demo-first README, BYO-key model-agnostic council.
- Integrations: Linear/Jira ingestion, Graphite/gh-stack stacked-PR tooling;
  run alongside GitHub Agent HQ / Anthropic Code Review rather than against.
- Open-core path: hosted operation, audit/compliance plane, org policy
  controls, dashboards.
