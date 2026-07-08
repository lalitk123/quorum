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
- **Hard budget caps** (per-ticket / per-run USD, from real per-session API
  cost) tripping fail-closed to escalation with preserved worktrees.
- **Per-seat model routing** (`council_models`) — weak-form jury diversity
  within Claude tiers.
- **Escalation reports**: ESCALATION.md root-cause file (rejecting verdicts +
  evidence) written into every preserved worktree.
- Goal → council-approved DAG decomposition; dependency-aware scheduling;
  stacked PRs; file-overlap locks; human-only merge.

## Next: reliability

- **Heterogeneous jury (highest priority).** Pluggable reviewer backends so
  seats can run on disjoint model families (GPT, Gemini, open-weight) — LLM
  errors are 60%+ correlated within a family; disjoint-family panels beat a
  single large judge at ~1/7 the cost (PoLL, arXiv 2404.18796).
- **Calibration gold set.** 30–200 labeled diffs (seeded defects + known-good)
  from the real distribution; measure per-seat precision/recall, κ vs human
  labels, swap-consistency; tune the aggregation thresholds empirically and
  weight seats by measured calibration rather than equally.
- **Canary diffs.** Periodically inject known-bad diffs into the review
  stream; alert when a seat approves one or its dissent rate collapses
  (sycophancy detection — "verify the verifier").
- **Clean-sandbox test verification.** A reviewer step that re-runs the
  worker's tests in a fresh sandbox and checks coverage/mutation, closing the
  remaining reward-hacking surface the static tamper scan can't see.
- **Published council benchmark.** Reproducible seeded-defect corpus +
  true/false approve/reject rates + cost per review, as a public artifact.

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
