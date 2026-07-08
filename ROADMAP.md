# Quorum — Roadmap

Grounded in 2026 multi-agent research (heterogeneous juries beat single-model
councils; LLM judges alone catch a minority of code errors, hybrids roughly
double coverage; token spend explains ~80% of multi-agent performance
variance). Positioning: the differentiator is not "agents that write code" —
it is a **trustworthy, auditable, fail-closed review gate** with published
benchmarks of its own quality.

## Research program (up a level: from an engineering system to a contribution)

The coding worker is becoming commoditized; the durable, unsolved problem is
*measurable trust in automated review*. Quorum's north star is an **AI software
assurance layer** whose reviewers are themselves measured instruments. Three
research questions organize the work:

- **RQ1 — Can we measure LLM reviewer reliability quantitatively?** Model each
  seat as a noisy sensor; estimate sensitivity/specificity with uncertainty from
  ground-truth-labeled judgments; aggregate with calibrated Bayesian inference
  instead of majority voting. *(Substrate shipped: `reliability.py`,
  `judgments.py` — see Done.)*
- **RQ2 — Can we adaptively pick reviewers to maximize defect recall per dollar?**
  Condition the reliability estimates on the change (bug type, language, risk
  class) and select the seats that buy the most detection per dollar.
- **RQ3 — Can we detect reviewer degradation automatically?** *(Shipped:
  `drift.py` — canary outcomes persist as a per-seat time series; a Bernoulli
  CUSUM raises a self-alarm, with the drift onset index, when a seat starts
  approving known-bad diffs. See Done.)*

The enabler for all three is a **free labeled-data engine** (`mutation.py`): gold
cases, canaries, and mutants are defective *by construction*, so every seat
verdict on them is a labeled reliability datapoint — no human labeling required.
Single-site mutants (`if authenticated:` → `if True:`, `==` → `!=`, ...) generate
labeled-bad diffs in bulk, giving RQ2 the per-cell data volume it needs, and
yield a **mutation score** — the fraction of injected faults the council kills —
as a review-quality metric stronger than "the seats agreed." The honest scope: on
a tiny corpus these estimates have wide intervals; the contribution is the
methodology + infrastructure, with results that sharpen as the log grows.

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
- **Canary diffs — verify the verifier** (`agent_team/canary.py`,
  `python main.py --canary`). Periodically injects a known-bad gold case into
  the *live* review stream and alerts when the gate approves a seeded defect
  (CRITICAL — the blind spot that ships real ones) or when dissent collapses
  below a healthy floor (the leading drift indicator), recording which seats
  were fooled for re-routing. Catches runtime sycophancy/model drift a one-time
  benchmark cannot. Charged to the run budget, never fatal to a real ticket;
  cadence and alert logic are deterministic and unit-tested. Every canary review
  now persists per-seat labeled judgments, feeding both RQ1 and RQ3.
- **Continuous reviewer-drift detection** (RQ3: `agent_team/drift.py`,
  `python main.py --drift`). Each seat's canary outcomes form a Bernoulli miss
  stream; a one-sided Page's CUSUM (log-likelihood ratio between a healthy miss
  rate `p0` and a degraded rate `p1`) raises a self-alarm — with the observation
  index where drift began — when a seat starts approving known-bad canaries. A
  lone miss decays away; sustained degradation trips the threshold. The `--drift`
  command exits non-zero if any seat is drifting (a CI/cron gate on the gate
  itself). Deterministic, unit-tested against hand-computed CUSUM trajectories.
- **Signed audit receipts** (`agent_team/receipts.py`,
  `python main.py --verify-receipts`). Every trust-boundary action — dry-run
  approval, push, PR open — emits an Ed25519-signed receipt (ticket, commit SHA,
  per-seat votes, timestamp) appended to a hash-chained, append-only log. A
  third party verifies the whole log offline with the public key alone:
  payload tampering fails the signature, and because each receipt embeds the
  previous one's hash, altering or deleting any receipt breaks the chain for all
  that follow. The tamper-evidence layer under the "auditable" claim.
  Deterministic, offline, unit-tested (sign/verify, forged-key rejection,
  mid-log deletion).
- **Reviewer reliability model + Bayesian trust aggregation** (RQ1 substrate:
  `agent_team/reliability.py`, `agent_team/judgments.py`). Every seat verdict is
  logged with its ground-truth label; from that corpus each seat's sensitivity
  (defect recall) and specificity (true-approve) are estimated as Beta-Binomial
  posteriors with exact credible intervals (self-contained regularized incomplete
  beta — no scipy). Aggregation upgrades from majority voting to a calibrated
  posterior `P(defect | votes)` that weights seats by measured reliability and
  ignores coin-flippers automatically; a `conservative` mode uses each seat's
  lower credible bound (fail-closed). `--bench` grows the labeled corpus and
  reports the model; `python main.py --reliability` prints it. Deterministic,
  unit-tested against hand-computed probabilities.
- Goal → council-approved DAG decomposition; dependency-aware scheduling;
  stacked PRs; file-overlap locks; human-only merge.

## Next: reliability

- **Chance-corrected quality measures** (`bench.py`). The benchmark now reports
  Cohen's **κ vs the human gold label** (per seat and for the whole gate — so a
  mostly-approve gate on a mostly-good corpus can't fake a high score the way raw
  accuracy would) and **inter-seat κ** (mean pairwise agreement — the direct
  health check on the heterogeneous jury: low κ = decorrelated seats = the design
  is working; high κ = redundant seats). Deterministic, unit-tested. These are
  the numbers to tune seat routing and aggregation thresholds against.
- **Grow the gold set + tune from it.** The measurement harness exists; now grow
  the corpus toward 30–200 labeled diffs from the real distribution, add
  repeat-run stability (verdict flip-rate under resampling), and use the per-seat
  κ / detection numbers to weight or re-route seats and set aggregation
  thresholds empirically rather than by intuition.
- **Baseline-relative canary alerting.** Canaries alert today on absolute
  thresholds (gate approval, dissent floor). Record a per-seat baseline from the
  gold sweep so a canary can flag a *seat-specific* dissent collapse (a seat that
  used to catch this case and now doesn't), not just a panel-wide one.
- **Deeper mutation coverage.** The teeth check ships today as a single
  source-revert; extend toward per-file mutation and coverage deltas so a
  partially-tested change is scored, not just pass/fail.

## Next: scale

- **Early-exit voting** (`early_exit_voting`, worker gates only). Reviewer
  sessions still in flight are cancelled the instant the outcome is forced — a
  single reject on the unanimous pre-push gate, or a reached majority on the plan
  gate — saving their tokens. Implemented *losslessly*: `_outcome_decided` only
  short-circuits a verdict the full panel would reach anyway, so it needs no
  calibration; the trade is fewer collected objections per rejected round.
  Deterministic and unit-tested (decision table + a cancellation test). A more
  aggressive, calibration-gated variant (exit on a *predicted* outcome before it
  is mathematically forced) remains future work.
- Structured context objects between phases instead of full plan/diff text
  where possible; targeted retrieval for reviewers on large repos.
- Persist decomposer plan + per-ticket state to external memory for very
  large runs; scale reviewer depth with diff size (light pass on small diffs).
- Tune worker concurrency to human merge throughput (practical ceiling is
  review, not compute: ~4–8 concurrent worktrees).

## Next: autonomy

- **Graduated review strictness by risk** (`agent_team/risk.py`). A deterministic
  classifier scores each change's blast radius from its files and added content;
  the gate adapts without ever weakening: **HIGH** (auth / secrets / crypto / DB
  migrations / CI-CD / infra / payment) is forced to the strictest unanimous gate
  and flagged permanent human-merge; **LOW** (docs, lockfiles) still clears the
  full council but is marked safe to fast-track a *human* merge; **MEDIUM** keeps
  the configured gate. The risk level and (always human) merge disposition ride
  in the signed receipt, the PR body, and the run summary. Because human merge is
  permanent here, the "tiers" govern strictness and recommendation — never
  auto-merge. Biased toward HIGH (a false HIGH only adds scrutiny). Unit-tested.
- Promote from a clean track record: mine the receipt log so a class with a long
  clean history can be recommended for lighter *human* ceremony — still never
  auto-merged (auth/migrations/security stay strictest forever).
- **Daemon mode**: long-lived service watching the backlog, maintaining the
  DAG, dispatching as dependencies clear, governed by merge throughput.
- Bounded self-repair on escalation (auto-decompose repeatedly-failing
  tickets) — partial today via revision rounds + ESCALATION.md.
- Self-improvement of reviewer prompts/decomposer heuristics gated on the
  gold-set regression eval only (spec + regression oracle, never a raw metric).

## Product

- MIT license, demo-first README, BYO-key model-agnostic council.
- Integrations: Linear/Jira ingestion, Graphite/gh-stack stacked-PR tooling;
  run alongside GitHub Agent HQ / Anthropic Code Review rather than against.
- Open-core path: hosted operation, audit/compliance plane, org policy
  controls, dashboards.
