# Quorum

**A quorum of adversarial AI reviewers must approve every change — fail-closed,
benchmarked, and cryptographically auditable.**

An AI engineering team built on the **Claude Agent SDK**: parallel worker agents
each take a ticket, and every worker must clear a **council of five specialist
reviewer agents** (architect, security, QA, product, and a red-team adversary)
twice — once to approve its plan, and again to approve its diff before anything is
pushed to GitHub. The differentiator is not the agents that write code — it is the
trustworthy, auditable, fail-closed *review gate* they must pass.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design and
[ROADMAP.md](ROADMAP.md) for the research program — Quorum's north star is an **AI
software assurance layer** whose reviewers are themselves *measured instruments*,
organized around three questions: can we measure reviewer reliability (RQ1),
adaptively pick reviewers per dollar (RQ2), and detect reviewer drift
automatically (RQ3)?

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
With `early_exit_voting` on, in-flight seats are cancelled once the outcome is
mathematically forced (a reject on the unanimous gate, a reached majority on the
plan gate) — a lossless token saving that never changes the decision.

Seats can run on **different model families** for a heterogeneous jury (the
highest-leverage reliability lever): each seat is a bare Claude model or an
OpenAI-compatible backend spec (GPT / Gemini / DeepSeek / Qwen, directly or
via OpenRouter). See `council_models` in `config.yaml`.

Every approval carries a **formal trust score** (`trust.py`): one 0–100 number
composing the calibrated council posterior, deterministic checks, tamper findings,
clean-sandbox reproduction + teeth, mutation score, and risk into an *itemized*
breakdown (`+ council posterior +56 · + checks +12 · − HIGH-risk −7 …`), written
into the receipt, the PR body, and the run summary. This is the "assurance layer"
made legible — the coding worker is interchangeable; the measurable trust verdict
is the product.

A **deterministic gate** runs before the diff council: configured test/linter
commands must pass in the worktree, and a tamper scan flags gamed tests
(deleted assertions, skip markers, CI edits). Optional **clean-sandbox
verification** re-runs the worker's *committed* tests in a fresh git worktree —
so a suite that passes only on the worker's dirty local state fails closed —
and a lightweight teeth check reverts the change's source to confirm its new
tests actually fail without it (vacuous/gamed tests are surfaced to the
council). Hard per-ticket and per-run USD budget caps trip fail-closed to
escalation.

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

## Benchmark the council

You can't tune a gate you don't measure. `--bench` runs the council against a
labeled gold set of diffs (seeded defects that should be rejected, known-good
diffs that should be approved) and reports how well it discriminates:

```bash
python main.py --bench            # uses bench/gold/ (ships with example cases)
```

It prints a confusion matrix — **defect recall** (fraction of bad diffs
caught; a false-approve is a defect shipped) and **false-reject rate**
(fraction of good diffs wrongly blocked) — plus per-seat detection rates and
cost per review. Two chance-corrected agreement measures round out the picture:
**κ vs human** (Cohen's kappa of the gate/each seat against the gold labels, so
a mostly-approve gate on a mostly-good corpus can't fake a high score), and
**jury correlation** (mean pairwise κ between seats — the direct health check
on the heterogeneous jury: genuinely decorrelated seats show *low* κ; high κ
means the extra seats are redundant). Add your own cases as `bench/gold/<name>/`
directories (`case.yaml` with a `label:`, a `diff.patch`, and an optional
`workspace/` tree). This is real API spend, bounded by `budget.per_ticket_usd`
per case.

## Reviewer reliability (measured, not assumed)

Majority voting treats every reviewer as equally trustworthy. Quorum instead
*measures* them. Every seat verdict is logged with its ground-truth label — gold
cases, canaries, and mutants are defective by construction, so labels come free —
and each seat is modeled as a noisy sensor: its **sensitivity** (fraction of real
defects it catches) and **specificity** (fraction of clean diffs it doesn't block)
are estimated as Beta-Binomial posteriors with credible intervals, so a seat
judged on three cases and one judged on three hundred aren't treated alike.

```bash
python main.py --bench          # grows the labeled corpus + reports reliability
python main.py --reliability    # prints the current per-seat model
```

```
=== Reviewer reliability (Bayesian, from labeled judgments) ===
  seat          sensitivity[90% CI] (n_def)   specificity[90% CI] (n_clean)
  product        50% [ 28- 72] (n=12)        96% [ 85-100] (n=12)
  security       96% [ 85-100] (n=12)        96% [ 85-100] (n=12)
  adversary      96% [ 85-100] (n=12)        73% [ 52- 90] (n=12)
```

Those estimates then drive **calibrated aggregation** (`reliability.py`): instead
of `3-of-5 approve`, the posterior `P(defect | votes)` weights each seat by its
measured reliability — a coin-flip seat (sensitivity ≈ specificity ≈ 0.5)
contributes nothing automatically, and a proven seat dominates. A `conservative`
mode uses each seat's *lower* credible bound, keeping the gate fail-closed against
thinly-evidenced reviewers. The math is dependency-free and unit-tested against
hand-computed probabilities. *(This is RQ1; RQ2 conditions these estimates on the
change to pick reviewers per dollar, RQ3 tracks them over time for drift.)*

## Do the reviewers actually disagree? (independence audit)

Both of the formulas above assume seats err **independently** — the Bayesian
posterior multiplies their likelihood ratios, and the reviewer selector computes
`P(detect) = 1 - Π(1 - sensitivityᵢ)`. A heterogeneous jury is *designed* for that
independence, but designing for something is not measuring it, and the assumption
fails in the direction that hurts: correlated seats make the posterior
overconfident, and make the selector believe a near-clone bought fresh coverage,
so it recommends a council that is **too small**.

Since every seat reviews every case, the judgment log already contains the full
**joint** outcome distribution — the approximation was never necessary. The run
below is *illustrative*, on a synthetic 60-case log with a deliberately planted
clone pair; real numbers require a real corpus:

```console
$ python main.py --independence
=== Independence audit (is the jury really decorrelated?) ===
  full council ['adversary', 'architect', 'product', 'qa', 'security']
    independence model predicts P(detect) = 98.8%
    measured from the joint          = 97.5% (90% CI lower bound 93.7%, n=60)
    inflation (predicted - measured) = +1.3%   <-- independence holds well

  mean pairwise φ = 0.12   design effect = 1.50
  5 seats are worth 3.3 independent ones (evidence discount 0.67)

  pair                     φ    joint-miss (obs vs indep)   P(b catches | a misses)
  architect + product    +0.83    53% vs 33% (!)           9%
  adversary + security   +0.23    12% vs 7%               59%
  ...
  adversary + product    -0.42    7% vs 16%               76%
```

Read that bottom table, not the top line. `architect + product` miss the same
defects (φ = +0.83) and share a blind spot 53% of the time where independence
predicts 33% — and when `architect` misses a defect, `product` catches it only
**9%** of the time. It does not cover for it. Meanwhile `adversary + product` are
genuinely complementary (φ = −0.42).

The full-council inflation looks harmless (+1.3%) only because five-seat detection
saturates near 100%. On the *subsets the selector actually evaluates* the same data
gives `architect + product`: **predicted 67.1%, measured 46.7% — a +20.4% overstatement.**
The error hides exactly where it does damage. So `selection.py` now scores each
candidate subset against its measured joint detection, and
`reliability.posterior_defect_prob` takes a Kish design-effect `discount` that
shrinks a correlated panel's evidence to its *effective* number of independent
seats — two perfect clones rejecting now persuade exactly as much as one seat, not
twice as much. Where joint data is too thin, it falls back to the independence
product and labels the step as such rather than pretending.

## Does a reviewer even *have* an accuracy? (repeat-run stability)

Every sensitivity above is measured from **one verdict per seat per case**. But an
LLM reviewer is stochastic: ask it the same question twice and it may reject once
and approve once. If a seat changes its mind that easily, "96% sensitivity" is not
a property of the seat — it is one draw the benchmark happened to catch on a good
day. So measure the seat against *itself*:

```bash
python main.py --stability         # exits non-zero on a ceiling violation
```

Let `p` be the seat's probability of rejecting a given defect. Run it twice on that
case and it contradicts itself with probability `2p(1-p)`. Pool that flip rate `f`
across defect cases, apply Jensen's inequality, and a hard bound falls out:

> **|s − ½| ≤ √(1 − 2f) / 2** — a seat cannot be more *accurate* than it is
> self-*consistent*.

It is distribution-free, and it bounds every accuracy number this platform reports.
A reviewer that flips 32% of its re-runs cannot have sensitivity above 80%, however
well it scored on a single pass. The run below is *illustrative*, on a synthetic log
where `architect` was planted to look strong on the once-judged gold set and be a
coin flip on the re-run canaries:

```console
$ python main.py --stability
=== Repeat-run stability (test-retest) ===
  seat          flip rate [90% CI]   pairs   kappa   ceiling   runs@5%
  architect      55% [ 52- 60]        60   -0.10     50%     never  (!) f>1/2: chance
  product        53% [ 42- 62]        60   -0.11     50%     never  (!) f>1/2: chance
  qa             47% [ 33- 58]        60    0.06     63%        55
  adversary      23% [ 10- 38]        60    0.08~    87%         5
  security       10% [  0- 20]        60   -0.05~    95%         3

  Note: reported accuracy is above even the most generous ceiling the flip data allows:
     architect  sensitivity reported 61% vs ceiling 50% from a 55% flip rate [+10.9 pts]
     This seat's apparent competence lives in cases it was never asked twice.
```

`--reliability` rates `architect` at 61% sensitivity and the Bayesian aggregation
duly weights its vote. Stability says the verdict is **irreproducible**: it
contradicts itself on 55% of re-runs, which is what a coin does. The decision that
follows is categorical — retire or re-route the seat — not a 0.61 weight.

`runs@5%` prices the flakiness: how many self-repeats, majority-voted, drive the
flip rate under 5%. `security` needs 3; `adversary` needs 5; `architect` says
**never**, and that is exact rather than pessimistic — the majority of `k` fair
coins is still a fair coin. That converts "this seat is flaky" into a line item
RQ2's cost model can act on.

Two details the arithmetic forces. Pairs drawn from one case share that case's `p`,
so a Beta interval over pairs would be too narrow; the CI is a **cluster bootstrap
over whole cases**. And the `~` on κ marks the **kappa paradox**: `security` rejects
95% of the time, so two of its runs already agree 90% of the time by chance, and its
κ goes *negative* despite a 10% flip rate. There κ measures the marginal, not the
consistency. Read the flip rate.

## Verify the verifier (canary diffs)

A benchmark measures the gate once; a live gate can still drift into a rubber
stamp. With `canary.enabled: true`, the orchestrator periodically slips a
known-bad gold case into the **live** review stream and alerts if the council
approves it (a seeded defect that would have shipped) or if too few seats
dissented (the leading indicator of drift). Run it on demand against the current
live configuration:

```bash
python main.py --canary            # every bad gold case vs the live council, once
```

Each canary is one extra council review, charged to the run budget and skipped
(never fatal) if the budget is tight. Every canary review also persists a labeled
per-seat judgment, which feeds both the reliability model and drift detection.

## Reviewer drift detection (the alarm rings itself)

A canary that fires once tells you the gate is healthy *now*. To catch a reviewer
*degrading* — a model update, creeping sycophancy — you need the outcomes over
time. Each seat's canary results form a miss stream, and a **CUSUM** (Page's
sequential test) raises an alarm the moment a seat's misses accumulate past a
threshold — with the observation index where drift began:

```bash
python main.py --drift             # exits non-zero if any seat is degrading
```

```
=== Reviewer drift (CUSUM over canary outcomes) ===
  seat          obs  misses  miss_rate  cusum_peak  status
  qa             16       6       38%      13.82  DRIFT ALARM @obs 11
  security       16       0        0%       0.00  stable
```

A lone miss decays back to zero (no false alarm on noise); sustained degradation
trips the threshold. `--drift` is a drop-in CI/cron gate on the gate itself — the
answer to "if the security seat starts approving SQL injections, does anyone
notice?" is now *yes, automatically*.

## Graduated review strictness (risk-aware)

With `autonomy.enabled: true`, a deterministic classifier scores each change's
blast radius and adapts the gate — only ever tightening it:

- **HIGH** (auth, secrets, crypto, DB migrations, CI/CD, infra, payment) is forced
  to the strictest unanimous gate and marked **permanent human-merge**;
- **LOW** (docs, lockfiles) still clears the full council but is flagged safe to
  fast-track a *human* merge;
- **MEDIUM** (ordinary source) keeps the configured gate.

The risk level and merge disposition ride in the signed receipt, the PR body, and
the run summary. Merge itself always stays human — a permanent design decision —
so these "autonomy tiers" govern strictness and recommendation, never auto-merge.
The classifier is biased toward HIGH: a false HIGH only adds scrutiny.

## Signed audit receipts

With `audit.enabled: true`, every trust-boundary action — a dry-run approval, a
push, a PR open — emits an **Ed25519-signed, hash-chained receipt** recording the
ticket, the exact commit SHA, and each seat's vote. The receipts land in an
append-only log (`audit.receipts_file`). Anyone with the public key can verify
the whole log offline, with no access to this process:

```bash
python main.py --verify-receipts        # checks signatures + the hash chain
```

Because each receipt carries the hash of the previous one, tampering with a
payload fails its signature *and* breaks the chain for every later receipt, and
deleting a receipt from the middle is detected too. The command exits non-zero if
any receipt fails — a drop-in audit gate for regulated contexts. The signing key
is created (mode `0600`) on first use if absent; back it up and pin its public
key, since receipts are only meaningful against a stable operator-held key.

`dry_run: true` (the default) stops after council approval: the branch and
worktree exist locally under `<repo>/.worktrees/`, nothing is pushed. Flip it
to `false` to let approved workers push and open PRs.

Escalated tickets (council never approved within `max_review_rounds`) keep
their worktree on disk so a human can inspect what the agent and the council
disagreed about.

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

## Layout

| File | Role |
|---|---|
| `main.py` | CLI entry point |
| `agent_team/decomposer.py` | Goal → council-approved ticket backlog (goal mode) |
| `agent_team/orchestrator.py` | Parallel scheduling, file-overlap locks, summary |
| `agent_team/worker.py` | Per-ticket lifecycle: plan → review → implement → review → PR |
| `agent_team/council.py` | The five reviewer personas + vote aggregation |
| `agent_team/reviewers.py` | Pluggable reviewer backends (Claude + OpenAI-compatible) |
| `agent_team/checks.py` | Deterministic test/linter gate + diff tamper scan |
| `agent_team/sandbox.py` | Clean-sandbox test verification (reproduction + teeth check) |
| `agent_team/risk.py` | Risk classifier + graduated review strictness (human merge stays) |
| `agent_team/bench.py` | Council quality benchmark over a labeled gold set |
| `agent_team/canary.py` | Live known-bad canary injection (verify the verifier) |
| `agent_team/reliability.py` | Reviewer reliability model + Bayesian trust aggregation (RQ1) |
| `agent_team/judgments.py` | Persistent labeled reviewer-judgment log (the dataset) |
| `agent_team/drift.py` | Continuous reviewer-drift detection via CUSUM (RQ3) |
| `agent_team/mutation.py` | Mutation-based review evaluation + free labeled-data engine (#7) |
| `agent_team/selection.py` | Adaptive min-cost reviewer selection per dollar (RQ2) |
| `agent_team/independence.py` | Audits the independence assumption; measured joint detection |
| `agent_team/stability.py` | Test-retest flip rate + the accuracy ceiling it implies |
| `agent_team/trust.py` | Formal trust score composing all assurance signals (#8/#10) |
| `agent_team/evolution.py` | Per-seat model comparison + swap recommendations (#6) |
| `agent_team/corpus.py` | Self-benchmarking corpus size + statistical-power report (#9) |
| `agent_team/evidence.py` | Structured evidence graphs for explainable verdicts (#4) |
| `agent_team/receipts.py` | Ed25519-signed, hash-chained audit receipts + verifier |
| `agent_team/workspace.py` | Git worktrees, commits, diffs, push/PR (deterministic code) |
| `agent_team/backlog.py` | tasks.yaml or GitHub Issues backlog |
| `agent_team/runner.py` | Thin `query()` wrapper + JSON extraction |
