# Quorum

**A quorum of adversarial AI reviewers must approve every change — fail-closed,
benchmarked, and cryptographically auditable.**

An AI engineering team built on the **Claude Agent SDK**: parallel worker agents
each take a ticket, and every worker must clear a **council of five specialist
reviewer agents** (architect, security, QA, product, and a red-team adversary)
twice — once to approve its plan, and again to approve its diff before anything is
pushed to GitHub. The differentiator is not the agents that write code — it is the
trustworthy, auditable, fail-closed *review gate* they must pass.

> [!IMPORTANT]
> **This spends real money and writes real code.** Every gate runs five reviewer
> agents, and a worker is gated twice — so a single ticket is ~10 model calls on
> top of the worker's own. A 20-case benchmark run costs roughly **$8** at the
> configured models. Set `budget.per_ticket_usd` / `budget.per_run_usd` in
> `config.yaml` before your first run; a tripped budget fails closed.
>
> Quorum ships with **`dry_run: true`**, which stops workers after council
> approval — nothing is pushed and no PR is opened. Setting `dry_run: false`
> grants autonomous agents push access to the repo you point at. Point
> `repo_path` at a scratch clone until you trust it. Merging is always a human
> action; there is no auto-merge, by design.

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

Inside **Claude Code**, the bundled `/quorum` skill
(`.claude/skills/quorum/SKILL.md`) wraps this same loop: type `/quorum` to run
the team against `goal.md`, or `/quorum <goal text>` to set the goal first. It
runs pre-flight checks (a real `repo_path`, seat credentials, `dry_run`) before
spending any tokens and keeps pushing behind explicit confirmation.

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

# Verify the numbers below without spending anything — every figure in this
# README is computed from a committed snapshot (see bench/judgments/README.md):
python main.py --reliability bench/judgments/2026-07-26-qa-tuned.jsonl
```

```
=== Reviewer reliability (Bayesian, from labeled judgments) ===
  seat          sensitivity[90% CI] (n_def)   specificity[90% CI] (n_clean)
  security       75% [ 55- 91] (n=13)        81% [ 56- 97] (n=7)
  adversary      89% [ 74- 99] (n=13)        69% [ 41- 91] (n=7)
  architect      96% [ 87-100] (n=13)        69% [ 41- 91] (n=7)
  qa             96% [ 87-100] (n=13)        44% [ 17- 72] (n=7)
  product        96% [ 87-100] (n=13)        56% [ 28- 83] (n=7)
```

These are **measured** on the 20-case gold corpus (`bench/gold/`, 100 labeled
judgments across the five seats), not planted — and the seats are genuinely
different instruments. `security` trades sensitivity (75% — it misses subtle
*non*-security logic bugs) for caution; `qa` and `product` catch nearly every
defect (96%) but block clean code far more often. That spread is the whole point
of weighting: on this corpus, reliability-weighted aggregation classified **all 20
cases correctly where the naive unanimous gate got only 15** — the gate rejected
five of the seven clean changes, while the posterior discounted the seats doing
the rejecting and recovered every one. Note what the gate actually failed on:
those five were not sunk by a lone outlier. Three of them (G5, G6, G7) were
rejected by **three of the five seats at once**, and only one was a single-seat
veto. A crying-wolf *bloc* is the harder case, and it is the one calibrated
weighting handles that a vote count cannot — three seats with poor measured
specificity carry less evidence than their headcount suggests.

Two honest caveats. This is an **in-sample** demonstration: the reliability model
is estimated from the same 20 cases it then scores, so it shows the aggregation
rule recovering labels the individual seats got wrong — not out-of-sample
generalization. And the corpus is small with wide CIs. The contribution is the
method; the numbers sharpen as the log grows. (Both figures use a neutral 0.5
prior, which `--bench` prints in its header. Deliberately *not* the gold set's own
~65% defect ratio: a benchmark over-samples defects by construction while a real
review stream is overwhelmingly clean, so using it as a prior would manufacture
false rejects that describe the corpus rather than the council.)

**The loop closes on itself.** These per-seat numbers are not just a report card;
they are a work queue. An earlier run measured `qa` at **27% specificity** against
98% sensitivity — it caught essentially every defect and blocked most clean code
too, which is a prompt defect, precisely located. Its persona had framed the job
as three open-ended "is it tested enough?" questions, and a model can always
answer *no* by imagining an untested edge case. Rewriting it to reject only when
an untested path maps to a **concrete defect that would ship** — with "more tests
would be nice" explicitly barred as grounds — moved `qa` to **44% specificity
with sensitivity unchanged at 96%**: on the same clean cases it went from
approving 1 of 7 to 3 of 7, giving up no defect detection. The residual false
rejects now cluster on genuinely arguable cases (a money-withdrawal and a SQL
endpoint merged with *zero* tests), and at n=7 clean the interval is far too wide
to justify tuning further — the next honest step is more corpus, not more prompt.
That is the intended cycle: measure a seat, fix what the measurement names,
re-measure, and stop when the data can no longer tell you anything.

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
**joint** outcome distribution — the approximation was never necessary. Measured
on the current 20-case corpus:

```console
$ python main.py --independence bench/judgments/2026-07-26-qa-tuned.jsonl
=== Independence audit (is the jury really decorrelated?) ===
  full council ['adversary', 'architect', 'product', 'qa', 'security']
    independence model predicts P(detect) = 100.0%
    measured from the joint          = 96.4% (90% CI lower bound 86.5%, n=13)
    inflation (predicted - measured) = +3.6%   <-- independence is OPTIMISTIC; the selector under-provisions

  mean pairwise φ = 0.00   design effect = 1.00
  5 seats are worth 5.0 independent ones (evidence discount 1.00)
```

On this corpus the result is a *clean bill of health*: mean φ ≈ 0, design effect
1.0, only +3.6% inflation. Even though `security` misses ~30% of the subtle
defects, no *two* seats miss the same one — the misses are spread across seats, so
the heterogeneous jury genuinely decorrelates on detection. That is the design goal
met, and the point is that it is now **measured, not assumed**.

The audit earns its keep because that will not always hold. When a corpus surfaces
a correlated pair, the same table flags it: a positive φ, a joint-miss rate above
what independence predicts, and a low `P(b catches | a misses)` — a seat that does
not cover for its neighbour. The danger is specific: full-council inflation stays
small only because five-seat detection saturates near 100%, so correlation would
hide on exactly the **small subsets the reviewer selector evaluates**, where
detection does not saturate. So `selection.py` scores each candidate subset against
its measured joint detection, and `reliability.posterior_defect_prob` takes a Kish
design-effect `discount` that shrinks a correlated panel's evidence to its
*effective* number of independent seats — two perfect clones rejecting persuade as
much as one seat, not twice. Where joint data is too thin it falls back to the
independence product and labels the step as such rather than pretending. *(On 13
defect cases this joint estimate is still thin; it sharpens as the corpus grows.)*

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
well it scored on a single pass. Measured over 17 repeat pairs — a seat judging the
same case on two separate `--bench` runs. This needs a corpus with repeat
observations, so it is measured on the **pre-tuning** snapshot — the single-pass
one backing the table above reports "all single-pass" and collects no pairs.
Mind the seam: the `qa` row below is therefore the *old* qa persona, not the
retuned one in the reliability table, and the two must not be read across.

```console
$ python main.py --stability bench/judgments/2026-07-25-pre-qa-tuning.jsonl
=== Repeat-run stability (test-retest) ===
  seat          flip rate [90% CI]   pairs   kappa   ceiling   runs@5%
  security       12% [  0- 29]        17    0.75     94%         3
  product         6% [  0- 18]        17    0.82     97%         3
  architect       0% [  0-  0]        17    1.00    100%         1
  qa              0% [  0-  0]        17    1.00~   100%         1
  adversary       0% [  0-  0]        17    1.00    100%         1

  No seat's reported accuracy exceeds its consistency ceiling.
```

The result is itself a finding: the two **OpenAI** seats are the only stochastic
ones (`security` flips 12% of re-runs, `product` 6%), while the three **Claude**
seats reproduce every verdict exactly (0% flip). No seat's reported accuracy exceeds
its ceiling, so `--stability` exits 0 here — but the check has teeth: a seat whose
accuracy's *lower* credible bound still cleared its flip-rate ceiling would fail the
command, a contradiction no estimation noise can manufacture.

`runs@5%` prices the flakiness: how many self-repeats, majority-voted, drive the flip
rate under 5%. `security` and `product` each need 3; the deterministic Claude seats
need 1. And the bound is exact at the pathological end — a seat that flipped >½ of its
re-runs would report `never`, since the majority of `k` fair coins is still a fair
coin. That converts "this seat is flaky" into a line item RQ2's cost model can act on.

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
| `.claude/skills/quorum/SKILL.md` | Claude Code `/quorum` skill: runs the build loop over `goal.md` |
