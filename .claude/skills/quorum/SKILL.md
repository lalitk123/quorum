---
name: quorum
description: "Run the Quorum autonomous engineering team — the five-seat adversarial council gating every AI-written change — over a backlog. Trigger when the user types /quorum, asks to run the council/build loop/agent team, or wants Quorum to build against goal.md. Kicks off goal -> backlog -> worker -> council plan review -> worktree implement -> council pre-push review, in dry_run by default."
---

# /quorum

Run the Quorum multi-agent engineering team over its backlog. A thin, safe
wrapper around `python main.py --config config.yaml`: workers draft plans and
diffs, and a **council of five reviewer seats** (architect, security, QA,
product, red-team adversary) must approve twice — majority on the plan,
unanimous on the pre-push diff — or the change is rejected fail-closed.

## Usage

```
/quorum                 # run the team on the current goal.md, dry_run
/quorum <goal text>     # overwrite goal.md with <goal text>, then run
/quorum --live          # run with dry_run disabled (workers push branches + open PRs)
/quorum --config <path> # use a non-default config file
```

## What to do when invoked

1. **Locate the project.** Run from the Quorum repo root (the dir with
   `main.py` and `config.yaml`). If not there, `cd` to it or ask the user.

2. **Pick the interpreter.** Use `.venv/bin/python` if it exists (it has
   `claude-agent-sdk`, `PyYAML`, `cryptography`); otherwise fall back to
   `python3` and warn if imports fail.

3. **Set the goal.** If the user passed goal text after `/quorum`, write it to
   `goal.md` (preserve the existing `# Goal` / `## Requirements` /
   `## Acceptance criteria` structure). With no argument, use the existing
   `goal.md` as-is — show its first ~15 lines so the user confirms what will be
   built.

4. **Pre-flight checks — do these BEFORE spending any tokens:**
   - **`repo_path` must be real.** Read `config.yaml`. If `repo_path` is still
     the `/path/to/target/repo` placeholder or points at a non-existent /
     non-git directory, STOP and tell the user to set it to a throwaway git
     clone with an `origin` remote. The build loop writes to a *target* repo,
     **not** the Quorum repo itself — never point it at the Quorum working tree.
   - **Credentials.** Claude seats need a Claude Code login or `ANTHROPIC_API_KEY`;
     any OpenAI seats in `council_models` need `OPENAI_API_KEY` exported. If a
     seat is routed to OpenAI and the key is unset, warn that those seats will
     fail-closed to reject.
   - **dry_run.** Confirm `config.yaml:dry_run`. Default is `true` (approve +
     receipt, no push). Only pass `--live` intent (flip dry_run) when the user
     explicitly asked; treat pushing as the outward-facing action it is and
     confirm first.
   - **Budget.** Note the `budget.per_run_usd` / `per_ticket_usd` caps so the
     user knows the ceiling. This spends real tokens.

5. **Run it.** Launch in the background so its progress streams and the session
   stays responsive:
   ```
   .venv/bin/python main.py --config config.yaml -v
   ```
   (append `--config <path>` if the user overrode it). The loop is
   run-once-and-exit: it decomposes the goal, builds the DAG, dispatches workers
   as dependencies clear, and stops when the backlog drains or the run budget
   trips.

6. **Report.** When it finishes, summarize per ticket: plan-review outcome,
   pre-push outcome, trust score, and whether it was approved (dry_run) or
   pushed (live). Surface any `BudgetExceeded` / `DecompositionRejected` abort,
   any seat that errored, and the path to any signed receipts
   (`audit.receipts_file`) if `audit.enabled`.

## Guardrails

- **Never run against the Quorum repo itself.** `repo_path` is a separate
  target clone. Refuse if they collide.
- **dry_run is the safe default.** Do not disable it silently. Pushing branches
  / opening PRs is outward-facing — get explicit confirmation.
- **Fail-closed is a feature, not a bug.** An unparseable review, an errored
  seat, or an approval with no cited evidence all count as rejections. If a run
  rejects everything, first check credentials and seat health, not the council.

## Related CLI (not this skill's job, but adjacent)

`--bench` benchmarks the council on gold cases; `--reliability` /
`--independence` / `--stability` / `--drift` / `--corpus` report reviewer
statistics from the judgment log; `--verify-receipts` validates the audit log.
See `config.yaml` and `README.md`.
