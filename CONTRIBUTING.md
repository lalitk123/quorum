# Contributing to Quorum

Thanks for taking a look. Quorum is a research-flavored engineering project: the
goal is a **review gate whose own reliability is measured**, so contributions
that sharpen the measurement are as valuable as ones that add features.

## Getting set up

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

The test suite is fully deterministic and **costs nothing** — no API keys, no
network, no model calls. If a change you make requires a key to test, that is
usually a sign the logic should be pulled out into a pure function.

## Ground rules

**Fail closed.** The core invariant is that anything ambiguous — an unparseable
verdict, a reviewer error, a tripped budget, an approval citing no evidence —
must resolve to *reject*, never to *approve*. A change that makes the gate more
permissive in an error path will be sent back. If you are adding a new failure
mode, add the test that proves it fails closed.

**Deterministic logic gets tests.** Vote aggregation, budget accounting, risk
classification, CUSUM math, the reliability posterior — all of it is plain code
precisely so it can be unit-tested against hand-computed values. Model behavior
is not testable; the code around it is, and that is where the guarantees live.

**Don't hand-write numbers into the docs.** Every figure in `README.md` is
reproducible from a committed snapshot in `bench/judgments/`. If your change
moves the numbers, re-run the relevant command and paste real output — and say
what corpus produced it. Synthetic examples must be labeled as illustrative.

## Especially wanted

- **Gold cases** (`bench/gold/`). The corpus is the bottleneck on every claim
  this project makes. Most wanted: *clean* changes that look alarming, and
  defects subtle enough that two seats miss them **together** — the latter is
  currently what stops the independence audit from having bite. One directory
  with `case.yaml` (`id`/`title`/`description`/`label`/`note`) and `diff.patch`.
- **Reviewer prompt tuning**, backed by a before/after `--bench` run. See the QA
  retune (`55a70d5`) for the shape: measure, name the mechanism, fix, re-measure,
  and report honestly if it only half-worked.
- **New reviewer backends.** `agent_team/reviewers.py` already speaks any
  OpenAI-compatible endpoint; a genuinely different model family is the highest
  -leverage reliability lever there is.

## Pull requests

Keep them focused, explain the *why* in the commit message, and make sure
`pytest -q` passes on 3.10–3.12 (CI checks all three). If a PR changes gate
behavior, say plainly what it makes stricter or looser.

By contributing you agree your work is licensed under the [MIT License](LICENSE).
