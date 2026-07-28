# Judgment corpus snapshots

Every number in the main README's reliability, independence, and stability
sections was computed from a file in this directory. They are committed so the
published claims are **independently verifiable without spending anything** —
point any analysis command at a snapshot instead of your own log:

```bash
python main.py --reliability  bench/judgments/2026-07-26-qa-tuned.jsonl
python main.py --independence bench/judgments/2026-07-26-qa-tuned.jsonl
python main.py --stability    bench/judgments/2026-07-25-pre-qa-tuning.jsonl
python main.py --select       bench/judgments/2026-07-26-qa-tuned.jsonl
```

Each line is one seat's verdict on one labeled case:
`{"seat", "verdict": approve|reject, "label": defect|clean, "source", "case_id"}`.
Labels are ground truth by construction — gold cases, canaries, and mutants are
defective (or clean) *by design*, which is what makes the corpus free to grow.

| snapshot | rows | corpus | notes |
|---|---|---|---|
| `2026-07-25-pre-qa-tuning.jsonl` | 170 | 20 gold cases, **repeat observations** | Accumulated over several `--bench` runs. The only snapshot with repeat pairs, so it is the one `--stability` can measure test-retest flip rates from. Contains the pre-tuning `qa` seat at 27% specificity. |
| `2026-07-26-qa-tuned.jsonl` | 100 | 20 gold cases, single pass | The current instrument, after the `qa` persona retune (`55a70d5`). Backs the reliability table and the independence audit in the main README. |

The two are kept separate on purpose. A seat's reliability history is keyed by
its **name**, so pooling verdicts from before and after a prompt change would
average two different instruments into one meaningless number — the same hazard
the roadmap's "model-tagged per-seat history" item exists to fix generally.

Reproducing these from scratch costs roughly **$8 per 20-case `--bench` run** at
the configured models, which is exactly why they are committed here.
