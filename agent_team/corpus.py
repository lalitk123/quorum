"""Self-benchmarking corpus (idea #9): the benchmark that grows itself.

Most projects never collect the data to answer "is the reviewer actually getting
better?" Quorum does — every gold case, canary, and mutant appends a labeled row,
and production outcomes can be labeled once known. This module reports the corpus
as an asset: how big it is, where the labels come from, and — the part that
matters — the **statistical power** it currently buys, measured as each seat's
sensitivity credible-interval width. A benchmark of 30 rows and one of 3000 give
very different interval widths, and this makes that concrete, so "grow the gold
set" has a number attached rather than being a vibe.

`record_production_outcome` is the hook that closes the loop: when a shipped
change is later found defective (a revert, an incident), its stored per-seat
verdicts get a ground-truth label and join the training corpus.

Deterministic; unit-tested.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from . import judgments as _judgments
from .reliability import estimate_reliability


@dataclass
class CorpusStats:
    total: int
    labeled: int
    unlabeled: int
    by_source: dict[str, int] = field(default_factory=dict)
    by_label: dict[str, int] = field(default_factory=dict)
    per_seat_n: dict[str, int] = field(default_factory=dict)
    mean_sens_ci_width: float | None = None      # power proxy: narrower = stronger


def corpus_stats(judgments) -> CorpusStats:
    rows = list(judgments)
    by_source = Counter(j.get("source", "unknown") for j in rows)
    by_label = Counter((j.get("label") or "unlabeled") for j in rows)
    labeled = sum(1 for j in rows if j.get("label") in ("defect", "clean"))
    per_seat = Counter(j.get("seat") for j in rows
                       if j.get("seat") and j.get("label") in ("defect", "clean"))

    model = estimate_reliability(rows)
    widths = [m.sens_hi - m.sens_lo for m in model.values()]
    return CorpusStats(
        total=len(rows), labeled=labeled, unlabeled=len(rows) - labeled,
        by_source=dict(by_source), by_label=dict(by_label),
        per_seat_n=dict(per_seat),
        mean_sens_ci_width=(sum(widths) / len(widths)) if widths else None)


def record_production_outcome(log_file: str, case_id: str, seat_verdicts: dict[str, str],
                              was_defect: bool, timestamp: str) -> None:
    """Label a shipped change once its real outcome is known (revert/incident).

    `seat_verdicts` maps seat -> "approve"|"reject" (what each seat said at review
    time). This appends ground-truth-labeled rows that join the training corpus —
    the accepted-PR-becomes-data feedback loop.
    """
    label = "defect" if was_defect else "clean"
    rows = [{"seat": seat, "verdict": verdict, "label": label,
             "source": "production", "case_id": case_id, "timestamp": timestamp}
            for seat, verdict in seat_verdicts.items()]
    _judgments.append_judgments(log_file, rows)


def format_corpus(stats: CorpusStats) -> str:
    lines = ["", "=== Self-benchmarking corpus ===",
             f"  {stats.total} judgment(s): {stats.labeled} labeled, "
             f"{stats.unlabeled} unlabeled",
             f"  by source: {stats.by_source}",
             f"  by label:  {stats.by_label}"]
    if stats.per_seat_n:
        seats = ", ".join(f"{s}={n}" for s, n in sorted(stats.per_seat_n.items()))
        lines.append(f"  labeled per seat: {seats}")
    if stats.mean_sens_ci_width is not None:
        lines.append(f"  statistical power: mean sensitivity 90% CI width = "
                     f"{stats.mean_sens_ci_width:.2f} (narrower = stronger; grow the "
                     "corpus to shrink it)")
    return "\n".join(lines)
