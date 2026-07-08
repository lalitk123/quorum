"""Persistent reviewer-judgment log — the dataset under the reliability model.

Every individual seat verdict, with its ground-truth label when one exists, is
appended here as one JSON line. This is the corpus that `reliability.py` learns
from and the thing that makes the three research questions answerable over time:

- RQ1 (measure reliability): estimate per-seat sensitivity/specificity from the
  labeled rows.
- RQ2 (adaptive selection): condition those estimates on `condition` (bug type,
  language, risk class) to learn which seats to trust where.
- RQ3 (drift): the rows are timestamped, so a seat's reliability becomes a time
  series you can run a sequential/change-point test on.

Labels come for free from sources that are defective *by construction* — gold
cases, injected canaries, and mutants — so the log grows without human labeling.
Production judgments are stored unlabeled (label=None) until an outcome is known.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Judgment:
    seat: str
    verdict: str                      # "approve" | "reject"
    label: str | None = None          # "defect" | "clean" | None (unlabeled)
    source: str = "production"        # gold | canary | mutation | production
    backend: str = "claude"
    model: str = ""
    case_id: str = ""
    condition: dict = field(default_factory=dict)  # e.g. {"risk": "high", "lang": "py"}
    timestamp: str = ""               # ISO-8601, supplied by the caller
    cost_usd: float = 0.0


def append_judgments(log_file: str, judgments) -> None:
    """Append judgments (Judgment instances or plain dicts) as JSON lines."""
    rows = [asdict(j) if isinstance(j, Judgment) else dict(j) for j in judgments]
    if not rows:
        return
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def load_judgments(log_file: str) -> list[dict]:
    """Read every judgment row; skips blank/corrupt lines rather than failing."""
    path = Path(log_file)
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
