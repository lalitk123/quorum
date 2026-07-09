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


_load_cache: dict[str, tuple[int, int, list[dict]]] = {}


def load_judgments(log_file: str) -> list[dict]:
    """Read every judgment row; skips blank/corrupt lines rather than failing.

    Cached by (mtime, size): the log is append-only and grows across the
    system's lifetime, but is re-read from scratch on every ticket's trust
    computation (`worker._build_trust`) plus every bench/mutation/selection
    run — an unchanged file returns the cached parse instead of re-reading
    and re-parsing every row again. Any change to the file (an append from a
    canary or a fresh run) changes its size and invalidates the cache.
    """
    path = Path(log_file)
    key = str(path)
    if not path.exists():
        _load_cache.pop(key, None)
        return []
    stat = path.stat()
    cached = _load_cache.get(key)
    if cached is not None and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2]
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    _load_cache[key] = (stat.st_mtime_ns, stat.st_size, out)
    return out
