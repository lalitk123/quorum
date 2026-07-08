"""Risk classification and graduated review policy.

Human merge is a permanent design decision here, so "graduated autonomy" never
means auto-merge. It means the gate scales its STRICTNESS and its merge
RECOMMENDATION to the blast radius of a change:

- **HIGH** — auth / authz / sessions / secrets / crypto / DB migrations / CI-CD /
  infra / deploy / payment. The strictest gate (forced-unanimous, never
  downgradable) plus a permanent human-merge requirement: never eligible for an
  expedited merge, no matter the track record.
- **LOW** — docs, lockfiles. The full council still runs (fail-closed
  everywhere), but the change is flagged as safe to fast-track a *human* merge.
- **MEDIUM** — ordinary source: the configured gate and a standard human merge.

The classifier is deterministic and path-first (with a few high-signal content
patterns) so it is unit-testable and cannot be argued with at review time. It is
biased toward HIGH on purpose: a false HIGH only adds scrutiny and keeps merge
human — it never removes a safeguard — so over-triggering is the safe direction.
"""

import re
from dataclasses import dataclass, field
from enum import Enum

from .checks import classify_changed_files


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Path substrings that mark a security- or operations-critical surface.
HIGH_PATH = re.compile(
    r"(auth|login|logout|session|passwd|password|oauth|openid|saml|jwt|"
    r"secret|credential|api[_-]?key|access[_-]?key|private[_-]?key|token|"
    r"crypto|cipher|encrypt|permission|rbac|\bacl\b|"
    r"migrat|alembic|flyway|schema|\.sql$|"
    r"\.github/|gitlab-ci|jenkinsfile|dockerfile|docker-compose|"
    r"\.tf$|terraform|kubernetes|(^|/)k8s/|helm|(^|/)deploy|"
    r"payment|billing|stripe|checkout|invoice)",
    re.I,
)

# Files with essentially no runtime blast radius.
LOW_PATH = re.compile(
    r"(\.md$|\.rst$|\.txt$|(^|/)docs?/|(^|/)LICENSE|CHANGELOG|"
    r"(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|"
    r"Pipfile\.lock|Cargo\.lock|go\.sum|composer\.lock|Gemfile\.lock)$)",
    re.I,
)

# Dangerous constructs that make a change HIGH regardless of where they land.
HIGH_CONTENT = re.compile(
    r"(eval\(|exec\(|pickle\.loads|subprocess\.|os\.system|__import__|"
    r"shell\s*=\s*True|verify\s*=\s*False|hashlib\.md5|\bmd5\(|"
    r"DROP\s+TABLE|ALTER\s+TABLE|TRUNCATE\s+)",
    re.I,
)


@dataclass
class RiskAssessment:
    level: RiskLevel
    reasons: list[str] = field(default_factory=list)


def _content_high(diff: str) -> list[str]:
    """Dangerous patterns among the diff's ADDED lines (deduped, capped)."""
    hits: list[str] = []
    for line in diff.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for m in HIGH_CONTENT.finditer(line):
            token = m.group(0).strip()
            if token not in hits:
                hits.append(token)
    return hits[:8]


def classify(diff: str, files: list[str] | None = None) -> RiskAssessment:
    """Classify a change's blast radius from its files and added content.

    Precedence: any HIGH signal wins; else any ordinary-source file is MEDIUM;
    else (only docs/lockfiles) LOW. An empty/unparseable diff is MEDIUM — we
    never fast-track something we couldn't read.
    """
    paths = files if files is not None else [c.path for c in classify_changed_files(diff)]
    high = [p for p in paths if HIGH_PATH.search(p)]
    low = [p for p in paths if not HIGH_PATH.search(p) and LOW_PATH.search(p)]
    other = [p for p in paths if not HIGH_PATH.search(p) and not LOW_PATH.search(p)]
    content = _content_high(diff)

    if high or content:
        reasons = [f"security/ops-sensitive path: {p}" for p in high]
        reasons += [f"dangerous construct in the diff: {c}" for c in content]
        return RiskAssessment(RiskLevel.HIGH, reasons)
    if other:
        return RiskAssessment(RiskLevel.MEDIUM, [])
    if low:
        return RiskAssessment(RiskLevel.LOW, [f"low-risk file: {p}" for p in low])
    return RiskAssessment(RiskLevel.MEDIUM, ["no files parsed from diff"])


def effective_push_rule(level: RiskLevel, configured_rule: str) -> str:
    """The pre-push consensus rule after risk adjustment.

    HIGH is forced to unanimous and can never be downgraded; LOW and MEDIUM keep
    the configured rule — the gate is never *weakened* by risk, only tightened.
    """
    if level == RiskLevel.HIGH:
        return "unanimous"
    return configured_rule


def merge_disposition(level: RiskLevel) -> str:
    """Human-facing merge recommendation. Merge itself always stays human."""
    return {
        RiskLevel.HIGH: "human-required (security/migration/infra — never auto-eligible)",
        RiskLevel.MEDIUM: "human-review (standard)",
        RiskLevel.LOW: "expedited-human (docs/lockfile — safe to fast-track a human merge)",
    }[level]


def material_section(assessment: RiskAssessment) -> str:
    """Risk block appended to the council's review material."""
    lines = [f"\n\n## Automated risk assessment\nLevel: {assessment.level.value.upper()}"]
    if assessment.reasons:
        lines += [f"- {r}" for r in assessment.reasons]
    if assessment.level == RiskLevel.HIGH:
        lines.append("This change touches a security- or operations-critical surface: "
                     "review under the strictest bar and treat any unjustified risk as "
                     "grounds for rejection.")
    return "\n".join(lines)
