"""Mutation-based review evaluation (idea #7) — and the free-label engine.

Two jobs, one mechanism. Mutation testing asks: if we inject a small, deliberate
fault, does the safety net catch it? Here the "safety net" is the *review
council*, not the test suite. We take real source, apply a single minimal
mutation (`if authenticated:` -> `if True:`, `==` -> `!=`, `and` -> `or`, ...),
and each resulting diff is a **known-bad change by construction**. Feeding those
to the council measures a *mutation score* — the fraction of injected faults the
gate kills — a far better signal of review quality than "5 LLMs agreed."

The second job is the important one for the research program: every mutant is a
ground-truth-labeled datapoint, generated in bulk with zero human labeling. That
is the data source RQ2 (adaptive reviewer selection) needs — enough labeled
judgments per (seat × condition) cell to actually learn where each reviewer is
strong. Mutants join gold cases and canaries as `label="defect"` rows in the
judgment log.

Mutant *generation* is deterministic and unit-tested; the council evaluation
reuses the benchmark machinery.
"""

import difflib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("mutation")

# Spaced binary-operator swaps — the surrounding spaces keep us from corrupting
# `<<`, `->`, `**`, `!=`, etc., and match ordinary infix code.
_SPACED_SWAPS: list[tuple[str, str, str]] = [
    ("eq->ne", " == ", " != "),
    ("ne->eq", " != ", " == "),
    ("lt->ge", " < ", " >= "),
    ("gt->le", " > ", " <= "),
    ("le->gt", " <= ", " > "),
    ("ge->lt", " >= ", " < "),
    ("and->or", " and ", " or "),
    ("or->and", " or ", " and "),
    ("add->sub", " + ", " - "),
    ("sub->add", " - ", " + "),
]

_WORD_SWAPS: list[tuple[str, re.Pattern, str]] = [
    ("true->false", re.compile(r"\bTrue\b"), "False"),
    ("false->true", re.compile(r"\bFalse\b"), "True"),
]

# `if <cond>:` / `elif <cond>:` -> force the branch always-taken (classic).
_FORCE_TRUE = re.compile(r"^(\s*)(el)?if\s+.+:\s*$")
# `if not X:` -> `if X:` (drop a guard's negation).
_DROP_NOT = re.compile(r"\bif not \b")

_MUTABLE_SUFFIXES = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".java",
                     ".rb", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp"}


@dataclass
class Mutant:
    id: str
    path: str
    operator: str
    lineno: int            # 1-based
    original_line: str
    mutated_line: str
    diff: str              # git-style unified diff, original -> mutant


def _file_diff(path: str, lines: list[str], idx: int, new_line: str) -> str:
    mutated = list(lines)
    mutated[idx] = new_line
    body = difflib.unified_diff(lines, mutated, fromfile=f"a/{path}",
                                tofile=f"b/{path}", lineterm="")
    return f"diff --git a/{path} b/{path}\n" + "\n".join(body) + "\n"


_COMMENT_MARKERS = ("#", "//", "*", "/*", "--", ";", "%")
_HASH_COMMENT_SUFFIXES = {".py", ".rb", ".sh", ".yaml", ".yml", ".pl"}


def _code_len(line: str, hash_comment: bool) -> int:
    """Index at which an inline comment starts (rest is not code). Approximate —
    a `#`/`//` inside a string literal ends scanning early, which only ever drops
    mutation sites, never invents a bad one."""
    marker = " #" if hash_comment else " //"
    i = line.find(marker)
    return i if i != -1 else len(line)


def _line_mutations(line: str, code_len: int):
    """Yield (operator, mutated_line) for single-site mutations inside code."""
    for name, old, new in _SPACED_SWAPS:
        start = 0
        while (i := line.find(old, start)) != -1:
            if i < code_len:
                yield name, line[:i] + new + line[i + len(old):]
            start = i + len(old)
    for name, pat, new in _WORD_SWAPS:
        for m in pat.finditer(line):
            if m.start() < code_len:
                yield name, line[:m.start()] + new + line[m.end():]
    if _FORCE_TRUE.match(line) and " True:" not in line:
        indent = line[:len(line) - len(line.lstrip())]
        kw = "elif" if line.lstrip().startswith("elif") else "if"
        yield "force-true", f"{indent}{kw} True:"
    if (m := _DROP_NOT.search(line)) and m.start() < code_len:
        yield "drop-not", line[:m.start()] + "if " + line[m.end():]


def _skippable(stripped: str) -> bool:
    return not stripped or stripped.startswith(_COMMENT_MARKERS)


def generate_mutants(source: str, path: str, max_mutants: int = 50) -> list[Mutant]:
    """Every single-site mutant of `source`, deterministically ordered, capped.

    Comment-only lines and Python triple-quoted docstrings are skipped so that
    injected faults land in executable code, not prose — a mutation in a comment
    is not a defect and would be a mislabeled training row.
    """
    lines = source.splitlines()
    hash_comment = Path(path).suffix in _HASH_COMMENT_SUFFIXES
    mutants: list[Mutant] = []
    in_docstring = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        triple = line.count('"""') + line.count("'''")
        if in_docstring:
            if triple % 2 == 1:
                in_docstring = False
            continue
        if triple % 2 == 1:
            in_docstring = True     # opens a multi-line docstring; skip this line too
            continue
        if _skippable(stripped) or (triple and stripped.startswith(('"""', "'''"))):
            continue
        for operator, mutated in _line_mutations(line, _code_len(line, hash_comment)):
            if mutated == line:
                continue
            mutants.append(Mutant(
                id=f"{Path(path).name}:{idx + 1}:{operator}:{len(mutants)}",
                path=path, operator=operator, lineno=idx + 1,
                original_line=line, mutated_line=mutated,
                diff=_file_diff(path, lines, idx, mutated),
            ))
            if len(mutants) >= max_mutants:
                return mutants
    return mutants


def _source_files(target: str) -> list[Path]:
    root = Path(target)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix in _MUTABLE_SUFFIXES
                  and ".worktrees" not in p.parts and "/." not in str(p))


def mutants_for_target(target: str, max_per_file: int = 20) -> list[Mutant]:
    out: list[Mutant] = []
    for path in _source_files(target):
        try:
            src = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        out.extend(generate_mutants(src, str(path), max_mutants=max_per_file))
    return out


# ---- Council evaluation (reuses the benchmark machinery) ---------------------

def mutation_score(results) -> dict:
    """From bench CaseResults over mutants: fraction of injected faults killed."""
    scored = [r for r in results if not r.error]
    killed = sum(not r.decision_approved for r in scored)   # rejected = fault caught
    return {
        "n": len(scored), "killed": killed, "survived": len(scored) - killed,
        "mutation_score": killed / len(scored) if scored else None,
    }


async def run_mutation_eval(cfg: dict, target: str, max_per_file: int = 20,
                            limit: int | None = None) -> dict:
    """Generate mutants under `target`, review each, and score + log the results."""
    import asyncio

    from . import bench, judgments, reliability

    mutants = mutants_for_target(target, max_per_file)
    if limit:
        mutants = mutants[:limit]
    if not mutants:
        raise ValueError(f"no mutants generated from {target}")
    log.info("evaluating council on %d mutant(s)", len(mutants))

    cases = [bench.GoldCase(id=m.id, title=f"mutant {m.operator}",
                            description=f"Injected fault ({m.operator}) at "
                                        f"{m.path}:{m.lineno}.",
                            expected_approve=False, diff=m.diff,
                            note=m.operator, workspace_src=None)
             for m in mutants]

    budget_cap = (cfg.get("budget") or {}).get("per_ticket_usd")
    sem = asyncio.Semaphore(int(cfg.get("max_concurrent_workers", 3)))

    async def one(case):
        async with sem:
            return await bench._run_case(case, cfg, budget_cap)

    results = list(await asyncio.gather(*(one(c) for c in cases)))
    score = mutation_score(results)

    rel_log = (cfg.get("reliability") or {}).get("log_file")
    if rel_log:
        judgments.append_judgments(
            rel_log, bench.judgments_from_results(results, "mutation"))
        model = reliability.estimate_reliability(judgments.load_judgments(rel_log))
        print(reliability.format_reliability(model))

    print(format_mutation_report(score, mutants, results))
    return score


def format_mutation_report(score: dict, mutants, results) -> str:
    lines = ["", "=== Mutation-based review evaluation ==="]
    by_op: dict[str, list[bool]] = {}
    for m, r in zip(mutants, results):
        if r.error:
            continue
        by_op.setdefault(m.operator, []).append(not r.decision_approved)
    for op in sorted(by_op):
        kills = by_op[op]
        lines.append(f"  {op:<12} killed {sum(kills)}/{len(kills)}")
    ms = score["mutation_score"]
    lines.append(f"mutation score: {ms * 100:.0f}% "
                 f"({score['killed']}/{score['n']} injected faults caught)"
                 if ms is not None else "mutation score: n/a")
    if score.get("survived"):
        lines.append(f"  {score['survived']} mutant(s) SURVIVED — faults the "
                     "council would have let through.")
    return "\n".join(lines)
