"""Deterministic gates that run before the council ever convenes on a diff.

LLM judges alone catch a minority of code errors; pairing them with
deterministic tooling roughly doubles coverage. Cheap code catches what it
can — failing tests, lint errors, tampered test files — so reviewer tokens
are spent only on genuine judgment calls.
"""

import asyncio
import re
import subprocess

CHECK_TIMEOUT_SECONDS = 600
OUTPUT_TAIL_CHARS = 4000

CI_PATH_PATTERN = re.compile(
    r"^(\.github/|\.gitlab-ci|\.circleci/|Jenkinsfile|\.pre-commit-config)"
)
TEST_PATH_PATTERN = re.compile(r"(^|/)(tests?/|test_[^/]*$|[^/]*_test\.[^/.]+$)")
SKIP_MARKER_PATTERN = re.compile(
    r"(pytest\.mark\.skip|unittest\.skip|@skip\b|\.skip\(|xfail|it\.skip|describe\.skip|test\.skip)"
)


def _run_one(command: str, cwd: str) -> tuple[str, bool, str]:
    try:
        proc = subprocess.run(
            command, shell=True, cwd=cwd, capture_output=True, text=True,
            timeout=CHECK_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return command, False, f"timed out after {CHECK_TIMEOUT_SECONDS}s"
    output = (proc.stdout + proc.stderr)[-OUTPUT_TAIL_CHARS:]
    return command, proc.returncode == 0, output


async def run_checks(commands: list[str], cwd: str) -> list[tuple[str, bool, str]]:
    """Run configured check commands (tests, linters) in the worktree.

    Returns (command, passed, output_tail) per command. Sequential on purpose:
    checks often share caches/DBs and the first failure is usually the story.
    """
    results = []
    for command in commands:
        results.append(await asyncio.to_thread(_run_one, command, cwd))
    return results


def tamper_scan(diff: str) -> list[str]:
    """Scan a diff for the documented worker->reviewer gaming surfaces:
    CI/harness edits, deleted assertions in tests, newly skipped tests."""
    findings: list[str] = []
    current_file = ""
    for line in diff.splitlines():
        if line.startswith("diff --git"):
            parts = line.split(" b/", 1)
            current_file = parts[1] if len(parts) == 2 else ""
            if CI_PATH_PATTERN.search(current_file):
                findings.append(f"modifies CI/harness file: {current_file}")
            continue
        in_test_file = bool(TEST_PATH_PATTERN.search(current_file))
        if not in_test_file:
            continue
        if line.startswith("-") and not line.startswith("---") and "assert" in line:
            findings.append(f"removes an assertion in {current_file}: {line[1:].strip()[:120]}")
        if line.startswith("+") and SKIP_MARKER_PATTERN.search(line):
            findings.append(f"adds a skip/xfail marker in {current_file}: {line[1:].strip()[:120]}")
    return findings
