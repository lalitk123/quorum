"""Git isolation for workers: one worktree + branch per ticket, push/PR helpers.

All git/gh calls are deterministic Python code, not agent actions — the agents
edit files; this module owns everything that touches shared or remote state.

A ticket with dependencies branches from its first dependency's branch and
merges in the rest, so its worktree contains the code it builds on. Its PR is
stacked on the dependency branch rather than the base branch.
"""

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path

from .models import invalid_ticket_id

log = logging.getLogger("workspace")

MAX_DIFF_CHARS = 60_000

# Build/cache artifacts the worker's own test runs leave behind. `git add -A`
# sweeps these into the diff the council reviews, which wastes a review round on
# bytecode and invites correct-but-worthless rejections — observed live: a seat
# spent a whole round objecting to `src/__pycache__/units.cpython-314.pyc`.
# Quorum must not depend on the *target* repo's .gitignore to keep junk out of
# its own review material.
#
# Deliberately conservative. Withholding a file the worker meant to ship would
# let the council approve a diff that does not match the worktree — a fail-OPEN
# failure, the one direction this system must never take. So this lists only
# artifacts that are never source, and omits ambiguous ones (`build/`, `dist/`)
# which can be real directories; those stay the target repo's business. Anything
# withheld is logged, never dropped silently.
#
# Applied via `-c core.excludesFile`, which is gitignore semantics: it affects
# untracked files ONLY. A repo that already tracks one of these keeps tracking
# it — we never start ignoring something the project chose to version.
ARTIFACT_EXCLUDES = (
    "__pycache__/", "*.py[cod]", "*.egg-info/",
    ".pytest_cache/", ".mypy_cache/", ".ruff_cache/", ".tox/", ".nox/",
    ".coverage", ".coverage.*", "htmlcov/",
    "node_modules/", ".DS_Store",
)

_excludes_path: str | None = None


def _artifact_exclude_file() -> str:
    """Path to a git excludes file holding ARTIFACT_EXCLUDES (written once).

    Passed per-invocation rather than written into the repo: `.git/info/exclude`
    is shared across every worktree of the repo, so concurrent workers writing
    it would race and it would outlive the run.
    """
    global _excludes_path
    if _excludes_path is None:
        fd, path = tempfile.mkstemp(prefix="quorum-artifact-excludes-")
        with open(fd, "w") as fh:
            fh.write("\n".join(ARTIFACT_EXCLUDES) + "\n")
        _excludes_path = path
    return _excludes_path


class GitError(RuntimeError):
    pass


def _git(repo: str | Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


async def _git_async(repo: str | Path, *args: str) -> str:
    return await asyncio.to_thread(_git, repo, *args)


class Workspace:
    def __init__(self, repo_path: str, base_branch: str, ticket_id: str,
                 start_refs: list[str] | None = None):
        # Validate at the sink, not just at the sources: this is where the id
        # stops being data and becomes a path. `Path / x` silently discards the
        # prefix when x is absolute, so an unchecked id escapes the repo entirely
        # rather than nesting under it. Fail closed — a ticket we cannot place
        # safely is a ticket we do not start.
        reason = invalid_ticket_id(ticket_id)
        if reason:
            raise ValueError(reason)
        self.repo = Path(repo_path)
        self.base = base_branch
        self.branch = f"agent/{ticket_id}"
        self.path = self.repo / ".worktrees" / ticket_id
        self.start_refs = start_refs or [base_branch]
        self.review_base: str | None = None  # HEAD after setup; diffs measure from here

    @property
    def pr_base(self) -> str:
        """Stack the PR on a dependency branch when there is one."""
        first = self.start_refs[0]
        return first if first != self.base else self.base

    async def create(self) -> None:
        """Create the worktree from the first start ref and merge in the rest.

        Raises GitError on merge conflict — the caller reports the ticket as
        blocked rather than letting an agent resolve conflicts silently.
        """
        self.path.parent.mkdir(exist_ok=True)
        first, *rest = self.start_refs
        await _git_async(self.repo, "worktree", "add", "-b", self.branch,
                         str(self.path), first)
        for ref in rest:
            try:
                await _git_async(self.path, "merge", "--no-edit", ref)
            except GitError:
                await _git_async(self.path, "merge", "--abort")
                raise
        self.review_base = (await _git_async(self.path, "rev-parse", "HEAD")).strip()

    async def commit_all(self, message: str) -> bool:
        """Stage and commit the worker's changes, minus build artifacts.

        Returns False when there is nothing to commit. Artifacts the worker's own
        test runs produced are excluded from staging so the council reviews the
        change, not the bytecode beside it; anything withheld is logged.
        """
        excludes = _artifact_exclude_file()
        await _git_async(self.path, "-c", f"core.excludesFile={excludes}", "add", "-A")

        withheld = (await _git_async(
            self.path, "ls-files", "--others", "--ignored",
            f"--exclude-from={excludes}")).split()
        if withheld:
            # Say it out loud. Silently withholding worker output is how a diff
            # under review drifts from what is actually on the branch.
            log.info("not staged (build artifacts, %d): %s", len(withheld),
                     ", ".join(withheld[:10]) + (" …" if len(withheld) > 10 else ""))

        # Ask the index, not the working tree. `git status` does not know about
        # the excludes passed to `add` above, so a round that produced only
        # artifacts looks dirty to it — and we would then run `git commit` with
        # an empty index and fail the ticket on a git error.
        staged = await _git_async(self.path, "diff", "--cached", "--name-only")
        if not staged.strip():
            return False
        await _git_async(self.path, "commit", "-m", message)
        return True

    async def head_sha(self) -> str:
        return (await _git_async(self.path, "rev-parse", "HEAD")).strip()

    async def diff_for_review(self) -> str:
        """Only this ticket's changes — dependency code is behind review_base."""
        diff = await _git_async(self.path, "diff", f"{self.review_base}..HEAD")
        if len(diff) > MAX_DIFF_CHARS:
            diff = diff[:MAX_DIFF_CHARS] + "\n... [diff truncated for review] ..."
        return diff

    async def push_and_open_pr(self, title: str, body: str) -> str:
        await _git_async(self.path, "push", "-u", "origin", self.branch)
        proc = await asyncio.to_thread(
            lambda: subprocess.run(
                ["gh", "pr", "create", "--title", title, "--body", body,
                 "--base", self.pr_base, "--head", self.branch],
                cwd=self.path, capture_output=True, text=True,
            )
        )
        if proc.returncode != 0:
            raise GitError(f"gh pr create failed: {proc.stderr.strip()}")
        return proc.stdout.strip()

    async def remove(self, keep_branch: bool = True) -> None:
        await _git_async(self.repo, "worktree", "remove", "--force", str(self.path))
        if not keep_branch:
            await _git_async(self.repo, "branch", "-D", self.branch)
