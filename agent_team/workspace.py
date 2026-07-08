"""Git isolation for workers: one worktree + branch per ticket, push/PR helpers.

All git/gh calls are deterministic Python code, not agent actions — the agents
edit files; this module owns everything that touches shared or remote state.

A ticket with dependencies branches from its first dependency's branch and
merges in the rest, so its worktree contains the code it builds on. Its PR is
stacked on the dependency branch rather than the base branch.
"""

import asyncio
import subprocess
from pathlib import Path

MAX_DIFF_CHARS = 60_000


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
        """Stage and commit everything; returns False when there is nothing to commit."""
        await _git_async(self.path, "add", "-A")
        status = await _git_async(self.path, "status", "--porcelain")
        if not status.strip():
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
