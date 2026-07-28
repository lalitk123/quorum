"""Workspace staging tests against a real git repo.

`commit_all` used to `git add -A`, so any bytecode the worker's own test runs
produced landed in the diff the council reviewed. Observed live: a seat spent a
full rejection round objecting to `src/__pycache__/units.cpython-314.pyc`.
"""

import subprocess

import pytest

from agent_team.workspace import Workspace


def _git(repo, *args):
    proc = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t.test")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "seed.txt").write_text("seed\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


async def _staged_change(tmp_path, build):
    """Run `build(worktree)` inside a workspace, commit, return changed paths."""
    repo = _repo(tmp_path)
    ws = Workspace(str(repo), "main", "T1")
    await ws.create()
    build(ws.path)
    committed = await ws.commit_all("work")
    if not committed:
        return []
    return sorted(_git(ws.path, "diff", "--name-only",
                       f"{ws.review_base}..HEAD").splitlines())


@pytest.mark.asyncio
async def test_build_artifacts_are_not_staged(tmp_path):
    """The real case from the live run: source is reviewed, bytecode is not."""
    def build(wt):
        (wt / "src").mkdir()
        (wt / "src" / "units.py").write_text("def c_to_f(c):\n    return c * 9 / 5 + 32\n")
        cache = wt / "src" / "__pycache__"
        cache.mkdir()
        (cache / "units.cpython-314.pyc").write_bytes(b"\x00compiled")
        (wt / ".pytest_cache").mkdir()
        (wt / ".pytest_cache" / "CACHEDIR.TAG").write_text("x")
        (wt / ".DS_Store").write_bytes(b"\x00")

    assert await _staged_change(tmp_path, build) == ["src/units.py"]


@pytest.mark.asyncio
async def test_ordinary_source_is_still_staged(tmp_path):
    """The guard must not eat real work — including files that merely look
    build-ish by name but are ordinary source."""
    def build(wt):
        (wt / "app.py").write_text("x = 1\n")
        (wt / "docs").mkdir()
        (wt / "docs" / "notes.md").write_text("hi\n")
        (wt / "build_helper.py").write_text("y = 2\n")   # 'build' prefix, real source
        (wt / "dist_config.yaml").write_text("k: v\n")   # 'dist' prefix, real source

    assert await _staged_change(tmp_path, build) == [
        "app.py", "build_helper.py", "dist_config.yaml", "docs/notes.md"]


@pytest.mark.asyncio
async def test_deletions_are_still_staged(tmp_path):
    """`add -A` semantics must survive: the live escalation hinged on the council
    seeing a DELETED file, so deletions must never be filtered out."""
    def build(wt):
        (wt / "seed.txt").unlink()

    assert await _staged_change(tmp_path, build) == ["seed.txt"]


@pytest.mark.asyncio
async def test_an_artifact_already_tracked_keeps_being_tracked(tmp_path):
    """gitignore semantics: excludes apply to untracked files only. A repo that
    chose to version one of these keeps it under review — we do not silently
    stop tracking a file the project committed on purpose."""
    repo = _repo(tmp_path)
    (repo / "keep.pyc").write_bytes(b"\x00original")
    _git(repo, "add", "-f", "keep.pyc")
    _git(repo, "commit", "-q", "-m", "vendored pyc")

    ws = Workspace(str(repo), "main", "T2")
    await ws.create()
    (ws.path / "keep.pyc").write_bytes(b"\x00modified")
    await ws.commit_all("touch tracked artifact")

    changed = _git(ws.path, "diff", "--name-only", f"{ws.review_base}..HEAD")
    assert "keep.pyc" in changed


@pytest.mark.asyncio
async def test_artifacts_only_means_nothing_to_commit(tmp_path):
    """A round that produced only bytecode has produced no change to review."""
    def build(wt):
        (wt / "__pycache__").mkdir()
        (wt / "__pycache__" / "m.cpython-312.pyc").write_bytes(b"\x00")

    assert await _staged_change(tmp_path, build) == []
