"""Clean-sandbox verification tests against real (throwaway) git repos.

Fully deterministic — no API keys. Each test builds a tiny git repo, commits a
base, branches a change, and drives `verify_in_sandbox` end to end, running the
suite as a real subprocess in the freshly created verification worktree.
"""

import subprocess
import sys
import textwrap

import pytest

from agent_team.sandbox import verify_in_sandbox

TEST_CMD = f"{sys.executable} -m pytest -q -p no:cacheprovider"


def _git(repo, *args):
    proc = subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _write(repo, rel, content):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))


def _init(repo):
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.test")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")


def _commit(repo, msg):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _base_repo(repo):
    """A repo whose existing suite passes, on a fresh feature branch."""
    _init(repo)
    _write(repo, "mathx.py", "def add(a, b):\n    return a + b\n")
    _write(repo, "tests/test_add.py",
           "from mathx import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    base = _commit(repo, "base")
    _git(repo, "checkout", "-q", "-b", "agent/T1")
    return base


@pytest.mark.asyncio
async def test_teeth_present_for_a_real_test(tmp_path):
    repo = tmp_path / "repo"
    base = _base_repo(repo)
    # Feature: modify source + add a test that actually exercises it.
    _write(repo, "mathx.py",
           "def add(a, b):\n    return a + b\n\ndef triple(x):\n    return x * 3\n")
    _write(repo, "tests/test_triple.py",
           "from mathx import triple\n\ndef test_triple():\n    assert triple(2) == 6\n")
    _commit(repo, "add triple + real test")
    diff = _git(repo, "diff", f"{base}..HEAD")

    report = await verify_in_sandbox(str(repo), "agent/T1", base, diff, TEST_CMD)
    assert report.reproduced is True
    assert report.teeth is True          # reverting source makes the new test fail
    assert report.hard_failure == ""
    assert report.findings == []


@pytest.mark.asyncio
async def test_teeth_absent_for_a_gamed_test(tmp_path):
    repo = tmp_path / "repo"
    base = _base_repo(repo)
    # Feature: real source change, but a vacuous test that never touches it.
    _write(repo, "mathx.py",
           "def add(a, b):\n    return a + b\n\ndef triple(x):\n    return x * 3\n")
    _write(repo, "tests/test_triple.py",
           "def test_triple():\n    assert True\n")
    _commit(repo, "add triple + vacuous test")
    diff = _git(repo, "diff", f"{base}..HEAD")

    report = await verify_in_sandbox(str(repo), "agent/T1", base, diff, TEST_CMD)
    assert report.reproduced is True
    assert report.teeth is False
    assert any("TEETH CHECK FAILED" in f for f in report.findings)


@pytest.mark.asyncio
async def test_reproduction_fails_when_change_needs_untracked_state(tmp_path):
    repo = tmp_path / "repo"
    base = _base_repo(repo)
    # A test that reads a file the worker never committed: passes only in a dirty
    # worktree, fails in a clean checkout.
    _write(repo, "tests/test_secret.py",
           "def test_secret():\n"
           "    with open('secret.txt') as fh:\n"
           "        assert fh.read().strip() == 'ok'\n")
    _commit(repo, "add test depending on untracked file")
    # The untracked file lives only in the worker's worktree, not the commit.
    (repo / "secret.txt").write_text("ok\n")
    diff = _git(repo, "diff", f"{base}..HEAD")

    report = await verify_in_sandbox(str(repo), "agent/T1", base, diff, TEST_CMD)
    assert report.reproduced is False
    assert "clean checkout" in report.hard_failure
    assert report.teeth is None


@pytest.mark.asyncio
async def test_new_source_file_teeth(tmp_path):
    repo = tmp_path / "repo"
    base = _base_repo(repo)
    # Brand-new source module + a test importing it: reverting deletes the module,
    # so the test errors -> teeth present.
    _write(repo, "shiny.py", "def shine():\n    return 'sparkle'\n")
    _write(repo, "tests/test_shiny.py",
           "from shiny import shine\n\ndef test_shine():\n    assert shine() == 'sparkle'\n")
    _commit(repo, "new module + test")
    diff = _git(repo, "diff", f"{base}..HEAD")

    report = await verify_in_sandbox(str(repo), "agent/T1", base, diff, TEST_CMD)
    assert report.reproduced is True
    assert report.teeth is True


@pytest.mark.asyncio
async def test_no_tests_flagged(tmp_path):
    repo = tmp_path / "repo"
    base = _base_repo(repo)
    # Source-only change with no accompanying test.
    _write(repo, "mathx.py",
           "def add(a, b):\n    return a + b\n\ndef triple(x):\n    return x * 3\n")
    _commit(repo, "source only, no test")
    diff = _git(repo, "diff", f"{base}..HEAD")

    report = await verify_in_sandbox(str(repo), "agent/T1", base, diff, TEST_CMD)
    assert report.reproduced is True
    assert report.teeth is None
    assert any("no test files" in f for f in report.findings)
