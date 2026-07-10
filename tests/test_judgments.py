"""Judgment-log read/append tests, including the mtime/size read cache."""

import os

from agent_team.judgments import append_judgments, load_judgments


def test_load_judgments_reads_appended_rows(tmp_path):
    log = str(tmp_path / "j.jsonl")
    append_judgments(log, [{"seat": "security", "verdict": "reject", "label": "defect"}])
    rows = load_judgments(log)
    assert len(rows) == 1 and rows[0]["seat"] == "security"


def test_load_judgments_missing_file_is_empty(tmp_path):
    assert load_judgments(str(tmp_path / "missing.jsonl")) == []


def test_load_judgments_skips_corrupt_lines(tmp_path):
    log = tmp_path / "j.jsonl"
    log.write_text('{"seat": "qa", "verdict": "approve"}\nnot json\n\n')
    rows = load_judgments(str(log))
    assert len(rows) == 1 and rows[0]["seat"] == "qa"


def test_load_judgments_cache_reflects_appends_within_one_process(tmp_path):
    log = str(tmp_path / "j.jsonl")
    append_judgments(log, [{"seat": "qa", "verdict": "approve", "label": "clean"}])
    first = load_judgments(log)
    assert len(first) == 1

    # A cache keyed only on a stale mtime would miss this second row; the
    # size check must catch it even when mtime resolution is coarse.
    append_judgments(log, [{"seat": "security", "verdict": "reject", "label": "defect"}])
    second = load_judgments(log)
    assert len(second) == 2
    assert {r["seat"] for r in second} == {"qa", "security"}


def test_load_judgments_cache_is_per_path(tmp_path):
    log_a = str(tmp_path / "a.jsonl")
    log_b = str(tmp_path / "b.jsonl")
    append_judgments(log_a, [{"seat": "qa", "verdict": "approve", "label": "clean"}])
    append_judgments(log_b, [{"seat": "security", "verdict": "reject", "label": "defect"}])
    assert len(load_judgments(log_a)) == 1
    assert len(load_judgments(log_b)) == 1


def test_cache_actually_short_circuits_the_reparse(tmp_path):
    """Rewrite the file behind the cache's back, preserving (mtime, size)."""
    log = tmp_path / "j.jsonl"
    log.write_text('{"seat": "qa", "verdict": "approve"}\n')
    assert load_judgments(str(log))[0]["seat"] == "qa"

    stat = log.stat()
    log.write_text('{"seat": "zz", "verdict": "approve"}\n')   # identical byte length
    os.utime(log, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert log.stat().st_size == stat.st_size

    assert load_judgments(str(log))[0]["seat"] == "qa"          # served from cache


# A cached row is evidence: reliability estimates, the trust score, and the signed
# receipts all read it. Handing callers the cache itself lets one of them rewrite
# what the next one sees — a judgment no reviewer ever made.

def test_mutating_the_returned_list_does_not_poison_the_cache(tmp_path):
    log = str(tmp_path / "j.jsonl")
    append_judgments(log, [{"seat": "qa", "verdict": "reject", "label": "defect"}])

    first = load_judgments(log)
    first.append({"seat": "INJECTED", "verdict": "approve", "label": "defect"})

    second = load_judgments(log)
    assert len(second) == 1
    assert not any(r["seat"] == "INJECTED" for r in second)


def test_mutating_a_returned_row_does_not_poison_the_cache(tmp_path):
    log = str(tmp_path / "j.jsonl")
    append_judgments(log, [{"seat": "qa", "verdict": "reject", "label": "defect"}])

    load_judgments(log)[0]["verdict"] = "approve"       # flip it fail-open

    assert load_judgments(log)[0]["verdict"] == "reject"


def test_mutating_a_returned_condition_does_not_poison_the_cache(tmp_path):
    """`condition` is the one nested field; a shallow copy alone would still alias it."""
    log = str(tmp_path / "j.jsonl")
    append_judgments(log, [{"seat": "qa", "verdict": "reject", "label": "defect",
                            "condition": {"risk": "high"}}])

    load_judgments(log)[0]["condition"]["risk"] = "low"

    assert load_judgments(log)[0]["condition"] == {"risk": "high"}


def test_each_load_returns_independent_objects(tmp_path):
    log = str(tmp_path / "j.jsonl")
    append_judgments(log, [{"seat": "qa", "verdict": "reject", "condition": {"risk": "high"}}])

    a, b = load_judgments(log), load_judgments(log)
    assert a == b
    assert a is not b and a[0] is not b[0] and a[0]["condition"] is not b[0]["condition"]
