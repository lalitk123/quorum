"""Judgment-log read/append tests, including the mtime/size read cache."""

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
