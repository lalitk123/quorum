"""Self-benchmarking corpus tests (idea #9)."""

from agent_team.corpus import corpus_stats, record_production_outcome
from agent_team.judgments import load_judgments


def test_corpus_stats_counts_sources_labels_and_seats():
    rows = [
        {"seat": "security", "verdict": "reject", "label": "defect", "source": "gold"},
        {"seat": "security", "verdict": "approve", "label": "clean", "source": "gold"},
        {"seat": "qa", "verdict": "reject", "label": "defect", "source": "canary"},
        {"seat": "qa", "verdict": "approve", "label": None, "source": "production"},
    ]
    s = corpus_stats(rows)
    assert s.total == 4 and s.labeled == 3 and s.unlabeled == 1
    assert s.by_source == {"gold": 2, "canary": 1, "production": 1}
    assert s.per_seat_n["security"] == 2 and s.per_seat_n["qa"] == 1
    assert s.mean_sens_ci_width is not None and 0 < s.mean_sens_ci_width <= 1


def test_more_data_narrows_the_power_interval():
    few = [{"seat": "s", "verdict": "reject", "label": "defect", "source": "gold"}]
    many = [{"seat": "s", "verdict": "reject", "label": "defect", "source": "gold"}
            for _ in range(200)]
    assert corpus_stats(many).mean_sens_ci_width < corpus_stats(few).mean_sens_ci_width


def test_record_production_outcome_appends_labeled_rows(tmp_path):
    log = str(tmp_path / "j.jsonl")
    record_production_outcome(
        log, case_id="PR-42",
        seat_verdicts={"security": "approve", "qa": "reject"},
        was_defect=True, timestamp="2026-07-08T00:00:00+00:00")
    rows = load_judgments(log)
    assert len(rows) == 2
    assert all(r["label"] == "defect" and r["source"] == "production" for r in rows)
    assert {r["seat"] for r in rows} == {"security", "qa"}
