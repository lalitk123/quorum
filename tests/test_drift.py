"""Reviewer-drift (CUSUM) tests. Deterministic; the trajectories are hand-checkable.

With p0=0.05, p1=0.5, threshold=4.0:
  a miss adds  ln(0.5/0.05)   = ln(10)     ≈ +2.3026
  a catch adds ln(0.5/0.95)   = ln(0.5263) ≈ -0.6419  (floored at 0)
So two consecutive misses (≈4.605) cross 4.0; a lone miss (≈2.30) does not.
"""

import pytest

from agent_team.drift import SeatDrift, bernoulli_cusum, detect_drift


# ---- CUSUM math --------------------------------------------------------------

def test_healthy_stream_never_alarms():
    traj, peak, ai = bernoulli_cusum([0] * 8)
    assert ai is None
    assert peak == 0.0
    assert traj[-1] == 0.0


def test_two_consecutive_misses_alarm():
    _, _, ai = bernoulli_cusum([1, 1, 1])
    assert ai == 1                       # crosses 4.0 on the second miss


def test_lone_miss_does_not_alarm_and_decays():
    traj, peak, ai = bernoulli_cusum([0, 0, 1, 0, 0])
    assert ai is None
    assert peak == pytest.approx(2.302585, abs=1e-4)   # single miss height
    assert traj[-1] < peak                             # catches decayed it back down


def test_drift_after_a_healthy_run_is_located():
    # 10 clean catches, then the seat degrades and starts missing.
    _, _, ai = bernoulli_cusum([0] * 10 + [1] * 4)
    assert ai == 11                      # first two misses are obs 10 and 11


def test_threshold_controls_sensitivity():
    stream = [1, 1]
    assert bernoulli_cusum(stream, threshold=4.0)[2] == 1
    assert bernoulli_cusum(stream, threshold=100.0)[2] is None


# ---- End-to-end from a judgment log -----------------------------------------

def _canary(seat, verdict, ts):
    return {"seat": seat, "verdict": verdict, "label": "defect",
            "source": "canary", "timestamp": ts}


def test_detect_drift_flags_a_degrading_seat_only():
    rows = []
    # "steady" always catches the canary; "sliding" catches then collapses.
    for i in range(12):
        ts = f"2026-07-08T00:{i:02d}:00+00:00"
        rows.append(_canary("steady", "reject", ts))
        rows.append(_canary("sliding", "reject" if i < 6 else "approve", ts))
    results = detect_drift(rows)
    assert results["steady"].alarm is False
    assert results["sliding"].alarm is True
    assert results["sliding"].alarm_index >= 6        # alarm lands in the degraded region
    assert results["sliding"].miss_rate == pytest.approx(0.5)


def test_timestamps_order_the_stream_not_file_order():
    # File order interleaves misses with catches, but timestamps put all the
    # misses last, where they should trigger the alarm.
    rows = [
        _canary("s", "approve", "2026-07-08T00:10:00+00:00"),
        _canary("s", "reject", "2026-07-08T00:01:00+00:00"),
        _canary("s", "approve", "2026-07-08T00:11:00+00:00"),
        _canary("s", "reject", "2026-07-08T00:02:00+00:00"),
    ]
    d = detect_drift(rows)["s"]
    # chronological stream is [catch, catch, miss, miss] -> alarms on the 4th obs.
    assert d.alarm is True and d.alarm_index == 3


def test_non_canary_sources_are_ignored():
    rows = [{"seat": "s", "verdict": "approve", "label": "defect",
             "source": "gold", "timestamp": "2026-07-08T00:00:00+00:00"}] * 5
    assert detect_drift(rows) == {}


def test_no_observations_is_empty():
    assert detect_drift([]) == {}
