import numpy as np
import pandas as pd

from mtah_bench.convert import resample_indices
from mtah_bench.labels import (
    derive_ball_status_labels,
    derive_metrica_labels,
    label_stats,
    spells,
)


def ev(rows):
    return pd.DataFrame(rows, columns=["team", "type", "subtype", "period", "start_frame", "end_frame"])


def run(rows, n=200):
    gt, bounds = derive_metrica_labels(ev(rows), np.arange(n))
    return "".join(gt), bounds


def test_basic_possession_and_x_outside_period():
    s, b = run([
        (0, "SET PIECE", "KICK OFF", 1, 10, 10),
        (0, "PASS", None, 1, 12, 20),
        (1, "RECOVERY", "INTERCEPTION", 1, 30, 30),
        (1, "PASS", None, 1, 31, 40),
    ], 60)
    assert b == [(10, 40)]
    assert s[:10] == "X" * 10 and s[41:] == "X" * 19
    assert s[10:30] == "H" * 20
    assert s[30:41] == "A" * 11


def test_ball_out_makes_dead_until_restart():
    s, _ = run([
        (0, "PASS", None, 1, 0, 10),
        (0, "BALL OUT", None, 1, 10, 25),     # dead from the frame the ball crosses the line
        (1, "SET PIECE", "THROW IN", 1, 60, 60),
        (1, "PASS", None, 1, 62, 70),
    ], 80)
    assert s[0:25] == "H" * 25
    assert s[25:60] == "D" * 35
    assert s[60:71] == "A" * 11


def test_h_d_h_is_two_possessions():
    s, _ = run([
        (0, "PASS", None, 1, 0, 9),
        (0, "FAULT RECEIVED", None, 1, 20, 20),
        (0, "SET PIECE", "FREE KICK", 1, 40, 40),
        (0, "PASS", None, 1, 41, 50),
    ], 60)
    gt = np.array(list(s))
    assert [(l) for l, *_ in spells(gt)] == ["H", "H"]
    assert s[20:40] == "D" * 20


def test_challenges_and_blocked_recovery_do_not_change_owner():
    s, _ = run([
        (0, "PASS", None, 1, 0, 10),
        (1, "CHALLENGE", "GROUND-LOST", 1, 12, 12),
        (1, "RECOVERY", "BLOCKED", 1, 14, 14),
        (1, "BALL LOST", "REFEREE HIT", 1, 15, 15),
        (0, "PASS", None, 1, 16, 30),
    ], 40)
    assert set(s[0:31]) == {"H"}


def test_goal_then_kick_off():
    s, _ = run([
        (0, "SHOT", "ON TARGET-GOAL", 1, 0, 20),
        (1, "SET PIECE", "KICK OFF", 1, 100, 100),
        (1, "PASS", None, 1, 101, 120),
    ], 130)
    assert s[0:20] == "H" * 20
    assert s[20:100] == "D" * 80
    assert s[100] == "A"


def test_offside_pass_dead_at_end_and_goal_kick_restarts():
    s, _ = run([
        (0, "PASS", "OFFSIDE", 1, 0, 15),
        (1, "PASS", "GOAL KICK", 1, 50, 60),
    ], 70)
    assert s[0:15] == "H" * 15
    assert s[15:50] == "D" * 35
    assert s[50:61] == "A" * 11


def test_two_periods_get_separate_bounds_and_gap_is_x():
    s, b = run([
        (0, "SET PIECE", "KICK OFF", 1, 5, 5), (0, "PASS", None, 1, 6, 20),
        (1, "SET PIECE", "KICK OFF", 2, 100, 100), (1, "PASS", None, 2, 101, 120),
    ], 130)
    assert b == [(5, 20), (100, 120)]
    assert set(s[21:100]) == {"X"}


def test_frame_ids_offset_mapping():
    fid = np.arange(1000, 1100)
    gt, b = derive_metrica_labels(ev([(0, "PASS", None, 1, 1010, 1020)]), fid)
    assert b == [(10, 20)] and gt[10] == "H" and gt[9] == "X"


def test_ball_status_labels_dead_alive_and_owner_persistence():
    alive = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0], bool)
    owner = np.array([-1, -1, 0, 0, 1, 1, 1, -1, 1, 1, 1])
    period = np.ones(11, int)
    gt, b = derive_ball_status_labels(alive, owner, period)
    assert "".join(gt) == "XXHHADDAAXX"
    assert b == [(2, 8)]


def test_ball_status_owner_unknown_uses_last_known():
    alive = np.ones(5, bool)
    owner = np.array([-1, 1, -1, -1, 0])
    gt, _ = derive_ball_status_labels(alive, owner, np.ones(5, int))
    assert "".join(gt) == "DAAAH"


def test_label_stats():
    gt = np.array(list("XXHHHHDDAAXX"))
    st = label_stats(gt, fps=1.0)
    assert st["possessions"] == 2 and st["mean_possession_s"] == 3.0
    assert st["evaluated_frames"] == 8 and st["pct_D"] == 25.0


def test_resample_indices():
    assert (resample_indices(5, 25) == np.arange(5)).all()
    idx = resample_indices(51, 50)
    assert len(idx) == 26 and idx[1] == 2
