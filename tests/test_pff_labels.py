import numpy as np

from mtah_bench.pff import (
    derive_pff_labels,
    derive_pff_state_changes,
    normalise_events,
    pff_pitch,
    resample_by_time,
)


def row(typ, t0, t1=None, home=None, sp=None, period=1, foul=None, nonevent=False):
    return {
        "startTime": t0,
        "endTime": t1,
        "gameEvents": {
            "gameEventType": typ, "setpieceType": sp, "homeTeam": home, "period": period,
            "initialNonEvent": nonevent,
        },
        "fouls": {"foulType": foul, "onFieldFoulOutcomeType": "N" if foul else None},
    }


def frames(n, fps=10.0, period=1):
    return np.arange(n) / fps + 100.0, np.full(n, period)


def labels(raw, n=40):
    t, p = frames(n)
    gt, bounds, info = derive_pff_labels(normalise_events(raw), t, p)
    return "".join(gt), bounds, info


def test_possession_out_and_restart():
    raw = [
        row("FIRSTKICKOFF", 100.0, 100.0, home=True, sp="K"),
        row("OTB", 100.5, 101.0, home=True, sp="O"),
        row("OTB", 101.5, 101.5, home=False, sp="O"),  # opponent touch -> A
        row("OUT", 102.0),  # ball dead from 102.0
        row("OTB", 103.0, 103.0, home=True, sp="T"),  # throw-in restarts at 103.0
        row("END", 103.9),
    ]
    s, bounds, info = labels(raw, n=45)
    # frame i is at 100 + i/10
    assert s[0] == "H" and s[14] == "H" and s[15] == "A" and s[19] == "A"
    assert s[20:30] == "D" * 10
    assert s[30:40] == "H" * 10  # END at 103.9 is the last labelled frame
    assert s[40:] == "X" * 5
    assert info["out_events"] == 1 and info["implied_stops"] == 0


def test_period_bounds_and_x_outside():
    raw = [row("FIRSTKICKOFF", 100.5, 100.5, home=True, sp="K"), row("END", 102.0)]
    s, bounds, _ = labels(raw, n=40)
    assert s[:5] == "XXXXX" and s[5] == "H" and bounds == [(5, 20)]
    assert s[21:] == "X" * 19


def test_restart_without_out_infers_dead_from_previous_touch_end():
    raw = [
        row("FIRSTKICKOFF", 100.0, 100.0, home=True, sp="K"),
        row("OTB", 100.5, 101.0, home=True, sp="O"),
        row("OTB", 103.0, 103.0, home=False, sp="F"),  # free kick, no OUT event
        row("END", 105.0),
    ]
    s, _, info = labels(raw, n=50)
    assert s[9] == "H" and s[10] == "D" and s[29] == "D" and s[30] == "A"
    assert info["implied_stops"] == 1


def test_given_foul_marks_stop_but_advantage_does_not():
    raw = [
        row("FIRSTKICKOFF", 100.0, 100.0, home=True, sp="K"),
        row("OTB", 100.5, 101.0, home=True, sp="O", foul="I"),  # foul given at 101.0
        row("OTB", 103.0, 103.0, home=False, sp="F"),
        row("OTB", 103.5, 104.0, home=False, sp="O", foul="A"),  # advantage: no stop
        row("OTB", 104.5, 104.5, home=True, sp="O"),
        row("END", 105.0),
    ]
    s, _, _ = labels(raw, n=50)
    assert s[9] == "H" and s[10] == "D" and s[29] == "D" and s[30] == "A"
    assert s[36] == "A" and s[45] == "H"


def test_disallowed_events_ignored_and_unknown_team_skipped():
    raw = [
        row("FIRSTKICKOFF", 100.0, 100.0, home=True, sp="K"),
        row("OTB", 100.5, 100.5, home=False, sp="O", nonevent=True),
        row("OTB", 101.0, 101.0, home=None, sp="O"),
        row("END", 102.0),
    ]
    s, _, _ = labels(raw, n=30)
    assert set(s[:21]) == {"H"}


def test_second_period_independent():
    raw = [
        row("FIRSTKICKOFF", 100.0, 100.0, home=True, sp="K", period=1),
        row("END", 100.9, period=1),
        row("SECONDKICKOFF", 200.0, 200.0, home=False, sp="K", period=2),
        row("END", 200.9, period=2),
    ]
    t = np.r_[np.arange(10) / 10 + 100.0, np.arange(10) / 10 + 200.0]
    p = np.r_[np.ones(10), np.full(10, 2)].astype(int)
    gt, bounds, _ = derive_pff_labels(normalise_events(raw), t, p)
    assert "".join(gt) == "H" * 10 + "A" * 10 and bounds == [(0, 9), (10, 19)]
    lo, hi, ch, _ = derive_pff_state_changes(normalise_events(raw), 2)
    assert lo == 200.0 and hi == 200.9 and ch == [(200.0, "A")]


def test_resample_by_time_per_period_grid():
    t = np.r_[np.arange(0, 1.0, 1 / 30), 50 + np.arange(0, 1.0, 1 / 30)]
    p = np.r_[np.ones(30), np.full(30, 2)].astype(int)
    idx = resample_by_time(t, p)
    assert len(idx) == 25 + 25 and idx[0] == 0 and idx[25] == 30 and np.all(np.diff(idx[:25]) >= 1)
    assert idx[24] in (28, 29)


def test_pitch_prefers_open_ended_entry():
    meta = {"stadium": {"pitches": [{"length": 100, "width": 60, "endDate": "2020-01-01"},
                                    {"length": 105.0, "width": 68.0, "endDate": None}]}}
    assert pff_pitch(meta) == (105.0, 68.0) and pff_pitch({}) == (105.0, 68.0)
