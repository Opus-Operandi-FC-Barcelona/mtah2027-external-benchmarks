import numpy as np

from mtah_bench.skillcorner import derive_phase_labels


def test_phase_labels_tiny():
    period = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    observed = np.array([0, 1, 1, 1, 1, 1, 0, 0, 1, 0], dtype=bool)
    phases = [(1, 2, 0), (3, 4, 1)]  # home then away; frame 5 is a gap
    gt, bounds = derive_phase_labels(period, observed, phases)
    assert "".join(gt) == "XHHAADXXDX"
    assert bounds == [(1, 8)]


def test_observed_gap_inside_phase_is_unknown_not_dead():
    period = np.ones(6, dtype=int)
    observed = np.array([1, 1, 0, 0, 1, 1], dtype=bool)
    gt, _ = derive_phase_labels(period, observed, [(0, 5, 0)])
    assert "".join(gt) == "HHXXHH"


def test_two_periods_bounds():
    period = np.array([1, 1, 0, 2, 2])
    gt, bounds = derive_phase_labels(period, np.ones(5, bool), [(0, 1, 1), (3, 4, 0)])
    assert "".join(gt) == "AAXHH" and bounds == [(0, 1), (3, 4)]
