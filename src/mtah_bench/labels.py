"""Algorithmic possession labelling rules (pure functions, no I/O).

Labels: 'H' home possession, 'A' away possession, 'D' dead ball, 'X' outside the
evaluated period. See docs/LABELLING_RULES.md for the rationale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

H, A, D, X = "H", "A", "D", "X"
TEAM_LABEL = {0: H, 1: A}

# ---------------------------------------------------------------- Metrica rules
# Events that give a team controlled possession of the ball at ``start_frame``.
CONTROL_TYPES = {"PASS", "SHOT", "CARRY", "RECOVERY", "SET PIECE"}
# RECOVERY subtypes that are deflections, not control.
NON_CONTROL_SUBTYPE_TOKENS = {"RECOVERY": {"BLOCKED"}}
# Events after which the ball is dead until the next restart.
STOP_TYPES = {"BALL OUT", "FAULT RECEIVED", "CARD"}
# (type, subtype-token) pairs that stop play.
STOP_TYPE_TOKENS = {
    ("BALL LOST", "OFFSIDE"),
    ("BALL LOST", "HAND BALL"),
    ("BALL LOST", "END HALF"),
    ("CHALLENGE", "FAULT"),
    ("SHOT", "GOAL"),
    ("SHOT", "OUT"),
    ("PASS", "OFFSIDE"),
}
# Stops whose moment is the *end* of the event (ball leaves / offside is flagged on reception).
STOP_AT_END = {("SHOT", "GOAL"), ("SHOT", "OUT"), ("PASS", "OFFSIDE")}
# Events that restart play (take precedence over normal control: they also set the owner).
RESTART_TOKENS = {"GOAL KICK"}


def _tokens(subtype) -> set[str]:
    if subtype is None or (isinstance(subtype, float) and np.isnan(subtype)):
        return set()
    return {t.strip() for t in str(subtype).upper().split("-") if t.strip()}


def metrica_state_changes(events: pd.DataFrame) -> list[tuple[int, str]]:
    """Return an ordered list of ``(frame, state)`` changes for ONE period.

    ``events`` columns: team (0 home/1 away), type, subtype, start_frame, end_frame.
    """
    changes: list[tuple[int, int, str]] = []  # (frame, order, state)
    order = 0
    for r in events.itertuples(index=False):
        typ = str(r.type).upper()
        toks = _tokens(r.subtype)
        team_state = TEAM_LABEL[int(r.team)]
        start, end = int(r.start_frame), int(r.end_frame)
        is_control = typ in CONTROL_TYPES and not (
            toks & NON_CONTROL_SUBTYPE_TOKENS.get(typ, set())
        )
        if is_control:
            changes.append((start, order, team_state))
            order += 1
        stop_frame = None
        if typ in STOP_TYPES:
            # BALL OUT: start = last touch, end = ball crosses the line -> dead from the end.
            stop_frame = max(start, end) if typ == "BALL OUT" else start
        else:
            for (t, tok) in STOP_TYPE_TOKENS:
                if typ == t and tok in toks:
                    stop_frame = max(start, end) if (t, tok) in STOP_AT_END else start
                    break
        if stop_frame is not None:
            changes.append((stop_frame, order, D))
            order += 1
    changes.sort(key=lambda c: (c[0], c[1]))
    return [(f, s) for f, _, s in changes]


def derive_metrica_labels(
    events: pd.DataFrame, frame_ids: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Derive per-frame labels from Metrica-style events.

    ``frame_ids`` are the (sorted) source frame numbers of the tracking array.
    Returns ``(gt, period_bounds)`` with ``period_bounds`` as inclusive
    ``(first_idx, last_idx)`` array indices per period. A period spans from its
    first event start to its last event end; frames outside are 'X'.
    """
    frame_ids = np.asarray(frame_ids)
    n = len(frame_ids)
    gt = np.full(n, X, dtype="<U1")
    bounds: list[tuple[int, int]] = []

    def to_idx(f: int) -> int:
        return int(np.searchsorted(frame_ids, f, side="left"))

    for _, ev in events.groupby("period", sort=True):
        ev = ev.sort_values("start_frame", kind="stable")
        lo = to_idx(int(ev["start_frame"].min()))
        hi_frame = int(max(ev["end_frame"].max(), ev["start_frame"].max()))
        hi = min(n - 1, int(np.searchsorted(frame_ids, hi_frame, side="right")) - 1)
        if hi < lo:
            continue
        bounds.append((lo, hi))
        state = D
        cursor = lo
        for f, s in metrica_state_changes(ev):
            i = min(max(to_idx(f), lo), hi + 1)
            if i > cursor:
                gt[cursor:i] = state
                cursor = i
            state = s
        gt[cursor : hi + 1] = state
    return gt, bounds


# ----------------------------------------------------------------- IDSSE rules
def derive_ball_status_labels(
    alive: np.ndarray, owner: np.ndarray, period: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Labels from provided ball status and possession channels.

    ``alive`` bool[N]; ``owner`` int[N] (0 home, 1 away, -1 unknown); ``period`` int[N].
    Dead -> 'D'; alive -> team in possession (unknown owner: last known owner, or 'D'
    when none exists yet). Per period, frames before the first / after the last alive
    frame are 'X'.
    """
    n = len(alive)
    gt = np.full(n, X, dtype="<U1")
    bounds: list[tuple[int, int]] = []
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        alive_idx = idx[alive[idx]]
        if len(alive_idx) == 0:
            continue
        lo, hi = int(alive_idx[0]), int(alive_idx[-1])
        bounds.append((lo, hi))
        last = -1
        for i in range(lo, hi + 1):
            if owner[i] >= 0:
                last = int(owner[i])
            if not alive[i]:
                gt[i] = D
            elif last >= 0:
                gt[i] = TEAM_LABEL[last]
            else:
                gt[i] = D
    return gt, bounds


# ------------------------------------------------------------------- statistics
def spells(gt: np.ndarray) -> list[tuple[str, int, int]]:
    """Maximal runs of H or A (a D or X cuts a possession). Returns (label, start, end_incl)."""
    out = []
    n = len(gt)
    i = 0
    while i < n:
        if gt[i] in (H, A):
            j = i
            while j + 1 < n and gt[j + 1] == gt[i]:
                j += 1
            out.append((str(gt[i]), i, j))
            i = j + 1
        else:
            i += 1
    return out


def label_stats(gt: np.ndarray, fps: float = 25.0) -> dict:
    sp = spells(gt)
    evaluated = int(np.sum(gt != X))
    dur = np.array([(e - s + 1) / fps for _, s, e in sp]) if sp else np.array([])
    play_min = evaluated / fps / 60.0
    return {
        "frames": int(len(gt)),
        "evaluated_frames": evaluated,
        "possessions": len(sp),
        "possessions_per_5min": round(len(sp) / (evaluated / fps / 300.0), 2) if evaluated else 0.0,
        "mean_possession_s": round(float(dur.mean()), 2) if len(dur) else 0.0,
        "median_possession_s": round(float(np.median(dur)), 2) if len(dur) else 0.0,
        "pct_H": round(100 * float(np.sum(gt == H)) / max(evaluated, 1), 2),
        "pct_A": round(100 * float(np.sum(gt == A)) / max(evaluated, 1), 2),
        "pct_D": round(100 * float(np.sum(gt == D)) / max(evaluated, 1), 2),
        "evaluated_minutes": round(play_min, 2),
    }
