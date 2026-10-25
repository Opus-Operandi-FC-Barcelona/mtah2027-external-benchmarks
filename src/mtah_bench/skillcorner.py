"""SkillCorner open data (broadcast tracking, 10 fps) -> common npz.

Per match folder ``data/skillcorner/<id>/``: ``<id>_match.json``,
``<id>_tracking_extrapolated.jsonl``, ``<id>_phases_of_play.csv``.
Positions are metres with the origin moved to the pitch corner (source: centre origin).
Labelling rules: docs/LABELLING_RULES.md (SkillCorner section).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .convert import TARGET_FPS, _save, remap_bounds, resample_indices
from .labels import A, D, H, X

SRC_FPS = 10.0


def derive_phase_labels(
    period: np.ndarray, observed: np.ndarray, phases: list[tuple[int, int, int]]
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Labels at the source rate.

    ``period`` int[N] (0 = outside a period), ``observed`` bool[N] (players tracked on
    screen), ``phases`` list of (first_frame, last_frame_incl, team 0 home/1 away).
    Inside a period: unobserved -> 'X' (no information, camera away from the pitch);
    observed and inside a phase -> team in possession; observed and between phases -> 'D'
    (SkillCorner phases are contiguous while play is live, gaps are interruptions).
    """
    n = len(period)
    owner = np.full(n, -1, dtype=np.int8)
    for lo, hi, team in phases:
        owner[max(lo, 0) : min(hi, n - 1) + 1] = team
    gt = np.full(n, X, dtype="<U1")
    live = (period > 0) & observed
    gt[live & (owner == 0)] = H
    gt[live & (owner == 1)] = A
    gt[live & (owner < 0)] = D
    bounds = []
    for p in np.unique(period[period > 0]):
        idx = np.flatnonzero(period == p)
        bounds.append((int(idx[0]), int(idx[-1])))
    return gt, bounds


def load_tracking(jsonl: Path, player_ids: list[int], L: float, W: float):
    slot = {pid: k for k, pid in enumerate(player_ids)}
    rows = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    n = len(rows)
    xy = np.full((n, len(player_ids), 2), np.nan, dtype=np.float32)
    det = np.zeros((n, len(player_ids)), dtype=bool)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    per = np.zeros(n, dtype=np.int16)
    observed = np.zeros(n, dtype=bool)
    fid = np.empty(n, dtype=np.int64)
    for i, r in enumerate(rows):
        fid[i] = r["frame"]
        per[i] = r["period"] or 0
        b = r["ball_data"]
        if b and b.get("x") is not None:
            ball[i] = (b["x"] + L / 2, b["y"] + W / 2)
        pd_ = r["player_data"]
        observed[i] = len(pd_) > 0
        for d in pd_:
            k = slot.get(d["player_id"])
            if k is not None and d["x"] is not None:
                xy[i, k] = (d["x"] + L / 2, d["y"] + W / 2)
                det[i, k] = bool(d["is_detected"])
    return fid, per, observed, xy, det, ball


def convert_skillcorner(data_dir: Path, match_id: str, out_dir: Path) -> Path:
    d = Path(data_dir) / "skillcorner" / str(match_id)
    meta = json.loads((d / f"{match_id}_match.json").read_text(encoding="utf-8"))
    L, W = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id, away_id = meta["home_team"]["id"], meta["away_team"]["id"]
    plist = sorted(meta["players"], key=lambda p: (p["team_id"] != home_id, p["id"]))
    pids = [p["id"] for p in plist]
    tos = np.array([0 if p["team_id"] == home_id else 1 for p in plist], dtype=np.int8)
    fid, per, observed, xy, det, ball = load_tracking(
        d / f"{match_id}_tracking_extrapolated.jsonl", pids, L, W
    )
    used = ~np.isnan(xy[:, :, 0]).all(axis=0)
    xy, tos = xy[:, used], tos[used]
    det = det[:, used]
    names = np.array([f"{'H' if t == 0 else 'A'}_{p}" for t, p, u in zip(tos, np.array(pids)[used], used)])
    ph = pd.read_csv(d / f"{match_id}_phases_of_play.csv")
    team = ph["team_in_possession_id"].map({home_id: 0, away_id: 1})
    phases = [(int(a), int(b), int(t)) for a, b, t in zip(ph.frame_start, ph.frame_end, team) if t == t]
    gt, bounds = derive_phase_labels(per, observed, phases)
    # keep only the span of the two periods
    lo, hi = bounds[0][0], bounds[-1][1]
    sl = slice(lo, hi + 1)
    xy, ball, gt, fid, det, observed = xy[sl], ball[sl], gt[sl], fid[sl], det[sl], observed[sl]
    bounds = [(a - lo, b - lo) for a, b in bounds]
    idx = resample_indices(len(fid), SRC_FPS, TARGET_FPS)
    bounds = remap_bounds(bounds, idx)
    path = _save(Path(out_dir) / f"skillcorner_{match_id}.npz", xy, tos, ball, L, W, bounds, gt, fid, names, idx)
    z = dict(np.load(path, allow_pickle=False))
    z["players_detected"] = det[idx]
    z["observed"] = observed[idx]
    np.savez_compressed(path, **z)
    return path

SC_MATCHES = "1874553 1886347 1899585 1925299 1927964 1953632 1959846 1986691 1996435 1996436 2006229 2006363 2007448 2007721 2010085 2011166 2013725 2015213 2016236 2017461".split()
