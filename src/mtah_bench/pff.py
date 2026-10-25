"""PFF FC World Cup 2022 (tracking + event data) -> common npz.

GATED dataset (the user accepted PFF's terms and downloaded the files). Layout, after
reorganising the Google-Drive zips (see docs/DATASETS.md):

    data/pff/<game_id>/metadata.json       (Metadata/<game_id>.json, one-element list)
    data/pff/<game_id>/rosters.json        (Rosters/<game_id>.json)
    data/pff/<game_id>/tracking.jsonl.bz2  (Tracking Data/<game_id>.jsonl.bz2, plain .jsonl also ok)
    data/pff/<game_id>/events.json         (Event Data/<game_id>.json, top-level = latest release)

Parsed with ``json``/``bz2`` only (no kloppy: the 3.18 reader neither exposes the foul/out
fields nor the source frame times we need). Findings on the real files (v4.1.0 tracking):

* 29.97 fps, centred metres (x along the length, increasing left to right as seen by the main
  camera; ball also has z). Pitch size from ``metadata.stadium.pitches`` (105 x 68 everywhere).
* One JSON object per frame; players listed by *jersey number* per team (no player id), only the
  players on the pitch; ``visibility`` is VISIBLE or ESTIMATED. The ball list is empty when
  untracked and VISIBLE/ESTIMATED otherwise.
* The file starts at the kick-off of each period and ends at its last frame (contiguous
  ``frameNum``), with no half-time frames; extra time (periods 3, 4) exists in knock-out games.
* ``videoTimeMs`` of a frame equals the ``startTime`` of the event data (same video clock).

Labelling rules: docs/LABELLING_RULES.md (PFF section).
"""
from __future__ import annotations

import bz2
import json
from pathlib import Path

import numpy as np

from .convert import TARGET_FPS, _save, remap_bounds
from .labels import A, D, H, X, TEAM_LABEL

KICKOFFS = {"FIRSTKICKOFF": 1, "SECONDKICKOFF": 2, "THIRDKICKOFF": 3, "FOURTHKICKOFF": 4}
MAX_SLOTS_PER_TEAM = 30


def pff_game_dirs(data_dir: Path) -> list[str]:
    base = Path(data_dir) / "pff"
    if not base.exists():
        return []
    return sorted(
        p.name
        for p in base.iterdir()
        if p.is_dir()
        and (p / "metadata.json").exists()
        and ((p / "tracking.jsonl.bz2").exists() or (p / "tracking.jsonl").exists())
        and (p / "events.json").exists()
    )


# ----------------------------------------------------------------------------- events
def normalise_events(raw: list[dict], home_team_id: str | None = None) -> list[dict]:
    """events.json rows -> flat dicts ``type, setpiece, team (0 home/1 away/-1), t0, t1,
    period, foul_stop`` sorted by start time (stable). Disallowed events are dropped.

    ``foul_stop``: a foul was *given* (foulType 'I' and on-field outcome is not 'F' = no foul).
    Advantages ('A') and missed infringements ('M') do not stop play.
    """
    out = []
    for k, e in enumerate(raw):
        g = e.get("gameEvents") or {}
        if g.get("initialNonEvent"):
            continue
        ht = g.get("homeTeam")
        team = -1 if ht is None else (0 if ht else 1)
        if team < 0 and g.get("teamId") is not None and home_team_id is not None:
            team = 0 if str(int(g["teamId"])) == str(home_team_id) else 1
        f = e.get("fouls") or {}
        t0 = e.get("startTime")
        if t0 is None:
            continue
        t1 = e.get("endTime")
        out.append(
            {
                "type": g.get("gameEventType"),
                "setpiece": g.get("setpieceType"),
                "team": team,
                "t0": float(t0),
                "t1": float(t1) if t1 is not None else float(t0),
                "period": int(g.get("period") or 0),
                "foul_stop": f.get("foulType") == "I" and f.get("onFieldFoulOutcomeType") != "F",
                "_order": k,
            }
        )
    out.sort(key=lambda r: (r["t0"], r["_order"]))
    return out


def derive_pff_state_changes(events: list[dict], period: int) -> tuple[float, float, list[tuple[float, str]], dict]:
    """Replay the events of one period.

    Returns ``(t_lo, t_hi, changes, info)``; ``changes`` is the ordered list of
    ``(time_s, state)`` with state in H/A/D, starting at the kick-off. ``t_lo`` is the
    kick-off time (or first touch if absent), ``t_hi`` the END event (``inf`` if absent).
    """
    ev = [e for e in events if e["period"] == period]
    t_lo = next((e["t0"] for e in ev if KICKOFFS.get(e["type"]) == period), None)
    if t_lo is None:
        t_lo = next((e["t0"] for e in ev if e["type"] == "OTB"), None)
    ends = [e["t0"] for e in ev if e["type"] == "END"]
    t_hi = max(ends) if ends else float("inf")
    info = {"implied_stops": 0, "out_events": 0, "restarts": 0}
    changes: list[tuple[float, str]] = []
    if t_lo is None:
        return float("nan"), t_hi, changes, info
    state = D
    prev_end = t_lo
    pending_foul: float | None = None
    for e in ev:
        if e["t0"] < t_lo or e["t0"] > t_hi:
            continue
        if e["type"] == "OUT":
            info["out_events"] += 1
            pending_foul = None
            if state != D:
                changes.append((e["t0"], D))
                state = D
            continue
        if e["type"] not in ("OTB", *KICKOFFS) or e["team"] < 0:
            continue
        restart = e["type"] in KICKOFFS or e["setpiece"] not in (None, "O")
        if restart:
            info["restarts"] += 1
            if state != D and e["t0"] > t_lo:  # play was not stopped by an OUT event
                stop = pending_foul if pending_foul is not None else prev_end
                stop = min(max(stop, t_lo), e["t0"])
                changes.append((stop, D))
                info["implied_stops"] += 1
                state = D
        if e["setpiece"] in (None, "O") and e["type"] == "OTB":
            pending_foul = None
        new = TEAM_LABEL[e["team"]]
        if new != state:
            changes.append((e["t0"], new))
            state = new
        prev_end = max(prev_end, e["t1"])
        if e["foul_stop"]:
            pending_foul = max(e["t1"], e["t0"])
    return t_lo, t_hi, changes, info


def derive_pff_labels(
    events: list[dict], video_t: np.ndarray, period: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, int]], dict]:
    """Per-frame labels at the source rate. ``video_t`` float[N] seconds (video clock),
    ``period`` int[N]. Frames outside [kick-off, END] of each period are 'X'. Returns
    ``(gt, bounds, info)``; bounds are inclusive source indices."""
    n = len(video_t)
    gt = np.full(n, X, dtype="<U1")
    bounds: list[tuple[int, int]] = []
    info = {"implied_stops": 0, "out_events": 0, "restarts": 0}
    for p in sorted(int(v) for v in np.unique(period) if v > 0):
        idx = np.flatnonzero(period == p)
        t_lo, t_hi, changes, inf = derive_pff_state_changes(events, p)
        for k in info:
            info[k] += inf[k]
        if not np.isfinite(t_lo):
            continue
        tt = video_t[idx]
        lo = int(idx[0] + np.searchsorted(tt, t_lo, side="left"))
        hi = int(idx[0] + np.searchsorted(tt, t_hi, side="right") - 1)
        hi = min(hi, int(idx[-1]))
        if hi < lo:
            continue
        bounds.append((lo, hi))
        state = D
        cursor = lo
        for t, s in changes:
            i = int(idx[0] + np.searchsorted(tt, t, side="left"))
            i = min(max(i, lo), hi + 1)
            if i > cursor:
                gt[cursor:i] = state
                cursor = i
            state = s
        gt[cursor : hi + 1] = state
    return gt, bounds, info


# --------------------------------------------------------------------------- tracking
def _open_tracking(d: Path):
    trk = d / "tracking.jsonl.bz2"
    return bz2.open(trk, "rt", encoding="utf-8") if trk.exists() else (d / "tracking.jsonl").open(encoding="utf-8")


def load_tracking(d: Path, L: float, W: float):
    """Stream the tracking file once. Returns video_t, period, xy[N,S,2] (corner-origin
    metres), vis[N,S], ball[N,2], ball_vis[N], team_of_slot, jerseys[S]."""
    cap = 1 << 18
    xy = np.full((cap, 2 * MAX_SLOTS_PER_TEAM, 2), np.nan, dtype=np.float32)
    vis = np.zeros((cap, 2 * MAX_SLOTS_PER_TEAM), dtype=bool)
    ball = np.full((cap, 2), np.nan, dtype=np.float32)
    bvis = np.zeros(cap, dtype=bool)
    vt = np.empty(cap, dtype=np.float64)
    per = np.zeros(cap, dtype=np.int16)
    slots: dict[tuple[int, str], int] = {}
    n = 0
    hx, hy = L / 2, W / 2
    with _open_tracking(d) as f:
        for line in f:
            r = json.loads(line)
            if n == cap:
                cap *= 2
                xy = np.concatenate([xy, np.full_like(xy, np.nan)])
                vis = np.concatenate([vis, np.zeros_like(vis)])
                ball = np.concatenate([ball, np.full_like(ball, np.nan)])
                bvis = np.concatenate([bvis, np.zeros_like(bvis)])
                vt = np.concatenate([vt, np.empty_like(vt)])
                per = np.concatenate([per, np.zeros_like(per)])
            vt[n] = r["videoTimeMs"] / 1000.0
            per[n] = r["period"] or 0
            for team, key in ((0, "homePlayers"), (1, "awayPlayers")):
                for pl in r[key] or ():
                    if pl.get("x") is None:
                        continue
                    k = slots.setdefault((team, str(pl["jerseyNum"])), len(slots))
                    if k >= 2 * MAX_SLOTS_PER_TEAM:
                        continue
                    xy[n, k] = (pl["x"] + hx, pl["y"] + hy)
                    vis[n, k] = pl.get("visibility") == "VISIBLE"
            b = r["balls"]
            if b and b[0].get("x") is not None:
                ball[n] = (b[0]["x"] + hx, b[0]["y"] + hy)
                bvis[n] = b[0].get("visibility") == "VISIBLE"
            n += 1
    s = len(slots)
    order = sorted(slots, key=lambda tj: (tj[0], int(tj[1]) if tj[1].lstrip("-").isdigit() else 10**6, tj[1]))
    cols = [slots[k] for k in order]
    return (
        vt[:n], per[:n], xy[:n][:, cols], vis[:n][:, cols], ball[:n], bvis[:n],
        np.array([t for t, _ in order], dtype=np.int8), [j for _, j in order],
    ) if s else (vt[:n], per[:n], xy[:n, :0], vis[:n, :0], ball[:n], bvis[:n], np.zeros(0, np.int8), [])


def resample_by_time(video_t: np.ndarray, period: np.ndarray, fps: float = TARGET_FPS) -> np.ndarray:
    """Nearest source frame to a regular ``fps`` grid, period by period on the video clock
    (robust to dropped frames; the grid restarts at the first frame of every period)."""
    parts = []
    for p in sorted(int(v) for v in np.unique(period) if v > 0):
        idx = np.flatnonzero(period == p)
        t = video_t[idx]
        grid = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * fps)) + 1) / fps
        j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1) if len(t) > 1 else np.zeros(len(grid), int)
        left = np.maximum(j - 1, 0)
        pick = np.where(np.abs(t[left] - grid) <= np.abs(t[j] - grid), left, j)
        parts.append(idx[pick])
    return np.concatenate(parts) if parts else np.zeros(0, dtype=int)


def pff_pitch(meta: dict) -> tuple[float, float]:
    pitches = (meta.get("stadium") or {}).get("pitches") or []
    cur = [p for p in pitches if p.get("endDate") in (None, "")] or pitches
    if not cur:
        return 105.0, 68.0
    return float(cur[-1]["length"]), float(cur[-1]["width"])


def convert_pff(data_dir: Path, game_id: str, out_dir: Path) -> Path:
    d = Path(data_dir) / "pff" / game_id
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    meta = meta[0] if isinstance(meta, list) else meta
    roster = json.loads((d / "rosters.json").read_text(encoding="utf-8"))
    raw_ev = json.loads((d / "events.json").read_text(encoding="utf-8"))
    L, W = pff_pitch(meta)
    home_id = str(meta["homeTeam"]["id"])
    vt, per, xy, vis, ball, bvis, tos, jerseys = load_tracking(d, L, W)
    events = normalise_events(raw_ev, home_id)
    gt, bounds, info = derive_pff_labels(events, vt, per)
    idx = resample_by_time(vt, per)
    bounds = remap_bounds(bounds, idx)
    jersey_to_pid = {
        (0 if str(r["team"]["id"]) == home_id else 1, str(r["shirtNumber"])): str(r["player"]["id"])
        for r in roster
    }
    pids = np.array(
        [f"{'H' if t == 0 else 'A'}_{jersey_to_pid.get((int(t), j), 'j' + j)}" for t, j in zip(tos, jerseys)]
    )
    fid = np.arange(len(vt), dtype=np.int64)  # source row index
    path = _save(Path(out_dir) / f"pff_{game_id}.npz", xy, tos, ball, L, W, bounds, gt, fid, pids, idx)
    z = dict(np.load(path, allow_pickle=False))
    z["ball_visible"] = bvis[idx]
    z["players_visible"] = vis[idx]
    z["period"] = per[idx]
    z["fps_source"] = np.float64(meta.get("fps", 29.97))
    z["label_info"] = np.array([info["implied_stops"], info["out_events"], info["restarts"]], dtype=np.int64)
    np.savez_compressed(path, **z)
    return path


# ------------------------------------------------------------------------- reporting
IDSSE_RANGES = {"pct_D": (29.0, 52.0), "possessions": (314, 511), "mean_possession_s": (5.5, 11.0)}


def pff_extra_stats(npz_path: Path) -> dict:
    """Aggregate-only extras of a converted PFF match (no per-frame data)."""
    z = np.load(npz_path, allow_pickle=False)
    gt = z["gt"]
    live = gt != X
    bv = z["ball_visible"]
    n_imp, n_out, n_rest = (int(v) for v in z["label_info"])
    return {
        "source_fps": float(z["fps_source"]),
        "ball_visible_pct": round(100 * float(bv[live].mean()), 2) if live.any() else 0.0,
        "ball_visible_pct_in_D": round(100 * float(bv[gt == D].mean()), 2) if (gt == D).any() else 0.0,
        "periods": int(len(z["period_bounds"])),
        "restart_events": n_rest,
        "out_events": n_out,
        "restarts_without_out_event": n_imp,
    }


def pff_summary(per_match: dict) -> dict:
    """``reports/pff_summary.json`` content: per-match stats, ranges and flags (information
    only: the IDSSE ranges are NOT used to tune any label rule)."""
    keys = ["pct_D", "pct_H", "pct_A", "possessions", "mean_possession_s", "median_possession_s",
            "possessions_per_5min", "evaluated_minutes", "ball_visible_pct", "ball_missing_pct"]
    rng = {k: [min(m[k] for m in per_match.values()), max(m[k] for m in per_match.values())] for k in keys}
    flags = {}
    for name, m in per_match.items():
        f = [f"{k}={m[k]} outside IDSSE {lo}-{hi}" for k, (lo, hi) in IDSSE_RANGES.items() if not lo <= m[k] <= hi]
        if f:
            flags[name] = f
    return {
        "matches": len(per_match),
        "ranges": rng,
        "idsse_reference": {k: list(v) for k, v in IDSSE_RANGES.items()},
        "outliers_vs_idsse": flags,
        "per_match": per_match,
    }
