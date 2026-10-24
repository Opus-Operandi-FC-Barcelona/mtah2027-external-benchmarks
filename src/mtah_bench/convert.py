"""Convert Metrica / IDSSE matches to a common ``cache/<match>.npz``.

Arrays in the npz (positions in metres, origin at the pitch corner, x along the
length, y along the width; NaN when absent):

* ``players_xy``   float32 [N, S, 2]
* ``team_of_slot`` int8 [S]            0 home / 1 away
* ``ball_xy``      float32 [N, 2]
* ``fps``          25
* ``pitch``        (length, width)
* ``period_bounds`` int [P, 2]         inclusive first/last frame index of each period
* ``gt``           '<U1' [N]           'H' / 'A' / 'D' / 'X'
* ``frame_ids``    int64 [N]           source frame numbers
* ``player_ids``   '<U..' [S]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import derive_ball_status_labels, derive_metrica_labels, label_stats

TARGET_FPS = 25
METRICA_GAMES = {
    "metrica_1": "Sample_Game_1",
    "metrica_2": "Sample_Game_2",
    "metrica_3": "Sample_Game_3",
}
IDSSE_MATCHES = ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]


# ------------------------------------------------------------------- utilities
def resample_indices(n_src: int, fps_src: float, fps_dst: float = TARGET_FPS) -> np.ndarray:
    """Nearest-neighbour source index for every destination frame (identity at equal fps)."""
    if abs(fps_src - fps_dst) < 1e-9:
        return np.arange(n_src)
    n_dst = int(np.floor((n_src - 1) / fps_src * fps_dst)) + 1
    return np.minimum(np.rint(np.arange(n_dst) * fps_src / fps_dst).astype(int), n_src - 1)


def remap_bounds(bounds, idx):
    """Map inclusive source-index bounds to destination indices after resampling."""
    return [
        (int(np.searchsorted(idx, lo)), int(np.searchsorted(idx, hi, side="right") - 1))
        for lo, hi in bounds
    ]


def _ground(team) -> int:
    return 0 if str(team.ground.value) == "home" else 1


def _tracking_to_arrays(ds, to_metres):
    """kloppy TrackingDataset -> (frame_ids, period_ids, players_xy, team_of_slot, player_ids, ball_xy)."""
    players = []
    for team in ds.metadata.teams:
        for p in sorted(team.players, key=lambda q: str(q.player_id)):
            players.append((_ground(team), p))
    slot_by_pid = {(t, str(p.player_id)): k for k, (t, p) in enumerate(players)}
    n, s = len(ds.records), len(players)
    xy = np.full((n, s, 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    fid = np.empty(n, dtype=np.int64)
    per = np.empty(n, dtype=np.int16)
    for i, f in enumerate(ds.records):
        fid[i] = f.frame_id
        per[i] = f.period.id
        if f.ball_coordinates is not None and f.ball_coordinates.x is not None:
            ball[i] = to_metres(f.ball_coordinates.x, f.ball_coordinates.y)
        for p, d in f.players_data.items():
            c = d.coordinates
            if c is None or c.x is None:
                continue
            xy[i, slot_by_pid[(_ground(p.team), str(p.player_id))]] = to_metres(c.x, c.y)
    used = ~np.isnan(xy[:, :, 0]).all(axis=0)  # drop squad members never on the pitch
    xy = xy[:, used]
    players = [pl for pl, u in zip(players, used) if u]
    team_of_slot = np.array([t for t, _ in players], dtype=np.int8)
    pids = np.array([f"{'H' if t == 0 else 'A'}_{p.player_id}" for t, p in players])
    return fid, per, xy, team_of_slot, pids, ball


def _save(path, xy, tos, ball, L, W, bounds, gt, fid, pids, idx) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        players_xy=xy[idx],
        team_of_slot=tos,
        ball_xy=ball[idx],
        fps=TARGET_FPS,
        pitch=np.array([L, W]),
        period_bounds=np.array(bounds, dtype=np.int64).reshape(-1, 2),
        gt=gt[idx],
        frame_ids=fid[idx],
        player_ids=pids,
    )
    return path


# --------------------------------------------------------------------- Metrica
def load_metrica_events(game_dir: Path, game: str, team_ids: dict) -> pd.DataFrame:
    """Normalise Metrica events (CSV for games 1-2, JSON for game 3) to columns
    team(0/1), type, subtype, period, start_frame, end_frame."""
    game_dir = Path(game_dir)
    if game in ("Sample_Game_1", "Sample_Game_2"):
        e = pd.read_csv(game_dir / f"{game}_RawEventsData.csv")
        out = pd.DataFrame(
            {
                "team": e["Team"].str.upper().map({"HOME": 0, "AWAY": 1}),
                "type": e["Type"],
                "subtype": e["Subtype"],
                "period": e["Period"],
                "start_frame": e["Start Frame"],
                "end_frame": e["End Frame"],
            }
        )
    else:
        raw = json.loads((game_dir / f"{game}_events.json").read_text(encoding="utf-8"))["data"]
        rows = []
        for ev in raw:
            st = ev.get("subtypes")
            if isinstance(st, dict):
                st = [st]
            rows.append(
                {
                    "team": team_ids[ev["team"]["id"]],
                    "type": ev["type"]["name"],
                    "subtype": "-".join(s["name"] for s in st) if st else None,
                    "period": ev["period"],
                    "start_frame": ev["start"]["frame"],
                    "end_frame": ev["end"]["frame"],
                }
            )
        out = pd.DataFrame(rows)
    out = out.dropna(subset=["team", "start_frame", "end_frame", "period"]).copy()
    for c in ("team", "period", "start_frame", "end_frame"):
        out[c] = out[c].astype(int)
    return out.reset_index(drop=True)


def convert_metrica(data_dir: Path, name: str, out_dir: Path) -> Path:
    from kloppy import metrica

    game = METRICA_GAMES[name]
    d = Path(data_dir) / "metrica" / game
    if game == "Sample_Game_3":
        ds = metrica.load_tracking_epts(
            d / f"{game}_metadata.xml", d / f"{game}_tracking.txt", coordinates="metrica"
        )
    else:
        ds = metrica.load_tracking_csv(
            d / f"{game}_RawTrackingData_Home_Team.csv",
            d / f"{game}_RawTrackingData_Away_Team.csv",
            coordinates="metrica",
        )
    pd_ = ds.metadata.pitch_dimensions
    L, W = float(pd_.pitch_length or 105.0), float(pd_.pitch_width or 68.0)
    fid, per, xy, tos, pids, ball = _tracking_to_arrays(ds, lambda x, y: (x * L, y * W))
    team_ids = {str(t.team_id): _ground(t) for t in ds.metadata.teams}
    events = load_metrica_events(d, game, team_ids)
    gt, bounds = derive_metrica_labels(events, fid)
    idx = resample_indices(len(fid), float(ds.metadata.frame_rate))
    if len(idx) != len(fid):
        bounds = remap_bounds(bounds, idx)
    return _save(Path(out_dir) / f"{name}.npz", xy, tos, ball, L, W, bounds, gt, fid, pids, idx)


# ----------------------------------------------------------------------- IDSSE
def convert_idsse(data_dir: Path, match_id: str, out_dir: Path) -> Path:
    from kloppy import sportec
    from kloppy.domain import BallState

    d = Path(data_dir) / "idsse" / match_id
    ds = sportec.load_tracking(
        d / "meta.xml", d / "tracking.xml", only_alive=False, coordinates="sportec"
    )
    L = float(ds.metadata.pitch_dimensions.pitch_length)
    W = float(ds.metadata.pitch_dimensions.pitch_width)
    fid, per, xy, tos, pids, ball = _tracking_to_arrays(ds, lambda x, y: (x + L / 2, y + W / 2))
    ground = {t.team_id: _ground(t) for t in ds.metadata.teams}
    alive = np.zeros(len(fid), dtype=bool)
    owner = np.full(len(fid), -1, dtype=np.int8)
    for i, f in enumerate(ds.records):
        alive[i] = f.ball_state == BallState.ALIVE
        if f.ball_owning_team is not None:
            owner[i] = ground[f.ball_owning_team.team_id]
    gt, bounds = derive_ball_status_labels(alive, owner, per)
    idx = resample_indices(len(fid), float(ds.metadata.frame_rate))
    if len(idx) != len(fid):
        bounds = remap_bounds(bounds, idx)
    return _save(Path(out_dir) / f"idsse_{match_id}.npz", xy, tos, ball, L, W, bounds, gt, fid, pids, idx)


def stats_of_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    st = label_stats(z["gt"], float(z["fps"]))
    st["slots"] = int(z["team_of_slot"].shape[0])
    st["ball_missing_pct"] = round(100 * float(np.isnan(z["ball_xy"][:, 0]).mean()), 2)
    st["pitch"] = [float(v) for v in z["pitch"]]
    return st
