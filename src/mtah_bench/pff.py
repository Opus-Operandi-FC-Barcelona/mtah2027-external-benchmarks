"""PFF FC World Cup 2022 -> common npz (GATED dataset: the user must obtain the files).

Expected layout (after the user accepted PFF's terms and downloaded the data):

    data/pff/<game_id>/metadata.json     (Metadata/<game_id>.json)
    data/pff/<game_id>/rosters.json      (Rosters/<game_id>.json)
    data/pff/<game_id>/tracking.jsonl.bz2 (Tracking Data/<game_id>.jsonl.bz2; plain .jsonl also ok)

Labels use the ball-status / possession channel that kloppy's PFF reader derives from the
tracking file's own ``game_event`` records (ball out / kick-off / on-the-ball with
``home_ball``), i.e. exactly the IDSSE rule (docs/LABELLING_RULES.md). NOTE: written against
kloppy 3.18.0's PFF deserialiser and not run on real files (not downloadable without a form).
"""
from __future__ import annotations

import bz2
import io
from pathlib import Path

import numpy as np

from .convert import TARGET_FPS, _ground, _save, _tracking_to_arrays, remap_bounds, resample_indices
from .labels import derive_ball_status_labels


def pff_game_dirs(data_dir: Path) -> list[str]:
    base = Path(data_dir) / "pff"
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "metadata.json").exists())


def convert_pff(data_dir: Path, game_id: str, out_dir: Path) -> Path:
    from kloppy import pff
    from kloppy.domain import BallState

    d = Path(data_dir) / "pff" / game_id
    trk = d / "tracking.jsonl.bz2"
    raw = io.BytesIO(bz2.open(trk).read()) if trk.exists() else (d / "tracking.jsonl").open("rb")
    ds = pff.load_tracking(
        meta_data=(d / "metadata.json").open("rb"),
        roster_meta_data=(d / "rosters.json").open("rb"),
        raw_data=raw,
        coordinates="pff",
        only_alive=False,
    )
    L = float(ds.metadata.pitch_dimensions.pitch_length or 105.0)
    W = float(ds.metadata.pitch_dimensions.pitch_width or 68.0)
    fid, per, xy, tos, pids, ball = _tracking_to_arrays(ds, lambda x, y: (x + L / 2, y + W / 2))
    ground = {t.team_id: _ground(t) for t in ds.metadata.teams}
    alive = np.array([f.ball_state == BallState.ALIVE for f in ds.records])
    owner = np.array(
        [ground[f.ball_owning_team.team_id] if f.ball_owning_team is not None else -1 for f in ds.records],
        dtype=np.int8,
    )
    gt, bounds = derive_ball_status_labels(alive, owner, per)
    idx = resample_indices(len(fid), float(ds.metadata.frame_rate), TARGET_FPS)
    if len(idx) != len(fid):
        bounds = remap_bounds(bounds, idx)
    return _save(Path(out_dir) / f"pff_{game_id}.npz", xy, tos, ball, L, W, bounds, gt, fid, pids, idx)
