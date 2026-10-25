# Datasets: licences, access and status

All data is third-party, stored under `data/` (gitignored, never committed, never sent to a
cloud LLM). Raw files are treated as untrusted input (parsed with `json`/`csv`/kloppy only).

| Dataset | Matches | Rate | Ball | Label source | Licence / terms | Access | Status |
|---|---|---|---|---|---|---|---|
| Metrica Sports sample-data | 3 | 25 | tracked, missing while dead | events (derived) | no licence file in the sample-data repo (terms not stated; verify before any redistribution) | open, GitHub | in cache |
| IDSSE Bundesliga (Bassek et al., Sci Data 12, 195, 2025) | 7 | 25 | tracked | official ball-status + possession channels | CC BY 4.0 | open, figshare | in cache |
| SkillCorner Open Data (A-League 2024/25) | 20 in the repo (README says 10) | 10 (broadcast) | ~75 % visible | `phases_of_play.csv` (+ per-frame possession) | MIT, asks to credit SkillCorner | open, GitHub (+ Git LFS) via `scripts/download_skillcorner.py` | in cache |
| PFF FC World Cup 2022 | 64 | 29.97 (resampled to 25) | tracked ~33 % VISIBLE + ~33 % ESTIMATED, rest missing | game events in `events.json` (OTB / OUT / set pieces / fouls) | PFF terms, accepted by the user on a web form; not redistributable, `data/` stays gitignored | user downloaded it (Google Drive, 3 zips) | in cache |
| Impect open data (Bundesliga 23/24) | - | - | - | events only | proprietary LICENSE.pdf | open | rejected: no tracking |
| StatsBomb open data / 360, Wyscout events | - | - | - | events only | various | open | rejected: no tracking |
| SoccerTrack v2 (amateur, BePro) | 10 | - | game state reconstruction | none for possession | - | open | not pursued: amateur panoramic video, no possession/ball-status channel |
| kloppy test samples (Tracab, SecondSpectrum, Signality, HawkEye) | seconds each | - | - | - | MIT (repo) | repo tests | rejected: seconds of data, not matches |

Metrica: the repo ships no LICENSE file; we
only use it locally for tuning. SkillCorner: MIT; credit "SkillCorner Open Data".

## PFF FC World Cup 2022: layout and conversion

The user accepted PFF's terms and downloaded the dataset (3 Google-Drive zip parts, ~2.8 GB,
`FIFA World Cup 2022-<stamp>-1-00{1,2,3}.zip`, placed in `data/pff/`). Zip content (one root
folder `FIFA World Cup 2022/`): `Metadata/<id>.json`, `Rosters/<id>.json`,
`Tracking Data/<id>.jsonl.bz2` (64 each), `Event Data/<id>.json` (latest release, 64) plus two older
snapshots `Event Data/March 14, 2025/` and `Event Data/May 1, 2025/` (not used), `players.csv`,
`competitions.csv`, a change log and the event / tracking specification PDFs. Zip names are
untrusted: members are extracted by regex on the exact expected paths only, nothing is executed.

Reorganised layout (all gitignored):

```
data/pff/<game_id>/metadata.json         # one-element list; fps 29.97, pitch, kick-off video times
data/pff/<game_id>/rosters.json
data/pff/<game_id>/tracking.jsonl.bz2    # kept compressed, streamed by the converter
data/pff/<game_id>/events.json           # game + possession events, fouls, same video clock
data/pff/_extracted/                     # specs (PDF), players.csv, competitions.csv, change log
data/pff/*.zip                           # originals; can be deleted once converted
```

Convert: `python scripts/build_benchmark.py --only pff_<game_id> [...]` (a default run converts every
folder under `data/pff/`). Output `cache/pff_<game_id>.npz` (npz format of the README plus
`ball_visible`, `players_visible`, `period`, `fps_source`, `label_info`) and the aggregate-only
`reports/pff_summary.json`. Label rules: [LABELLING_RULES.md](LABELLING_RULES.md).

Notes for users of these files: (1) the ball is missing in ~1/3 of the frames, so the engine's
default `only_complete_ball` filter excludes them from specialist training; use them as
real-world occlusion validation (the engine's degrader is not needed on top, the ball is already
partly missing). (2) `cache/pff_*.npz` sorts between `metrica_*` and `skillcorner_*`: the
SkillCorner per-match seed offsets of `run_optuna_sweep.py` (index in `sorted(cache/*.npz)`)
shift by the number of PFF files; IDSSE offsets are unchanged. (3) 5 of the 64 games have extra
time (periods 3 and 4) and become 4 periods in `period_bounds`.

## Using SkillCorner in the sweep

It is partial-view broadcast data (`X` where the camera is off the pitch, ball missing ~20-40 %,
10 fps). It is useful as extra training/validation of the ball-absent specialist and the
decoder under realistic occlusion, not as a replacement of full-view matches; do not apply the
engine's degrader blindly on top (it is already degraded). The sweep script derives per-match
seeds from the position in `sorted(cache/*.npz)`; the new `skillcorner_*` files sort after
`idsse_*`/`metrica_*`, so existing seeds are unchanged.
