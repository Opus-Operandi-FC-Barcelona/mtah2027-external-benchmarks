# Datasets: licences, access and status

All data is third-party, stored under `data/` (gitignored, never committed, never sent to a
cloud LLM). Raw files are treated as untrusted input (parsed with `json`/`csv`/kloppy only).

| Dataset | Matches | Rate | Ball | Label source | Licence / terms | Access | Status |
|---|---|---|---|---|---|---|---|
| Metrica Sports sample-data | 3 | 25 | tracked, missing while dead | events (derived) | no licence file in the sample-data repo (terms not stated; verify before any redistribution) | open, GitHub | in cache |
| IDSSE Bundesliga (Bassek et al., Sci Data 12, 195, 2025) | 7 | 25 | tracked | official ball-status + possession channels | CC BY 4.0 | open, figshare | in cache |
| SkillCorner Open Data (A-League 2024/25) | 20 in the repo (README says 10) | 10 (broadcast) | ~75 % visible | `phases_of_play.csv` (+ per-frame possession) | MIT, asks to credit SkillCorner | open, GitHub (+ Git LFS) via `scripts/download_skillcorner.py` | in cache |
| PFF FC World Cup 2022 | 64 | ~30 | tracked, visibility flag | game events in tracking (kloppy PFF reader) | PFF terms, accepted on a web form; see note | **needs user action** | converter ready, untested on real data |
| Impect open data (Bundesliga 23/24) | - | - | - | events only | proprietary LICENSE.pdf | open | rejected: no tracking |
| StatsBomb open data / 360, Wyscout events | - | - | - | events only | various | open | rejected: no tracking |
| SoccerTrack v2 (amateur, BePro) | 10 | - | game state reconstruction | none for possession | - | open | not pursued: amateur panoramic video, no possession/ball-status channel |
| kloppy test samples (Tracab, SecondSpectrum, Signality, HawkEye) | seconds each | - | - | - | MIT (repo) | repo tests | rejected: seconds of data, not matches |

Metrica: the repo ships no LICENSE file; we
only use it locally for tuning. SkillCorner: MIT; credit "SkillCorner Open Data".

## PFF FC World Cup 2022: what the user must do

1. Open https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset (also linked from
   https://fc.pff.com), read the data terms of use, and fill the access form yourself
   (the agent must not accept terms on your behalf). The response gives instant access.
2. Download Tracking Data (`<game_id>.jsonl.bz2`), Metadata (`<game_id>.json`) and Rosters
   (`<game_id>.json`) for the games you want (64 available; each tracking file is large, check
   disk first; the repo rule is to skip anything over 15 GB in total).
3. Place them as `data/pff/<game_id>/metadata.json`, `rosters.json`, `tracking.jsonl.bz2`.
4. `python scripts/build_benchmark.py --only pff_<game_id>` (any `data/pff/*` folder is also
   picked up automatically on a default run). Check the aggregate stats against the other
   matches; the converter was written from kloppy 3.18's PFF reader and has not seen real files,
   so expect to verify frame rate / orientation on the first game.

## Using SkillCorner in the sweep

It is partial-view broadcast data (`X` where the camera is off the pitch, ball missing ~20-40 %,
10 fps). It is useful as extra training/validation of the ball-absent specialist and the
decoder under realistic occlusion, not as a replacement of full-view matches; do not apply the
engine's degrader blindly on top (it is already degraded). The sweep script derives per-match
seeds from the position in `sorted(cache/*.npz)`; the new `skillcorner_*` files sort after
`idsse_*`/`metrica_*`, so existing seeds are unchanged.
