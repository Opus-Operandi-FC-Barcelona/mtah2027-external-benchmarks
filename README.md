# mtah2027-external-benchmarks

Validation and hyper-parameter tuning for `mtah2027-possession-engine` on open tracking data from
**other matches only**. The competition rules forbid tuning on the provided BAR-ATH match — the
public 5 minutes are a pipeline check, not a tuning set — so every number in the engine's
`configs/default.yaml` that is not a published constant is chosen here.

## Datasets

Inventory of the local `cache/` (verified from the file names): **94 matches** = IDSSE 7 + Metrica 3 +
SkillCorner 20 + PFF 64.

| Dataset | Matches | Content | Possession labels |
|---|---|---|---|
| Metrica Sports sample-data, Games 1-3 | 3 | Anonymised tracking (25 fps) + event data | Derived algorithmically from events (team of the last controlled event; dead between stoppage and restart) |
| IDSSE Bundesliga open data (Bassek et al., CC BY 4.0) | 7 | Positional data + official events | Ball-status and possession channels shipped with the data |
| SkillCorner Open Data (A-League 24/25, broadcast tracking 10 fps; MIT) | 20 | Broadcast tracking, ball mostly visible | `phases_of_play` (team in possession; gaps = dead; off-screen = `X`) |
| PFF FC World Cup 2022 (tracking 29.97 fps + event data; terms accepted by the user, local only) | 64 | Camera-derived tracking with real occlusions (ball missing in ~1/3 of frames) | `OTB` owner / `OUT` dead / set-piece restarts from the event data |

Licences, access and the steps for gated data: [docs/DATASETS.md](docs/DATASETS.md); open-data search log:
[docs/DATASET_SEARCH_2026-10-26.md](docs/DATASET_SEARCH_2026-10-26.md) (no further Metrica-style tracking is public).
Loaded with `kloppy` (Metrica, IDSSE) or plain JSON (SkillCorner, PFF). No labels are produced by hand, and nothing
from the competition match is in this repository.

## What the engine takes from here

* **Training data of the engine's Specialists A/B**: the 7 IDSSE matches (degraded) + 20 PFF matches at weight 0.5
  (`models/specialists*.json` in the engine).  SkillCorner was tried and left off (hurt the degraded validation).
* **Calibrated degrader profile (engine Issue #38).**  The engine's `external_degrader_target` section turns clean
  matches into inputs that look like the competition tracking in label-free gap statistics: log-normal ball-gap
  durations (median 12.5 s, sigma 0.95, 5-240 s) separated by >= 9 s of visible ball, 42.4 % of the ball hidden,
  single-player dropouts (2.4 per min, 5-30 s) and group dropouts (0.5 per min, 3-6 players for 5-20 s).  The previous
  profile is kept as `external_degrader_legacy`.  The degrader code lives in the engine; this repository supplies the
  clean matches and the sweeps.
* **Gap placement over dead ball (engine Issue #48).**  The calibrated profile can place the hidden-ball blocks
  preferentially over dead-ball stretches (`dead_affinity`; 1 = blind to `D`).  A label-free estimator validated on
  these matches (IDSSE with planted affinities, PFF, Metrica) puts the competition tracking well above random placement,
  so the engine's default model is trained at `dead_affinity` 33 and evaluated on B'(33), with B'(1) and PFF as guards.
* **Validation protocols**: B' = 7 IDSSE leave-one-match-out under the calibrated degrader, seeds 0-2 (B'(33) / B'(1) by gap placement); A = PFF
  held-out (4 folds) plus 8 PFF matches never used in training.

## Sweeps done and upcoming

| Sweep | Where | Status |
|---|---|---|
| Decoder parameters, heuristic route (Optuna TPE seed 0, 300 trials, 7 IDSSE degraded, nested LOMO) | `scripts/run_optuna_sweep.py`, `reports/optuna_heuristic*.{json,yaml}` | done, adopted in the engine (`route_overrides.heuristic`) |
| Decoder parameters, specialists route | `reports/optuna_specialists*.{json,yaml}` | done twice (first sweep; refreshed after the window features, issue #3 / engine #31), adopted (`route_overrides.specialists`) |
| Bounded decoder sweep under the calibrated degrader (nested LOMO seeds 0-1, blind seed 2, PFF guard) | `reports/issue44/` (engine `scripts/decoder_sweep.py`) | done: best +0.0091 on B'(33) < +0.01, guards pass, **not adopted** (code + `--model-variant ensemble` kept OFF) |
## Quick start (exact commands)

```bash
python -m pip install -r requirements.txt           # kloppy 3.18.0, numpy, pandas, pyyaml, pytest
python scripts/download_metrica_sample.py            # -> data/metrica/   (~175 MB)
python scripts/download_idsse.py --matches all       # -> data/idsse/     (~2.5 GB, 7 matches; or list ids)
python scripts/download_skillcorner.py --matches all  # -> data/skillcorner/ (~1.8 GB, MIT, no form)
python scripts/build_benchmark.py                    # -> cache/<match>.npz + reports/benchmark_summary.json
python scripts/build_benchmark.py --download         # same, downloading first
python scripts/build_benchmark.py --only metrica_1 idsse_J03WPY   # subset
python scripts/build_benchmark.py --only pff_3812      # PFF game (data/pff/<id>/, see docs/DATASETS.md); -> reports/pff_summary.json
python -m pytest -q                                  # labelling-rule tests (no network)
```

Match names: `skillcorner_<id>` (20 ids, see `mtah_bench.skillcorner.SC_MATCHES`), `pff_<game_id>`, `metrica_1..3`, `idsse_J03WPY J03WMX J03WN1 J03WOH J03WQQ J03WOY J03WR9`.
`data/` and `cache/` are gitignored. `floodlight` is deliberately not used (kloppy covers both
providers), so it is not pinned. Note: `springernature.figshare.com` (the URL kloppy embeds)
is behind an AWS WAF challenge and returns empty bodies, so `download_idsse.py` uses
`ndownloader.figshare.com` with the same file ids.

## npz format (`cache/<match>.npz`, 25 fps, metres, origin at a pitch corner)

`players_xy[N,S,2]`, `team_of_slot[S]` (0 home/1 away), `ball_xy[N,2]`, `fps`, `pitch=(L,W)`,
`period_bounds[P,2]` (inclusive frame indices), `gt[N]` in `H/A/D/X`, plus `frame_ids`, `player_ids`.
SkillCorner files add `observed[N]` (players on screen) and `players_detected[N,S]`; PFF files add
`ball_visible[N]`, `players_visible[N,S]`, `period[N]`, `fps_source`, `label_info`.
Labelling rules: [docs/LABELLING_RULES.md](docs/LABELLING_RULES.md).

## Pipeline

1. `scripts/download_*.py` — fetch the open datasets into `data/` (gitignored).
2. Convert each match to the engine's ATD-like arrays and ground-truth `H/A/D` labels.
3. Degrade them with the engine's `external_degrader` (ball hidden on ~41% of frames in
   realistic blocks, track identity breaks) so they look like the automatic BAR-ATH tracking.
4. `scripts/run_optuna_sweep.py` — maximise mean `spellF1` (official `possession_metrics.py`)
   across the degraded matches, leave-one-match-out.
5. Export the best parameters to the engine's `configs/default.yaml`.
