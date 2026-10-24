# mtah2027-external-benchmarks

Validation and hyper-parameter tuning for `mtah2027-possession-engine` on open tracking data from
**other matches only**. The competition rules forbid tuning on the provided BAR-ATH match — the
public 5 minutes are a pipeline check, not a tuning set — so every number in the engine's
`configs/default.yaml` that is not a published constant is chosen here.

## Datasets

| Dataset | Content | Possession labels |
|---|---|---|
| Metrica Sports sample-data, Games 1–3 | Anonymised tracking (25 fps) + event data | Derived algorithmically from events (team of the last controlled event; dead between stoppage and restart) |
| IDSSE Bundesliga open data (Bassek et al.) | Positional data + official events, 7 matches | Ball-status and possession channels shipped with the data |

Both are loaded with `kloppy` / `floodlight`. No labels are produced by hand.

## Quick start (exact commands)

```bash
python -m pip install -r requirements.txt           # kloppy 3.18.0, numpy, pandas, pyyaml, pytest
python scripts/download_metrica_sample.py            # -> data/metrica/   (~175 MB)
python scripts/download_idsse.py --matches all       # -> data/idsse/     (~2.5 GB, 7 matches; or list ids)
python scripts/build_benchmark.py                    # -> cache/<match>.npz + reports/benchmark_summary.json
python scripts/build_benchmark.py --download         # same, downloading first
python scripts/build_benchmark.py --only metrica_1 idsse_J03WPY   # subset
python -m pytest -q                                  # labelling-rule tests (no network)
```

Match names: `metrica_1..3`, `idsse_J03WPY J03WMX J03WN1 J03WOH J03WQQ J03WOY J03WR9`.
`data/` and `cache/` are gitignored. `floodlight` is deliberately not used (kloppy covers both
providers), so it is not pinned. Note: `springernature.figshare.com` (the URL kloppy embeds)
is behind an AWS WAF challenge and returns empty bodies, so `download_idsse.py` uses
`ndownloader.figshare.com` with the same file ids.

## npz format (`cache/<match>.npz`, 25 fps, metres, origin at a pitch corner)

`players_xy[N,S,2]`, `team_of_slot[S]` (0 home/1 away), `ball_xy[N,2]`, `fps`, `pitch=(L,W)`,
`period_bounds[P,2]` (inclusive frame indices), `gt[N]` in `H/A/D/X`, plus `frame_ids`, `player_ids`.
Labelling rules: [docs/LABELLING_RULES.md](docs/LABELLING_RULES.md).

## Pipeline

1. `scripts/download_*.py` — fetch the open datasets into `data/` (gitignored).
2. Convert each match to the engine's ATD-like arrays and ground-truth `H/A/D` labels.
3. Degrade them with the engine's `external_degrader` (ball hidden on ~41% of frames in
   realistic blocks, track identity breaks) so they look like the automatic BAR-ATH tracking.
4. `scripts/run_optuna_sweep.py` — maximise mean `spellF1` (official `possession_metrics.py`)
   across the degraded matches, leave-one-match-out.
5. Export the best parameters to the engine's `configs/default.yaml`.
