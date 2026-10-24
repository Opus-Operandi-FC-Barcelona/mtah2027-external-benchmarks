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

## Pipeline

1. `scripts/download_*.py` — fetch the open datasets into `data/` (gitignored).
2. Convert each match to the engine's ATD-like arrays and ground-truth `H/A/D` labels.
3. Degrade them with the engine's `external_degrader` (ball hidden on ~41% of frames in
   realistic blocks, track identity breaks) so they look like the automatic BAR-ATH tracking.
4. `scripts/run_optuna_sweep.py` — maximise mean `spellF1` (official `possession_metrics.py`)
   across the degraded matches, leave-one-match-out.
5. Export the best parameters to the engine's `configs/default.yaml`.
