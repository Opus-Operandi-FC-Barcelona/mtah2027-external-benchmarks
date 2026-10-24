# Ground-truth labelling rules

Labels per frame at 25 fps: `H` home possession, `A` away possession, `D` dead ball,
`X` outside the evaluated period. No label is produced by hand. A `D` (or `X`) cuts a
possession: `H D H` is two possessions (same convention as the competition metric).
Implementation: `src/mtah_bench/labels.py` (pure functions, unit-tested in `tests/`).

Coordinates in the npz files are metres with the origin at a pitch corner. Metrica
frames are already 25 fps and so are IDSSE frames; `convert.resample_indices` does
nearest-neighbour resampling for any other rate.

## IDSSE Bundesliga (Bassek et al. 2025)

The tracking XML ships an official `BallStatus` (alive/dead) and `BallPossession`
(home/away) channel per frame (via kloppy: `ball_state`, `ball_owning_team`).

* ball dead -> `D`
* ball alive -> team in `BallPossession` (if momentarily missing: last known owner;
  `D` if none yet)
* per half, frames before the first / after the last alive frame -> `X`

This mirrors how the competition derives its GT mechanically from tracking channels
(`D` where the tracking says dead, otherwise the team the tracking names).
Note that the IDSSE possession channel keeps naming a team during dead ball; we ignore
it there because status takes precedence.

## Metrica Sports sample games (events -> labels)

Events are normalised to `team, type, subtype, period, start_frame, end_frame`
(games 1-2 from the CSV, game 3 from the JSON; subtypes joined with `-`). Per period,
events are replayed in frame order, producing state changes:

| Rule | Events | Effect |
|---|---|---|
| Control | `PASS`, `SHOT`, `CARRY`, `RECOVERY`, `SET PIECE` | at `start_frame`, state := team of the event |
| Not control | `CHALLENGE` (no stop), `BALL LOST` (non stop), `RECOVERY-BLOCKED` (deflection), `REFEREE HIT`, `CARD` aside | no change |
| Stop at start | `FAULT RECEIVED`, `CARD`, `CHALLENGE-*FAULT*`, `BALL LOST-OFFSIDE`, `BALL LOST-HAND BALL`, `BALL LOST-END HALF` | at `start_frame`, state := `D` |
| Stop at end | `BALL OUT` (end = ball crosses the line), `SHOT-*GOAL*`, `SHOT-*OUT*`, `PASS-OFFSIDE` | at `end_frame`, state := `D` |
| Restart | the next control event, normally `SET PIECE` (kick-off, throw-in, corner, free kick) or `PASS/GOAL KICK` | state := restarting team |

The possession of a pass stays with the passer until an opposing control event
(interception/recovery) or a stop; so the ball in flight counts for the passing team.
A period is bounded by its first event start and last event end; everything else is `X`
(Metrica tracking includes warm-up/half-time frames).

Validation of the rules without human labels: in the Metrica data the tracker loses the
ball while it is dead. Over the three games, ball-missing is 92-99 % of `D` frames and only
2-5 % of `H/A` frames, and 92-97 % of ball-missing frames are labelled `D`.
Timing precision is bounded by event timestamps (typically a few frames), which is
adequate for tuning but means IDSSE is the better benchmark for exact boundaries.

## Sanity statistics (not tuning targets)

The competition's public 5 minutes show 17 possessions, mean 14.2 s, ~20 % `D`.
Our matches differ because real matches have ~40 % dead time over 95 minutes and many
short `D`-separated spells; see `reports/benchmark_summary.json` for the per-match numbers.
These numbers are only a sanity check and must never be used to select hyper-parameters.
