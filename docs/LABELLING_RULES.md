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

## SkillCorner A-League open data (phases of play -> labels)

Broadcast tracking at 10 fps (resampled to 25 fps, nearest neighbour). All 22 players are
present in a frame when the pitch is on screen (off-screen players are extrapolated,
`is_detected=False`, kept in `players_detected`); frames where the broadcast shows something else
(replays, close-ups) have an empty `player_data`. Implementation: `src/mtah_bench/skillcorner.py`.

* frame outside a period, or with no tracked players (`observed=False`) -> `X` (no information;
  it is NOT assumed dead);
* observed and inside a `phases_of_play` row -> team of `team_in_possession_id` (`H`/`A`);
* observed and between phases -> `D`. Phases are contiguous while play is live; every gap
  between consecutive phases is >= 1.8 s and coincides with an interruption (throw-in, free
  kick, goal kick, corner, goal). In these frames the ball is visible (100 % in the checked
  match) and the per-frame `possession.player_id` is set in only ~7 % of them.

Caveats, deliberately not tuned away: (1) the per-frame `possession.group` agrees with the phase
team in ~80 % of the frames (it flips on every brief duel; `chaotic` phases agree in 47 %); the
phase team is the smoother, spell-level label and is used. (2) Possession counts are lower than in
full-view matches because `X` runs (about 25-40 % of the period) cut spells and unobserved time
is not counted; compare `possessions_per_5min`, not totals. (3) Boundaries are accurate to the
10 fps source (0.1 s), the phase boundaries being annotated by SkillCorner's pipeline.
The `observed` array in the npz marks the frames with information.

## PFF FC World Cup 2022 (tracking + event data)

Tracking is 29.97 fps (resampled to 25 fps by nearest frame on the video clock, per period) and
centred metres (converted to corner-origin). Labels come from PFF's own event data
(`events.json`, same video clock as the tracking: `startTime` == `videoTimeMs / 1000`), replayed
per period in time order. Implementation: `src/mtah_bench/pff.py`
(`normalise_events`, `derive_pff_state_changes`, `derive_pff_labels`), unit-tested in
`tests/test_pff_labels.py`. No label is produced by hand and none is tuned.

| Event (`gameEvents.gameEventType`) | Effect |
|---|---|
| `FIRSTKICKOFF` / `SECONDKICKOFF` / `THIRDKICKOFF` / `FOURTHKICKOFF` | start of the period: state := team of the kick-off (the tracking file starts at this moment) |
| `OTB` (a player on the ball), any `setpieceType` | at `startTime`, state := team of the event (`homeTeam` -> `H`, else `A`). A pass stays with the passer until the opponent's next `OTB`, as in the Metrica rules. |
| `OUT` (ball out of play: touch, whistle, goal; `outType` T/W/H/A) | at `startTime`, state := `D` |
| `OTB` with `setpieceType` != `O` (throw-in, free kick, goal kick, corner, kick-off, drop ball, penalty) while still alive | no `OUT` was recorded: dead from the foul that was *given* (`foulType == 'I'` and on-field outcome != `F`), else from the end of the previous touch, until this restart (`restarts_without_out_event` in `pff_summary.json`; 41 of 6,376 restarts over the 64 games) |
| `OTB` with `setpieceType == O` after a foul | clears a pending foul stop (advantage `A` and missed infringements `M` never stop play) |
| `END` | period end; frames after it are `X` (so are frames before the kick-off; the files hardly contain any) |
| `SUB`, `ON`, `OFF`, `FOUL` (additional foul row), `G` (woodwork), rows with `initialNonEvent == true` | ignored |

Per frame: `D` while dead, the owner team while alive. Boundaries are as accurate as PFF's event
timestamps (a few frames); there is no tracking-derived ball-status channel as in IDSSE.

Caveats, deliberately not tuned away: (1) the ball is untracked in about a third of the frames and
`ESTIMATED` (inferred from the players) in another third; `ball_xy` carries every coordinate PFF
gives (VISIBLE and ESTIMATED), and the npz adds `ball_visible[N]` / `players_visible[N,S]` for the
VISIBLE ones. In `D` frames the ball is usually missing (4-14 % visible per match).
(2) Players are tracked per jersey number, so a slot is a (team, jersey) pair, not a track id; the
ids in `player_ids` are the roster player ids. (3) `OUT` events are recorded when the ball leaves
the field or play is whistled; long gaps before a restart (VAR, injuries, goal celebrations) are
therefore `D`, which is why `D` is 28-52 % of the evaluated time, in line with IDSSE. Possessions per match are 288-517; the mean spell is 6.4-14.3 s, above the IDSSE range (5.5-11 s) in many matches and below 314 possessions in about a third of them; these are reported as flags in `reports/pff_summary.json`, not tuned.

## Sanity statistics (not tuning targets)

The competition's public 5 minutes show 17 possessions, mean 14.2 s, ~20 % `D`.
Our matches differ because real matches have ~40 % dead time over 95 minutes and many
short `D`-separated spells; see `reports/benchmark_summary.json` for the per-match numbers.
These numbers are only a sanity check and must never be used to select hyper-parameters.
