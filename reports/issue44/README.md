# Bounded decoder / fusion / ensemble sweep under the calibrated degrader (engine issue #44, benchmarks issue #5)

External data only (7 IDSSE Bundesliga matches, 20 + 8 PFF World Cup 2022 matches); aggregates only; nothing from BAR-ATH.
Tool: `scripts/decoder_sweep.py` and `scripts/train_cv_models.py` of the engine repository (`mtah2027-possession-engine`).

## Protocol

* **B'(33)** = 7 IDSSE leave-one-match-out, held-out match degraded by the calibrated degrader at `dead_affinity` 33
  (the world the default model is trained for).  Selection on seeds 0-1 (mean spellF1), seed 2 = blind confirmation.
* **B'(1)** = same with `dead_affinity` 1 (blind-gap world): guard, reject if the mean drops > 0.01.
* **A** = the first 20 PFF matches, 4-fold (model trained without the fold) + 8 never-used PFF matches with the committed
  models: guard, reject if the drop > 0.03.
* Baseline = the current default (`models/specialists.json` = `gap`) with the current `route_overrides.specialists`.
* Swept (`route_overrides.specialists`): `specialist_fusion.{scale_visible, scale_hidden, blend_visible, blend_hidden}`,
  `transition_matrix.{ha_closed, ha_open, ha_occluded, dead_open, dead_closed, d_exit_closed, d_exit_open, dead_ramp, alive_ramp,
  touch_halfwidth, touch_min_score, occlusion_min_s}`, `spell_optimizer.{micro_poss_s, min_dead_s, min_isolated_poss_s,
  bridge_min_poss_s, bridge_dead_threshold, keep_min_visible_frac}`.  Candidate files: `stage*.json`.
* Ensembles: per-frame mix of the class probabilities of committed models (`gap` = IDSSE aff 33 + PFF 0.5, `mixed` = legacy
  degrader + PFF 0.5, `calib` = calibrated degrader aff 1 + PFF 0.5), linear and log-linear, weight of `gap` 0.25 / 0.5 / 0.75, 3-way mixes.
* Adoption rule: B'(33) mean >= +0.01, no seed worse than -0.003, A drop <= 0.03, B'(1) drop <= 0.01.

## Selection (B'(33), seeds 0-1, mean spellF1 of 14 held-out runs; base 0.6692)

| stage | candidates | best | gain |
|---|---|---|---|
| 1 one-at-a-time (22 parameters) | 60 | `ha_closed -11` | +0.0019 |
| 2 random joint search | 234 | `scale_visible 0.65, ha_closed -15, micro_poss_s 0.6` (+ small ones) | +0.0074 |
| 3 grid around that core | 81 | `scale_visible 0.75, ha_closed -18, micro_poss_s 0.6, blend_hidden 0, dead_open -2, min_isolated_poss_s 0.8` (D1) | +0.0083 |
| ensembles (gap + mixed / calib, linear / log-linear) | 47 | with D1: `lin gap0.75+mixed` | +0.0082 (no gain over D1 alone; without D1: +0.0013) |
| 4 joint random (gap alone, lin 0.75/0.25 gap+mixed, lin 0.5/0.5, log 0.75/0.25 gap+calib; 80 each) | 320 | `lin gap0.75+mixed` with block `lin75_r06` | +0.0090 |

The landscape is flat: most single parameters change B'(33) by < 0.002; only `scale_visible` (lower), `ha_closed`
(much more expensive H<->A switches), `micro_poss_s` (shorter) and `blend_hidden` (0) matter.  `blend_visible` > 0.2 and
`micro_poss_s` > 1.4 hurt.  Ensembles of `gap` with the legacy-degrader models do not add on B'(33) without the decoder block
(+0.0013); alone `mixed` / `calib` lose 0.018-0.020 there (trained for another world).

## Confirmation of the finalists (all seeds; delta against the baseline)

| candidate | B'(33) mean (0 / 1 / 2 blind) | B'(1) | A 20 PFF 4-fold | A 8 unseen PFF |
|---|---|---|---|---|
| base (default `gap`, current decoder) | 0.6693 (0.6709 / 0.6675 / 0.6695) | 0.6017 | 0.5263 | 0.5263 |
| D1 (decoder block, `gap`) | 0.6759 +0.0066 (+0.0071 / +0.0095 / +0.0033) | 0.6046 +0.0029 | 0.5322 +0.0059 | 0.5326 +0.0063 |
| gap_r11 (stage 4 best, `gap`) | 0.6757 +0.0064 (+0.0084 / +0.0080 / +0.0027) | 0.6041 +0.0024 | 0.5333 +0.0070 | 0.5346 +0.0083 |
| **lin75_r06** (`lin gap0.75 + mixed0.25`, block r06) | **0.6784 +0.0091 (+0.0113 / +0.0068 / +0.0091)** | 0.6085 +0.0068 | 0.5360 +0.0097 | 0.5350 +0.0087 |
| lin gap0.75+mixed + D1 | 0.6764 +0.0071 (+0.0050 / +0.0115 / +0.0048) | 0.6088 +0.0071 | 0.5333 +0.0071 | 0.5338 +0.0075 |
| lin gap0.5+mixed + D1 | 0.6746 +0.0052 | 0.6095 +0.0078 | 0.5325 +0.0062 | 0.5348 +0.0086 |
| lin gap0.75+calib + D1 | 0.6748 +0.0054 | 0.6077 +0.0061 | 0.5351 +0.0088 | 0.5309 +0.0046 |
| mixed alone | 0.6512 -0.0181 | 0.6047 +0.0030 | 0.5231 -0.0032 | 0.5257 -0.0006 |
| calib alone | 0.6510 -0.0184 | 0.6073 +0.0056 | 0.5300 +0.0037 | 0.5303 +0.0040 |

Strata (pooled B'(33), 3 seeds, `strata_b33.txt`): predicted D matches the true D in every stratum (< 16 players 50.1 vs 48.1 %,
< 20: 46.1 vs 45.1 %, >= 20: 44.7 vs 44.4 %), so there is no D over-prediction to correct; macroF1 +0.005 / +0.003 / +0.001.

## Decision

Best candidate `lin75_r06`: B'(33) +0.0091 (selection seeds +0.0090, blind +0.0091, no seed negative) -- **below the +0.01
bar**, so it is not adopted; all guards pass (A +0.0097 / +0.0087, B'(1) +0.0068).  Selection and gain are of the same size
(max over ~700 candidates), so the true gain is probably lower.  The code, the `ensemble` model variant and the block
(`decoder_candidates.issue_44` in the engine `configs/default.yaml`) stay available, OFF.
