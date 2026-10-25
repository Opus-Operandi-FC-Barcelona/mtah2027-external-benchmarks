#!/usr/bin/env python
"""Optuna sweep of the decoding / post-processing hyper-parameters of ``mtah2027-possession-engine``.

External matches only (``cache/idsse_*.npz``, never BAR-ATH).  Every match is degraded with the engine's
realistic degrader (``external_degrader``); the objective is the mean *official* spellF1 (engine
``eval_bridge`` -> ``possession_metrics``) over the matches, with the mean macroF1 as a tiny tie-break.

Cost control.  Everything that does not depend on the searched parameters is computed once and cached in
``cache/optuna/`` (gitignored): the per-frame features of the front end, the heuristic emissions and, for
``--route specialists``, the raw emissions of the leave-one-match-out specialists (each held-out match is
scored by specialists trained on the *other* matches only).  A trial only rebuilds emissions (heuristic
route) or re-tempers / blends them (specialists route), the transition matrix, the Viterbi decode and the
spell optimiser, then scores.

Searched (ranges around the defaults of the engine ``configs/default.yaml``):
  transition_matrix   every log-penalty, touch window, ramps
  spell_optimizer     every threshold
  heuristic_emissions (route heuristic) evidence / dead / carry / touch weights
  specialist_fusion   (route specialists) tempering scales and heuristic blend weights
NOT searched: the KPI alarm ranges (alarms, not targets), the front-end feature parameters.

Overfitting guard (reported in ``reports/optuna_<route>.json``):
  * ``nested``: for each held-out match, the best trial of the pool is chosen on the *other* matches and
    scored on the held-out one (leave-one-match-out selection) -- the honest estimate;
  * ``check``: default vs best vs nested-selected parameters on degradation seeds that were not used in
    the search.

Only aggregate numbers are printed.  Usage::

    python scripts/run_optuna_sweep.py --route heuristic --trials 150 --timeout 5400 --export out.yaml
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = ROOT.parent / "mtah2027-possession-engine"
H_, A_, D_, X_ = 0, 1, 2, 3


def _engine_paths(engine: Path) -> None:
    for extra in (engine / "src", engine):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))


# ---------------------------------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------------------------------
# (name, section, kind, low, high_or_choices)   kind: float | int | cat
SPACE = {
    "common": [
        ("ha_closed", "transition_matrix", "float", -16.0, -7.0),
        ("ha_open", "transition_matrix", "float", -3.5, -0.3),
        ("touch_halfwidth", "transition_matrix", "int", 2, 10),
        ("touch_min_score", "transition_matrix", "float", 0.3, 0.9),
        ("touch_both_dist_m", "transition_matrix", "float", 1.2, 3.5),
        ("require_both_teams", "transition_matrix", "cat", [False, True], None),
        ("touch_gate_halfwidth", "transition_matrix", "int", 0, 5),
        ("ha_occluded", "transition_matrix", "float", -10.0, -3.0),
        ("occlusion_min_s", "transition_matrix", "float", 1.0, 5.0),
        ("dead_closed", "transition_matrix", "float", -12.0, -4.0),
        ("dead_open", "transition_matrix", "float", -4.0, -0.3),
        ("dead_ramp_lo", "transition_matrix", "float", 0.1, 0.5),
        ("dead_ramp_w", "transition_matrix", "float", 0.1, 0.5),
        ("d_exit_closed", "transition_matrix", "float", -13.0, -5.0),
        ("d_exit_open", "transition_matrix", "float", -4.0, -0.5),
        ("alive_ramp_lo", "transition_matrix", "float", 0.2, 0.7),
        ("alive_ramp_w", "transition_matrix", "float", 0.1, 0.5),
        ("micro_poss_s", "spell_optimizer", "float", 0.4, 3.0),
        ("keep_min_visible_frac", "spell_optimizer", "float", 0.2, 0.9),
        ("keep_min_touches", "spell_optimizer", "int", 1, 3),
        ("sp_touch_min_score", "spell_optimizer", "float", 0.3, 0.9),
        ("min_dead_s", "spell_optimizer", "float", 0.2, 2.0),
        ("min_isolated_poss_s", "spell_optimizer", "float", 0.3, 2.0),
        ("bridge_min_poss_s", "spell_optimizer", "float", 10.0, 40.0),
        ("bridge_min_dead_s", "spell_optimizer", "float", 2.0, 8.0),
        ("bridge_dead_threshold", "spell_optimizer", "float", 0.4, 0.9),
        ("bridge_requires_geometry", "spell_optimizer", "cat", [True, False], None),
    ],
    "heuristic": [
        ("w_evidence", "heuristic_emissions", "float", 0.05, 2.0),
        ("dead_scale", "heuristic_emissions", "float", 0.5, 2.0),
        ("dead_scale_no_ball", "heuristic_emissions", "float", 0.3, 2.0),
        ("carry_conf_threshold", "heuristic_emissions", "float", 0.15, 0.6),
        ("carry_weight", "heuristic_emissions", "float", 0.0, 4.0),
        ("carry_tau_s", "heuristic_emissions", "float", 5.0, 90.0),
        ("w_touch", "heuristic_emissions", "float", 0.0, 6.0),
        ("touch_tau_s", "heuristic_emissions", "float", 10.0, 120.0),
    ],
    "specialists": [
        ("scale_visible", "specialist_fusion", "float", 0.2, 1.5),
        ("scale_hidden", "specialist_fusion", "float", 0.2, 1.5),
        ("blend_visible", "specialist_fusion", "float", 0.0, 0.6),
        ("blend_hidden", "specialist_fusion", "float", 0.0, 0.6),
    ],
}
# sp_touch_min_score is the spell_optimizer's own `touch_min_score` (kept distinct from the transition one)
RENAMED = {"sp_touch_min_score": "touch_min_score"}


def space_for(route: str) -> list[tuple]:
    return SPACE["common"] + SPACE[route]


# ---------------------------------------------------------------------------------------------------
# Parameter objects
# ---------------------------------------------------------------------------------------------------
class Bundle:
    """Engine parameter dataclasses with the trial values applied."""

    def __init__(self, route: str, values: dict, cfg_path=None):
        from mtah2027.models.heuristic_emissions import EmissionParams
        from mtah2027.models.specialist_fusion import FusionParams
        from mtah2027.postprocessing.spell_optimizer import SpellParams
        from mtah2027.decoding.transition_matrix import TransitionParams

        v = dict(values)
        tv = {k: v[k] for k in ("ha_closed", "ha_open", "touch_halfwidth", "touch_min_score", "touch_both_dist_m",
                                "require_both_teams", "touch_gate_halfwidth", "ha_occluded", "occlusion_min_s",
                                "dead_closed", "dead_open", "d_exit_closed", "d_exit_open")}
        tv["dead_ramp"] = (v["dead_ramp_lo"], min(v["dead_ramp_lo"] + v["dead_ramp_w"], 1.0))
        tv["alive_ramp"] = (v["alive_ramp_lo"], min(v["alive_ramp_lo"] + v["alive_ramp_w"], 1.0))
        self.tp = dataclasses.replace(TransitionParams.from_config(cfg_path), **tv)
        sv = {RENAMED.get(n, n): v[n] for n, sec, *_ in SPACE["common"] if sec == "spell_optimizer"}
        self.sp = dataclasses.replace(SpellParams.from_config(cfg_path), **sv)
        self.ep = EmissionParams.from_config(cfg_path)
        self.fp = FusionParams.from_config(cfg_path)
        if route == "heuristic":
            self.ep = dataclasses.replace(self.ep, **{n: v[n] for n, *_ in SPACE["heuristic"]})
        else:
            self.fp = dataclasses.replace(self.fp, **{n: v[n] for n, *_ in SPACE["specialists"]})


def default_values(route: str, cfg_path=None) -> dict:
    """The engine's current defaults, in the trial-parameter naming."""
    from mtah2027.config import load_section

    out: dict = {}
    for name, sec, *_ in space_for(route):
        if name.endswith(("_ramp_lo", "_ramp_w")):
            continue
        out[name] = load_section(sec, cfg_path)[RENAMED.get(name, name)]
    for key in ("dead_ramp", "alive_ramp"):
        lo, hi = load_section("transition_matrix", cfg_path)[key]
        out[key + "_lo"], out[key + "_w"] = float(lo), float(hi) - float(lo)
    return out


def suggest(trial, route: str) -> dict:
    vals = {}
    for name, _sec, kind, lo, hi in space_for(route):
        if kind == "float":
            vals[name] = trial.suggest_float(name, lo, hi)
        elif kind == "int":
            vals[name] = trial.suggest_int(name, lo, hi)
        else:
            vals[name] = trial.suggest_categorical(name, lo)
    return vals


# ---------------------------------------------------------------------------------------------------
# Cache (parameter-independent part)
# ---------------------------------------------------------------------------------------------------
def cache_path(cache_dir: Path, stem: str, seed: int) -> Path:
    return cache_dir / f"{stem}_s{seed}.pkl"


def _build_worker(args):
    engine, bench_dir, cache_dir, stems, seeds, route, verify = args
    _engine_paths(Path(engine))
    import dataclasses as dc

    from mtah2027.models.external_degrader import degrade_match
    from mtah2027.pipeline import compute_front_end, heuristic_emissions, run_core
    from mtah2027.validation import eval_bridge as eb

    cache_dir = Path(cache_dir)
    spec_mod = None
    training = None
    if route == "specialists":
        from mtah2027.models import specialist_fusion as sf
        from mtah2027.models.specialist_data import build_training_set

        spec_mod = sf
        training = build_training_set(bench_dir, log=lambda m: None)
    report = []
    for stem in stems:
        z = np.load(Path(bench_dir) / f"{stem}.npz", allow_pickle=True)
        players, team = z["players_xy"].astype(np.float32), z["team_of_slot"].astype(np.int8)
        ball0 = z["ball_xy"].astype(np.float32)
        fps, pitch = float(z["fps"]), tuple(float(v) for v in z["pitch"])
        periods = [(int(a), int(b)) for a, b in z["period_bounds"]]
        gt = eb.to_codes(z["gt"])
        spec = None
        if route == "specialists":
            t0 = time.perf_counter()
            full = spec_mod.train_specialists(bench_dir, training=training, exclude_match=stem)
            raw_fp = dc.replace(full.fusion, scale_visible=1.0, scale_hidden=1.0, blend_visible=0.0,
                                blend_hidden=0.0)
            spec = dc.replace(full, fusion=raw_fp)
            report.append(f"trained specialists without {stem} ({time.perf_counter() - t0:.0f} s)")
        for i_s, seed in enumerate(seeds):
            out = cache_path(cache_dir, stem, seed)
            if out.exists():
                old = pickle.loads(out.read_bytes())
                if route != "specialists" or "learned_raw" in old:
                    continue
            # same per-match seed convention as scripts/eval_external.py (seed + index of the match in the dir)
            idx = sorted(p.stem for p in Path(bench_dir).glob("*.npz")).index(stem)
            dm = degrade_match(players, team, ball0, gt, fps=fps, seed=seed + idx)
            ball = dm.ball_xy
            blocks = periods

            def fe_of(b, quiet=True, _pl=dm.players_xy):
                return compute_front_end(_pl, team, b, play_blocks=blocks, fps=fps, pitch=pitch,
                                         collective=route == "specialists")

            fe = fe_of(ball)
            em = heuristic_emissions(fe, fps)
            bk, sig = fe.bk, fe.sig
            n = len(ball)
            support = np.zeros(n, bool)
            if sig is not None:
                support = (sig.geom_touchline > 0) | (sig.geom_corner > 0) | (sig.geom_goal_area > 0) | \
                          (sig.geom_kickoff > 0) | (sig.ball_out > 0)
            rec = {
                "stem": stem, "seed": seed, "fps": fps, "periods": periods, "gt": gt,
                "visible": fe.visible, "dead_score": fe.dead_score,
                "touch_score": bk.touch_score if bk is not None else None,
                "d_home": bk.d_home[:, 0] if bk is not None else None,
                "d_away": bk.d_away[:, 0] if bk is not None else None,
                "ball_out": sig.ball_out if sig is not None else None, "support": support,
                "evidence": bk.possession_evidence if bk is not None else np.zeros(n),
                "evidence_valid": bk.evidence_valid if bk is not None else np.zeros(n, bool),
                "status_dead": sig.status_dead if (sig is not None and sig.has_status) else None,
                "phase": fe.phase, "owner": fe.owner, "touches": fe.touches,
                "heur_lp": em.log_probs,
            }
            if route == "specialists":
                res = spec_mod.fuse_emissions(spec, fe, em, play_blocks=blocks, fps=fps, rebuild=fe_of,
                                              raw_ball=np.asarray(ball, np.float32))
                rec["learned_raw"] = res.emissions.log_probs
                if verify and i_s == 0:
                    real = dc.replace(full, fusion=full.fusion)
                    core = run_core(dm.players_xy, team, ball, play_blocks=blocks, fps=fps, pitch=pitch,
                                    specialists=dc.replace(real, fusion=spec_mod.FusionParams.from_config()),
                                    log=lambda m: None)
                    rec["verify_codes"] = core.codes
            elif verify and i_s == 0:
                core = run_core(dm.players_xy, team, ball, play_blocks=blocks, fps=fps, pitch=pitch,
                                log=lambda m: None)
                rec["verify_codes"] = core.codes
            tmp = out.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(rec, protocol=4))
            for attempt in range(8):
                try:
                    os.replace(tmp, out)
                    break
                except PermissionError:
                    time.sleep(1.0 + attempt)
            report.append(f"cached {stem} seed {seed}")
    return report


def build_cache(engine: Path, bench_dir: Path, cache_dir: Path, stems: list[str], seeds: list[int],
                route: str, workers: int, verify: bool) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunks = [stems[i::workers] for i in range(min(workers, len(stems)))]
    jobs = [(str(engine), str(bench_dir), str(cache_dir), c, seeds, route, verify) for c in chunks if c]
    if len(jobs) == 1:
        results = [_build_worker(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            results = list(ex.map(_build_worker, jobs))
    for r in results:
        for line in r:
            print("  " + line, flush=True)


def load_records(cache_dir: Path, stems: list[str], seeds: list[int], route: str) -> dict:
    recs: dict = {}
    for seed in seeds:
        recs[seed] = []
        for stem in stems:
            rec = pickle.loads(cache_path(cache_dir, stem, seed).read_bytes())
            if route == "specialists" and "learned_raw" not in rec:
                raise SystemExit(f"cache of {stem} seed {seed} lacks the specialist emissions; rebuild with "
                                 "--route specialists --rebuild-cache")
            recs[seed].append(rec)
    return recs


# ---------------------------------------------------------------------------------------------------
# Scoring one record with a parameter bundle
# ---------------------------------------------------------------------------------------------------
def decode_record(rec: dict, route: str, b: Bundle) -> np.ndarray:
    from mtah2027.decoding.dynamic_viterbi import decode_match_halves, default_log_init
    from mtah2027.decoding.transition_matrix import build_log_transitions
    from mtah2027.models.heuristic_emissions import compute_emissions
    from mtah2027.postprocessing.spell_optimizer import optimize_spells

    fps, blocks = rec["fps"], rec["periods"]
    n = len(rec["gt"])
    if route == "heuristic":
        lp = compute_emissions(
            rec["dead_score"], rec["evidence"], rec["evidence_valid"], rec["visible"],
            status_dead=rec["status_dead"], phase_prior_home=rec["phase"], owner_team=rec["owner"],
            touch_signed=rec["touches"], fps=fps, params=b.ep).log_probs
    else:
        vis = rec["visible"]
        fp = b.fp
        scale = np.where(vis, fp.scale_visible, fp.scale_hidden)[:, None]
        blend = np.where(vis, fp.blend_visible, fp.blend_hidden)[:, None]
        lp = np.round(scale * ((1.0 - blend) * rec["learned_raw"] + blend * rec["heur_lp"]), 6)
    lt = build_log_transitions(
        n, touch_score=rec["touch_score"], d_home=rec["d_home"], d_away=rec["d_away"],
        ball_visible=rec["visible"], dead_score=rec["dead_score"], ball_out=rec["ball_out"], fps=fps,
        params=b.tp)
    codes = decode_match_halves(lp, lt, play_blocks=blocks, dead_blocks=(), log_init=default_log_init())
    codes, _ = optimize_spells(
        codes, play_blocks=blocks, ball_visible=rec["visible"], touch_score=rec["touch_score"],
        dead_score=rec["dead_score"], dead_support=rec["support"], forced_dead_mask=None, fps=fps,
        params=b.sp)
    return codes


def score_records(recs: list[dict], route: str, values: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-record ``(spellF1, macroF1)`` of the parameter values."""
    from mtah2027.validation import eval_bridge as eb

    b = Bundle(route, values)
    sp, mf = [], []
    for rec in recs:
        r = eb.evaluate_arrays(rec["gt"], decode_record(rec, route, b))
        sp.append(float(r["spell_f1_mean"]))
        mf.append(float(r["macro_f1"]))
    return np.array(sp), np.array(mf)


def per_match(recs_by_seed: dict, seeds: list[int], route: str, values: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(spell[M], macro[M])`` averaged over ``seeds``."""
    sps, mfs = zip(*(score_records(recs_by_seed[s], route, values) for s in seeds))
    return np.mean(sps, axis=0), np.mean(mfs, axis=0)


def _nested_fold(job):
    """One truly nested fold: search on every match but ``held``, score the winner on ``held``."""
    engine, cache_dir, stems, seeds, check_seeds, route, held, trials, timeout = job
    _engine_paths(Path(engine))
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    all_seeds = seeds + check_seeds
    recs = load_records(Path(cache_dir), stems, all_seeds, route)
    keep = [i for i, s in enumerate(stems) if i != held]
    sub = {s: [recs[s][i] for i in keep] for s in all_seeds}
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0, multivariate=True))
    study.enqueue_trial(default_values(route))

    def objective(trial):
        sp, mf = per_match(sub, seeds, route, suggest(trial, route))
        return float(sp.mean() + 1e-4 * mf.mean())

    study.optimize(objective, n_trials=trials, timeout=timeout)
    one = {s: [recs[s][held]] for s in all_seeds}
    got = {}
    for name, ss in (("search_seeds", seeds), ("check_seeds", check_seeds)):
        if ss:
            got[name] = (float(per_match(one, ss, route, study.best_params)[0][0]),
                         float(per_match(one, ss, route, default_values(route))[0][0]))
    return held, got


def run_nested(args, stems, seeds, check_seeds, cache_dir):
    jobs = [(args.engine, str(cache_dir), stems, seeds, check_seeds, args.route, m, args.nested_trials,
             args.nested_timeout) for m in range(len(stems))]
    with ProcessPoolExecutor(max_workers=min(len(jobs), args.workers)) as ex:
        res = dict(ex.map(_nested_fold, jobs))
    out = {}
    for key in ("search_seeds", "check_seeds"):
        if all(key in r for r in res.values()):
            tuned = [res[m][key][0] for m in range(len(stems))]
            dflt = [res[m][key][1] for m in range(len(stems))]
            out[key] = {"tuned_spellF1": float(np.mean(tuned)), "default_spellF1": float(np.mean(dflt)),
                        "gain": float(np.mean(tuned) - np.mean(dflt)), "per_match_gain": [t - d for t, d in zip(tuned, dflt)]}
    return out


# ---------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--route", choices=("heuristic", "specialists"), default="heuristic")
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--timeout", type=float, default=5400.0, help="seconds for the search (default 90 min)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="degradation seeds used by the search")
    ap.add_argument("--check-seeds", type=int, nargs="*", default=[1],
                    help="degradation seeds NOT used by the search (overfitting check)")
    ap.add_argument("--export", default=None, help="write the best parameters as YAML (engine config sections)")
    ap.add_argument("--engine", default=str(DEFAULT_ENGINE), help="engine checkout (src/ is put on sys.path)")
    ap.add_argument("--bench-dir", default=str(ROOT / "cache"))
    ap.add_argument("--cache-dir", default=str(ROOT / "cache" / "optuna"))
    ap.add_argument("--matches", nargs="*", default=None, help="match stems (default: every idsse_*)")
    ap.add_argument("--workers", type=int, default=4, help="processes of the cache build")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="check that the cached decoder reproduces the engine run_core at the defaults")
    ap.add_argument("--report", default=None)
    ap.add_argument("--nested-trials", type=int, default=0,
                    help="truly nested leave-one-match-out: per held-out match a separate search of this many "
                         "trials on the other matches (0 = skip)")
    ap.add_argument("--nested-timeout", type=float, default=2400.0)
    args = ap.parse_args(argv)

    engine = Path(args.engine)
    _engine_paths(engine)
    bench, cache_dir = Path(args.bench_dir), Path(args.cache_dir)
    stems = args.matches or sorted(p.stem for p in bench.glob("idsse_*.npz"))
    if not stems:
        print("no idsse_*.npz benchmark files")
        return 1
    seeds = list(args.seeds)
    check_seeds = [s for s in args.check_seeds if s not in seeds]
    all_seeds = seeds + check_seeds
    if args.rebuild_cache:
        for p in cache_dir.glob("*.pkl"):
            p.unlink()
    t0 = time.perf_counter()
    print(f"route={args.route} matches={len(stems)} search_seeds={seeds} check_seeds={check_seeds}", flush=True)
    build_cache(engine, bench, cache_dir, stems, all_seeds, args.route, args.workers, args.verify)
    print(f"cache ready ({time.perf_counter() - t0:.0f} s)", flush=True)
    if args.build_only:
        return 0
    recs = load_records(cache_dir, stems, all_seeds, args.route)

    defaults = default_values(args.route)
    if args.verify:
        for rec in recs[seeds[0]]:
            if "verify_codes" not in rec:
                continue
            mine = decode_record(rec, args.route, Bundle(args.route, defaults))
            same = bool(np.array_equal(mine, rec["verify_codes"]))
            print(f"  verify {rec['stem']}: cached decoder == engine run_core: {same}", flush=True)
            if not same:
                raise SystemExit("cached decoder does not reproduce the engine pipeline")

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    t_score = time.perf_counter()
    d_sp, d_mf = per_match(recs, seeds, args.route, defaults)
    print(f"default: spellF1 {d_sp.mean():.4f} macroF1 {d_mf.mean():.4f} "
          f"(one evaluation {time.perf_counter() - t_score:.1f} s)", flush=True)

    study_dir = ROOT / "optuna_studies"
    study_dir.mkdir(exist_ok=True)
    storage = f"sqlite:///{(study_dir / f'{args.route}.db').as_posix()}"
    try:
        optuna.delete_study(study_name=args.route, storage=storage)
    except KeyError:
        pass
    study = optuna.create_study(study_name=args.route, storage=storage, direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0, multivariate=True))
    study.enqueue_trial(defaults)

    def objective(trial):
        vals = suggest(trial, args.route)
        sp, mf = per_match(recs, seeds, args.route, vals)
        trial.set_user_attr("spell", [float(x) for x in sp])
        trial.set_user_attr("macro", [float(x) for x in mf])
        return float(sp.mean() + 1e-4 * mf.mean())      # macroF1 only breaks ties

    t_search = time.perf_counter()
    last = [t_search]

    def progress(st, tr):
        if tr.number % 10 == 0 or time.perf_counter() - last[0] > 120:
            last[0] = time.perf_counter()
            print(f"  trial {tr.number:4d}  best {st.best_value:.4f}  "
                  f"({(time.perf_counter() - t_search) / (tr.number + 1):.1f} s/trial)", flush=True)

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, callbacks=[progress])
    search_s = time.perf_counter() - t_search

    trials = [t for t in study.trials if t.state.name == "COMPLETE"]
    S = np.array([t.user_attrs["spell"] for t in trials])          # [T, M]
    F = np.array([t.user_attrs["macro"] for t in trials])
    best_i = int(np.argmax([t.value for t in trials]))
    best = trials[best_i]
    # --- nested leave-one-match-out selection over the pool of trials --------------------------------
    M = S.shape[1]
    sel = []
    for m in range(M):
        others = np.delete(S, m, axis=1).mean(axis=1)
        sel.append(int(np.argmax(others)))
    nested_sp = float(np.mean([S[sel[m], m] for m in range(M)]))
    nested_mf = float(np.mean([F[sel[m], m] for m in range(M)]))
    # --- degradation seeds not used by the search -------------------------------------------------------
    check = {}
    if check_seeds:
        bv = dict(best.params)
        cs_d, cm_d = per_match(recs, check_seeds, args.route, defaults)
        cs_b, cm_b = per_match(recs, check_seeds, args.route, bv)
        nested_check = []
        cache_sel: dict = {}
        for m in range(M):
            i = sel[m]
            if i not in cache_sel:
                cache_sel[i] = per_match(recs, check_seeds, args.route, dict(trials[i].params))
            nested_check.append(cache_sel[i][0][m])
        check = {"seeds": check_seeds,
                 "default_spellF1": float(cs_d.mean()), "best_spellF1": float(cs_b.mean()),
                 "nested_spellF1": float(np.mean(nested_check)),
                 "default_macroF1": float(cm_d.mean()), "best_macroF1": float(cm_b.mean()),
                 "per_match_default": [float(x) for x in cs_d], "per_match_best": [float(x) for x in cs_b]}
    nested_true = {}
    if args.nested_trials > 0:
        t_n = time.perf_counter()
        nested_true = run_nested(args, stems, seeds, check_seeds, cache_dir)
        nested_true["seconds"] = round(time.perf_counter() - t_n, 1)
        print(f"nested (true) {nested_true}", flush=True)
    default_trial = trials[0]
    rep = {
        "route": args.route, "matches": stems, "search_seeds": seeds, "n_trials": len(trials),
        "search_seconds": round(search_s, 1), "seconds_per_trial": round(search_s / max(len(trials), 1), 2),
        "default": {"spellF1": float(d_sp.mean()), "macroF1": float(d_mf.mean()),
                    "per_match": [float(x) for x in d_sp], "params": defaults},
        "best": {"trial": best.number, "spellF1": float(S[best_i].mean()), "macroF1": float(F[best_i].mean()),
                 "per_match": [float(x) for x in S[best_i]], "params": best.params},
        "nested_honest": {"spellF1": nested_sp, "macroF1": nested_mf,
                          "gain_vs_default": nested_sp - float(d_sp.mean()),
                          "selected_trials": sel},
        "check_seeds": check, "nested_true": nested_true,
        "default_trial_in_study_spellF1": float(S[0].mean()) if default_trial.number == 0 else None,
    }
    gain = rep["nested_honest"]["gain_vs_default"]
    if nested_true.get("search_seeds"):
        gain = nested_true["search_seeds"]["gain"]      # the truly nested estimate overrides the pool one
    chk_gain = (check["best_spellF1"] - check["default_spellF1"]) if check else None
    rep["recommend_change"] = bool(gain >= 0.02 and (chk_gain is None or chk_gain > 0.0))
    rep["recommend_rule"] = "(true nested if run, else pool-LOMO) honest gain >= 0.02 spellF1 and best beats default on the unused seeds"
    out = Path(args.report) if args.report else ROOT / "reports" / f"optuna_{args.route}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")

    print(f"\n== {args.route}: {len(trials)} trials in {search_s:.0f} s ({rep['seconds_per_trial']} s/trial)")
    print(f"default          spellF1 {d_sp.mean():.4f}  macroF1 {d_mf.mean():.4f}")
    print(f"best (search)    spellF1 {S[best_i].mean():.4f}  macroF1 {F[best_i].mean():.4f}")
    print(f"nested honest    spellF1 {nested_sp:.4f}  macroF1 {nested_mf:.4f}  (gain {gain:+.4f})")
    if check:
        print(f"check seeds {check_seeds}: default {check['default_spellF1']:.4f}  best {check['best_spellF1']:.4f}"
              f"  nested {check['nested_spellF1']:.4f}")
    print(f"recommend change: {rep['recommend_change']}")

    if args.export:
        import yaml

        sections: dict = {}
        for name, sec, *_ in space_for(args.route):
            if name.endswith(("_ramp_lo", "_ramp_w")):
                continue
            sections.setdefault(sec, {})[RENAMED.get(name, name)] = best.params[name]
        sections["transition_matrix"]["dead_ramp"] = [
            round(best.params["dead_ramp_lo"], 4), round(min(best.params["dead_ramp_lo"] + best.params["dead_ramp_w"], 1), 4)]
        sections["transition_matrix"]["alive_ramp"] = [
            round(best.params["alive_ramp_lo"], 4), round(min(best.params["alive_ramp_lo"] + best.params["alive_ramp_w"], 1), 4)]
        clean = {s: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
                 for s, d in sections.items()}
        Path(args.export).write_text(yaml.safe_dump(clean, sort_keys=False), encoding="utf-8")
        print(f"exported {args.export}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
