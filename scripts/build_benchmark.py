#!/usr/bin/env python
"""Download (optional) + convert all external matches; write cache/<match>.npz and
reports/benchmark_summary.json (aggregate stats only, no per-frame data).

    python scripts/build_benchmark.py --download
    python scripts/build_benchmark.py --only metrica_1 idsse_J03WPY
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mtah_bench import convert as C  # noqa: E402
from mtah_bench.pff import convert_pff, pff_game_dirs  # noqa: E402
from mtah_bench.skillcorner import SC_MATCHES, convert_skillcorner  # noqa: E402

ALL = list(C.METRICA_GAMES) + [f"idsse_{m}" for m in C.IDSSE_MATCHES] + [f"skillcorner_{m}" for m in SC_MATCHES]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true", help="fetch raw data first (idempotent)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="default: every match whose raw data is present under --data")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--cache", type=Path, default=ROOT / "cache")
    ap.add_argument("--report", type=Path, default=ROOT / "reports" / "benchmark_summary.json")
    a = ap.parse_args(argv)

    if a.only is None:
        a.only = [n for n in ALL if (a.data / ("metrica" if n.startswith("metrica") else n.split("_")[0]) /
                                     (C.METRICA_GAMES.get(n) or n.split("_", 1)[1])).exists()]
    a.only = a.only + [f"pff_{g}" for g in pff_game_dirs(a.data) if f"pff_{g}" not in a.only] if a.only else a.only
    for n in a.only:
        if n not in ALL and not n.startswith("pff_"):
            ap.error(f"unknown match {n}")
    if a.download:
        sys.path.insert(0, str(ROOT / "scripts"))
        from download_idsse import main as dl_idsse
        from download_metrica_sample import main as dl_metrica

        if any(n.startswith("metrica") for n in a.only):
            dl_metrica(["--out", str(a.data / "metrica")])
        ids = [n.split("_", 1)[1] for n in a.only if n.startswith("idsse_")]
        if ids:
            dl_idsse(["--out", str(a.data / "idsse"), "--matches", *ids])

    per_match = {}
    for n in a.only:
        t = time.time()
        if n.startswith("metrica"):
            p = C.convert_metrica(a.data, n, a.cache)
        elif n.startswith("pff_"):
            p = convert_pff(a.data, n.split("_", 1)[1], a.cache)
        elif n.startswith("skillcorner_"):
            p = convert_skillcorner(a.data, n.split("_", 1)[1], a.cache)
        else:
            p = C.convert_idsse(a.data, n.split("_", 1)[1], a.cache)
        per_match[n] = C.stats_of_npz(p)
        print(f"{n}: {per_match[n]}  [{time.time() - t:.0f}s]", flush=True)

    agg = {
        "matches": len(per_match),
        "total_possessions": sum(m["possessions"] for m in per_match.values()),
        "total_evaluated_minutes": round(sum(m["evaluated_minutes"] for m in per_match.values()), 1),
        "mean_possession_s_avg": round(sum(m["mean_possession_s"] for m in per_match.values()) / len(per_match), 2),
        "pct_D_avg": round(sum(m["pct_D"] for m in per_match.values()) / len(per_match), 2),
        "competition_public_5min_reference": {"possessions": 17, "mean_possession_s": 14.2, "pct_D_approx": 20},
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps({"aggregate": agg, "per_match": per_match}, indent=2), encoding="utf-8")
    print(f"wrote {a.report}")


if __name__ == "__main__":
    main()
