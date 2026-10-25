#!/usr/bin/env python
"""Download SkillCorner open data (MIT, https://github.com/SkillCorner/opendata) into
data/skillcorner/<match_id>/. No registration needed. ~90 MB per match (tracking via Git LFS
through media.githubusercontent.com); 20 matches = ~1.8 GB.

    python scripts/download_skillcorner.py --matches all
    python scripts/download_skillcorner.py --matches 2017461 2016236
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mtah_bench.fetch import download  # noqa: E402

RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data"
LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data"


def list_ids() -> list[str]:
    req = urllib.request.Request(f"{RAW}/matches.json", headers={"User-Agent": "mtah-bench/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return [str(m["id"]) for m in json.load(r)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "skillcorner")
    ap.add_argument("--matches", nargs="*", default=["all"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    ids = list_ids() if a.matches == ["all"] else a.matches
    for m in ids:
        for name, base in (("match.json", RAW), ("phases_of_play.csv", RAW),
                           ("tracking_extrapolated.jsonl", LFS)):
            download(f"{base}/matches/{m}/{m}_{name}", a.out / m / f"{m}_{name}", force=a.force)


if __name__ == "__main__":
    main()
