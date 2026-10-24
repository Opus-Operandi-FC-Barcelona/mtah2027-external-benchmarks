#!/usr/bin/env python
"""Download the IDSSE open Bundesliga dataset (Bassek et al., Sci Data 12, 195, 2025;
CC BY 4.0, figshare) into data/idsse/<match_id>/{meta,event,tracking}.xml.

Seven matches exist (~0.5 GB each of tracking XML). Default: all seven (~2.5 GB total);
use ``--matches J03WPY J03WMX`` to fetch fewer.
File ids come from ``kloppy.sportec.get_IDSSE_url`` (kloppy 3.18).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kloppy._providers.sportec import get_IDSSE_url  # noqa: E402

from mtah_bench.fetch import download  # noqa: E402

ALL_MATCHES = ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]


def direct_url(match_id: str, kind: str) -> str:
    """kloppy's springernature.figshare.com URL is behind an AWS WAF JS challenge
    (returns an empty 202); ndownloader.figshare.com serves the same file ids."""
    return get_IDSSE_url(match_id, kind).replace(
        "springernature.figshare.com/ndownloader", "ndownloader.figshare.com"
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "idsse")
    ap.add_argument("--matches", nargs="*", default=ALL_MATCHES, help="match ids or: all")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    ids = ALL_MATCHES if a.matches == ["all"] else a.matches
    for m in ids:
        for kind in ("meta", "event", "tracking"):
            download(direct_url(m, kind), a.out / m / f"{kind}.xml", force=a.force)


if __name__ == "__main__":
    main()
