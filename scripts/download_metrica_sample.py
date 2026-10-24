#!/usr/bin/env python
"""Download Metrica Sports open sample data (Games 1-3) into data/metrica/.

Raw files are fetched from https://github.com/metrica-sports/sample-data and are
later parsed with kloppy (``kloppy.metrica.load_tracking_csv`` / ``load_tracking_epts``
/ ``load_event``) by ``mtah_bench.convert``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mtah_bench.fetch import download  # noqa: E402

BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
FILES = {
    "Sample_Game_1": [
        "Sample_Game_1_RawEventsData.csv",
        "Sample_Game_1_RawTrackingData_Home_Team.csv",
        "Sample_Game_1_RawTrackingData_Away_Team.csv",
    ],
    "Sample_Game_2": [
        "Sample_Game_2_RawEventsData.csv",
        "Sample_Game_2_RawTrackingData_Home_Team.csv",
        "Sample_Game_2_RawTrackingData_Away_Team.csv",
    ],
    "Sample_Game_3": [
        "Sample_Game_3_events.json",
        "Sample_Game_3_metadata.xml",
        "Sample_Game_3_tracking.txt",
    ],
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "metrica")
    ap.add_argument("--games", nargs="*", default=list(FILES), choices=list(FILES))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    for g in a.games:
        for fn in FILES[g]:
            download(f"{BASE}/{g}/{fn}", a.out / g / fn, force=a.force)


if __name__ == "__main__":
    main()
