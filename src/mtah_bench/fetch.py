"""Tiny resumable-ish HTTP download helper (stdlib only)."""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path


def download(url: str, dest: Path, *, force: bool = False) -> Path:
    """Download ``url`` to ``dest`` atomically; skip if it already exists."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"[skip] {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "mtah-bench/1.0"})
    print(f"[get ] {url}")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    if tmp.stat().st_size == 0:
        tmp.unlink()
        raise RuntimeError(f"empty response from {url} (blocked or bad id)")
    tmp.replace(dest)
    print(f"[done] {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest
