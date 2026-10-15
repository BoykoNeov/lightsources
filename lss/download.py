"""Data the repository may not hold, downloaded on first use into a local cache.

Each file is pinned by its SHA-256: a download that differs from the pinned one is
refused, so every result computed from it is reproducible.
"""

import hashlib
import os
import urllib.request
from pathlib import Path

USER_AGENT = "lss (light source simulator)"


def cache_dir() -> Path:
    """``$LSS_CACHE``, or ``~/.cache/lss``."""
    return Path(os.environ.get("LSS_CACHE") or Path.home() / ".cache" / "lss")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str, path: Path, digest: str) -> Path:
    """``path``, downloaded from ``url`` first if absent; its SHA-256 is checked either way."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
    got = sha256(path)
    if got != digest:
        raise ValueError(f"{path}: SHA-256 {got} is not the pinned file's ({digest}); delete it to refetch")
    return path
