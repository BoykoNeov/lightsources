"""Code version for provenance records (design §11, "provenance everywhere")."""

import subprocess
from functools import lru_cache
from pathlib import Path

import lss


@lru_cache(maxsize=1)
def code_version() -> str:
    """Package version plus the git commit when running from a checkout."""
    root = Path(lss.__file__).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "describe", "--always", "--dirty"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return f"lss {lss.__version__} ({out.stdout.strip()})"
    except (OSError, subprocess.SubprocessError):
        return f"lss {lss.__version__}"
