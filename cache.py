"""Disk cache for computed pipeline artifacts.

Keys are stable hashes of the inputs (prompt + config), values are pickled
objects (StateTrajectory, coords, Landscape, TerrainMesh, ...).  This is what
keeps repeated explorations of the same prompt under the interactive budget.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any


def default_dir(kind: str | None = None) -> Path:
    """Mottled's per-user cache root (or its `kind` subdirectory).

    Resolution order, first hit wins:
      1. ``$MOTTLED_CACHE_DIR`` (``$MOTTLED_CACHE``, the older name, also works)
      2. ``$XDG_CACHE_HOME/mottled``
      3. the platform's cache home: ``~/Library/Caches/mottled`` (macOS),
         ``%LOCALAPPDATA%\\mottled\\Cache`` (Windows), ``~/.cache/mottled``

    Never the working directory: a cache that follows `cwd` is a cache two
    projects silently share or one project silently loses.
    """
    env = os.environ.get("MOTTLED_CACHE_DIR") or os.environ.get("MOTTLED_CACHE")
    if env:
        root = Path(env).expanduser()
    elif os.environ.get("XDG_CACHE_HOME"):
        root = Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "mottled"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches" / "mottled"
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"]) / "mottled" / "Cache"
    else:
        root = Path.home() / ".cache" / "mottled"
    return root / kind if kind else root


def make_key(*parts: Any, **kw: Any) -> str:
    """Stable hash of arbitrary JSON-serialisable inputs."""
    blob = json.dumps([parts, kw], sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


class DiskCache:
    def __init__(self, directory: str | Path | None = None):
        self.dir = Path(directory) if directory else default_dir("pipeline")
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.pkl"

    def __contains__(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, key: str, default: Any = None) -> Any:
        path = self._path(key)
        if not path.exists():
            return default
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except Exception:
            path.unlink(missing_ok=True)  # corrupt entry: drop it
            return default

    def put(self, key: str, value: Any) -> None:
        tmp = self._path(key).with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self._path(key))

    def clear(self) -> None:
        for p in self.dir.glob("*.pkl"):
            p.unlink(missing_ok=True)
