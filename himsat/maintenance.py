"""Housekeeping: keep the data volume bounded on long-running deployments."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from himsat.config import Settings, get_settings

log = logging.getLogger(__name__)


def _prune(root: Path, pattern: str, older_than_days: float, dry_run: bool) -> tuple[int, int]:
    if not root.exists():
        return 0, 0
    cutoff = time.time() - older_than_days * 86400
    n = size = 0
    for f in root.rglob(pattern):
        if f.is_file() and f.stat().st_mtime < cutoff:
            n += 1
            size += f.stat().st_size
            if not dry_run:
                f.unlink(missing_ok=True)
    return n, size


def prune(settings: Settings | None = None, *, cache_days: float = 120, velocity_days: float = 400,
          dry_run: bool = False) -> dict:
    """Delete raster-cache files and velocity fields older than the given ages.

    Velocity fields are kept > 1 year so watch-cell and same-season baselines remain available.
    S2 composites are never pruned (one file per tile, continuously updated).
    """
    s = settings or get_settings()
    c_n, c_b = _prune(s.cache_dir / "rasters", "*.tif", cache_days, dry_run)
    v_n, v_b = _prune(s.products_dir, "velocity/*/*.npz", velocity_days, dry_run)
    res = {"cache_files": c_n, "cache_mb": round(c_b / 1e6, 1), "velocity_files": v_n,
           "velocity_mb": round(v_b / 1e6, 1), "dry_run": dry_run}
    log.info("prune: %s", res)
    return res
