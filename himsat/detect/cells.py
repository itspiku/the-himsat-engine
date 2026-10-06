"""Watch cells: systematic slow-motion monitoring of every steep, high ice/rock slope.

Slope-failure precursors are often a few cm/day. At that speed a single Sentinel-1 pair (noise
≈ 0.1 m/day per node) cannot see anything. Two steps make it detectable:

* **spatial aggregation**: the median downslope velocity of all nodes in a ≈1 km² cell
  (standard error ≈ node noise / √(independent chips))
* **temporal stacking**: inverse-variance weighting of all pairs (several orbits, 12–36-day
  baselines) in a recent window, compared with the same cell's own baseline months

A cell is anomalous when its recent velocity is significantly (z-score), materially (ratio,
absolute speed) and independently (≥ 2 orbits) above its baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from himsat.detect.velocity import VelocityField, downslope
from himsat.geo.grid import Grid, Tile


@dataclass
class CellStat:
    cell: str
    v: float
    se: float
    n: int


def tile_offset(aoi_grid: Grid, tile: Tile) -> tuple[int, int]:
    g = tile.grid
    return int(round((aoi_grid.y0 - g.y0) / g.res)), int(round((g.x0 - aoi_grid.x0) / g.res))


def cell_stats(vf: VelocityField, aspect: np.ndarray, watch: np.ndarray, tile: Tile, aoi_grid: Grid,
               noise_px: float, cell_m: float = 1000.0, chip: int = 64, min_nodes: int = 8) -> list[CellStat]:
    d = downslope(vf, aspect)
    rr, cc = np.meshgrid(vf.rows, vf.cols, indexing="ij")
    ok = np.isfinite(d) & vf.lattice_mask(watch)
    core = ((rr >= tile.core_row0) & (rr < tile.core_row0 + tile.core_height)
            & (cc >= tile.core_col0) & (cc < tile.core_col0 + tile.core_width))
    ok &= core
    if not ok.any():
        return []
    r0, c0 = tile_offset(aoi_grid, tile)
    cell_px = max(1, int(round(cell_m / vf.grid.res)))
    cr = (rr[ok] + r0) // cell_px
    ccol = (cc[ok] + c0) // cell_px
    vals = d[ok]
    step = float(vf.rows[1] - vf.rows[0]) if vf.rows.size > 1 else float(chip)
    node_sd = noise_px * vf.grid.res / vf.dt_days
    out = []
    keys = cr.astype(np.int64) * 100000 + ccol
    order = np.argsort(keys, kind="stable")
    keys, vals = keys[order], vals[order]
    bounds = np.flatnonzero(np.diff(keys)) + 1
    for grp_v, k in zip(np.split(vals, bounds), keys[np.r_[0, bounds]], strict=True):
        n = grp_v.size
        if n < min_nodes:
            continue
        n_eff = max(1.0, n * (step / chip) ** 2)
        se = 1.2533 * node_sd / math.sqrt(n_eff)  # standard error of a median
        out.append(CellStat(f"{int(k // 100000)}_{int(k % 100000)}", float(np.median(grp_v)), float(se), int(n)))
    return out


def cell_polygon(cell: str, aoi_grid: Grid, cell_m: float = 1000.0):
    from shapely.geometry import box

    from himsat.geo.grid import reproject_geom

    r, c = (int(x) for x in cell.split("_"))
    cell_px = int(round(cell_m / aoi_grid.res))
    x0 = aoi_grid.x0 + c * cell_px * aoi_grid.res
    y0 = aoi_grid.y0 - r * cell_px * aoi_grid.res
    return reproject_geom(box(x0, y0 - cell_m, x0 + cell_m, y0), aoi_grid.epsg, 4326)


def neighbours(cell: str) -> list[str]:
    r, c = (int(x) for x in cell.split("_"))
    return [f"{r + dr}_{c + dc}" for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def weighted_mean(vs: list[float], ses: list[float], clip: float = 3.5,
                  min_se: float = 1e-3) -> tuple[float, float] | None:
    """Inverse-variance mean with one pass of outlier rejection (``min_se`` guards against zero errors)."""
    if not vs:
        return None
    v = np.asarray(vs, float)
    s = np.maximum(np.asarray(ses, float), min_se)
    w = 1 / s**2
    m = float(np.sum(w * v) / np.sum(w))
    keep = np.abs(v - np.median(v)) <= clip * s + 1e-9
    if keep.sum() >= max(1, len(v) // 2):
        v, s, w = v[keep], s[keep], w[keep]
        m = float(np.sum(w * v) / np.sum(w))
    return m, float(1 / math.sqrt(np.sum(w)))


# Glaciers and permafrost slopes speed up every melt season. A near-zero spring baseline must not
# turn ordinary summer motion of a few cm/day into a "10x" anomaly.
MIN_V = 0.05  # m/day, recent downslope velocity required
RATIO_FLOOR = 0.03  # m/day, smallest baseline used as the ratio denominator


@dataclass
class Anomaly:
    v_recent: float
    se_recent: float
    v_base: float
    se_base: float
    z: float
    ratio: float
    n_recent: int
    orbits_confirming: int

    @property
    def significant(self) -> bool:
        return (self.z >= 3.0 and self.v_recent >= MIN_V and self.ratio >= 2.0 and self.orbits_confirming >= 2)


def anomaly(rows: list[tuple[datetime, float, float, int | None]], t: datetime, recent_days: float = 16,
            base_days: tuple[float, float] = (30, 150), min_base: int = 3) -> Anomaly | None:
    """rows: (pair_end, v, se, orbit). Compare the recent window with the cell's own baseline."""
    rec = [r for r in rows if t - timedelta(days=recent_days) < r[0] <= t]
    base = [r for r in rows if t - timedelta(days=base_days[1]) <= r[0] <= t - timedelta(days=base_days[0])]
    if len(rec) < 2 or len(base) < min_base:
        return None
    wr = weighted_mean([r[1] for r in rec], [r[2] for r in rec])
    wb = weighted_mean([r[1] for r in base], [r[2] for r in base])
    if wr is None or wb is None:
        return None
    (vr, sr), (vb, sb) = wr, wb
    z = (vr - vb) / math.hypot(sr, sb)
    ratio = vr / max(vb, RATIO_FLOOR)
    confirming = {r[3] for r in rec if r[1] - vb > r[2]}
    return Anomaly(vr, sr, vb, sb, z, ratio, len(rec), len(confirming))
