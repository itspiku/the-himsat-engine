"""Downstream flow routing and exposure of settlements and infrastructure.

For each monitored site the flood path is traced downstream over the Copernicus DEM by steepest
descent (D8). Pits are escaped with a local priority flood. An asset counts as exposed when it
lies close to the path both horizontally *and* vertically, using a height-above-channel test with
a threshold that decays downstream as the flood wave attenuates. The arrival time at each asset
is the path distance divided by a hazard-specific wave speed.
"""

from __future__ import annotations

import heapq
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from himsat.geo.grid import Grid, from_lonlat, reproject_geom, to_lonlat

log = logging.getLogger(__name__)

_D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_downstream(dem: np.ndarray, start: tuple[int, int], res: float, max_km: float = 150.0,
                     max_pit_cells: int = 200_000) -> list[tuple[int, int]]:
    """Steepest-descent path from ``start`` until the grid edge or ``max_km``.

    When the path hits a pit or flat (lakes, DEM artefacts, valley-floor flats), it floods
    outward from the pit in order of elevation until it reaches a cell lower than the pit, then
    continues from there. The breach route is added to the path, so the path stays connected.
    """
    h, w = dem.shape
    r, c = start
    path = [(r, c)]
    visited = {(r, c)}
    length = 0.0
    max_len = max_km * 1000.0
    while length < max_len:
        z = dem[r, c]
        best, best_drop = None, 0.0
        for dr, dc in _D8:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                return path  # left the grid: done
            d = res * (1.4142135 if dr and dc else 1.0)
            drop = (z - dem[rr, cc]) / d
            if drop > best_drop and (rr, cc) not in visited:
                best, best_drop = (rr, cc), drop
        if best is None:
            breach = _escape_pit(dem, (r, c), visited, max_pit_cells)
            if not breach:
                log.debug("trace stopped in closed depression at %s", (r, c))
                return path
            for cell in breach:
                step = res * (1.4142135 if (cell[0] != r and cell[1] != c) else 1.0)
                length += step
                r, c = cell
                path.append(cell)
                visited.add(cell)
                if not (0 < r < h - 1 and 0 < c < w - 1):
                    return path
            continue
        length += res * (1.4142135 if (best[0] != r and best[1] != c) else 1.0)
        r, c = best
        path.append(best)
        visited.add(best)
    return path


def _escape_pit(dem: np.ndarray, pit: tuple[int, int], visited: set, max_cells: int) -> list[tuple[int, int]]:
    """Priority flood from a pit to the first lower (or edge) cell; returns the route to it."""
    h, w = dem.shape
    z0 = dem[pit]
    heap = [(float(z0), pit)]
    parent: dict[tuple[int, int], tuple[int, int] | None] = {pit: None}
    n = 0
    while heap and n < max_cells:
        _, cell = heapq.heappop(heap)
        n += 1
        r, c = cell
        for dr, dc in _D8:
            rr, cc = r + dr, c + dc
            nb = (rr, cc)
            if nb in parent:
                continue
            parent[nb] = cell
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if (dem[rr, cc] < z0 and nb not in visited) or rr in (0, h - 1) or cc in (0, w - 1):
                route = [nb]
                p = cell
                while p is not None and p != pit:
                    route.append(p)
                    p = parent[p]
                return route[::-1]
            heapq.heappush(heap, (float(dem[rr, cc]), nb))
    return []


@dataclass
class FlowPath:
    grid: Grid
    rows: np.ndarray
    cols: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    dist_m: np.ndarray  # cumulative along-path distance

    def lonlat_linestring(self, simplify_m: float = 30.0) -> LineString:
        line = LineString(np.column_stack([self.x, self.y])).simplify(simplify_m)
        return reproject_geom(line, self.grid.epsg, 4326)

    @property
    def length_km(self) -> float:
        return float(self.dist_m[-1] / 1000.0) if self.dist_m.size else 0.0


def flow_path_from_lonlat(dem: np.ndarray, grid: Grid, lon: float, lat: float, max_km: float = 150.0,
                          snap_radius_px: int = 3) -> FlowPath | None:
    r, c = grid.lonlat_rowcol(lon, lat)
    if not grid.contains_rc(r, c):
        return None
    # start at the lowest cell nearby (a lake outlet / the foot of the source area)
    r0, r1 = max(0, r - snap_radius_px), min(grid.height, r + snap_radius_px + 1)
    c0, c1 = max(0, c - snap_radius_px), min(grid.width, c + snap_radius_px + 1)
    win = dem[r0:r1, c0:c1]
    k = np.unravel_index(np.argmin(win), win.shape)
    cells = trace_downstream(dem, (r0 + k[0], c0 + k[1]), grid.res, max_km)
    rows = np.array([p[0] for p in cells])
    cols = np.array([p[1] for p in cells])
    x, y = grid.xy(rows, cols)
    x, y = np.asarray(x, float), np.asarray(y, float)
    seg = np.hypot(np.diff(x), np.diff(y))
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    return FlowPath(grid, rows, cols, x, y, dem[rows, cols].astype(float), dist)


@dataclass
class ExposedAsset:
    asset_id: int
    kind: str
    name: str
    name_ne: str
    path_distance_km: float
    offset_m: float
    height_above_channel_m: float
    travel_time_min: float


@dataclass
class ExposureParams:
    corridor_max_offset_m: float = 1500.0
    height_max_near_m: float = 60.0
    height_min_far_m: float = 15.0
    height_decay_m_per_km: float = 0.5
    max_distance_km: float = 150.0

    def height_limit(self, dist_km: np.ndarray) -> np.ndarray:
        return np.maximum(self.height_min_far_m, self.height_max_near_m - self.height_decay_m_per_km * dist_km)


def exposed_assets(path: FlowPath, assets: list[dict], dem: np.ndarray, params: ExposureParams,
                   wave_speed_m_s: float) -> list[ExposedAsset]:
    """Assets (dicts with id, kind, name, name_ne, lon, lat, at_channel) exposed along ``path``."""
    if path is None or path.x.size < 2 or not assets:
        return []
    g = path.grid
    tree = cKDTree(np.column_stack([path.x, path.y]))
    lons = np.array([a["lon"] for a in assets])
    lats = np.array([a["lat"] for a in assets])
    ax, ay = from_lonlat(lons, lats, g.epsg)
    ax, ay = np.asarray(ax), np.asarray(ay)
    d, idx = tree.query(np.column_stack([ax, ay]), distance_upper_bound=params.corridor_max_offset_m)
    out = []
    for i, a in enumerate(assets):
        if not np.isfinite(d[i]):
            continue
        j = int(idx[i])
        dist_km = float(path.dist_m[j] / 1000.0)
        if dist_km > params.max_distance_km:
            continue
        if a.get("at_channel"):
            hac = 0.0
        else:
            r, c = g.rowcol(ax[i], ay[i])
            if not g.contains_rc(int(r), int(c)):
                continue
            hac = float(dem[int(r), int(c)] - path.z[j])
        if hac > float(params.height_limit(np.array(dist_km))):
            continue
        out.append(ExposedAsset(
            asset_id=a["id"], kind=a["kind"], name=a.get("name", ""), name_ne=a.get("name_ne", ""),
            path_distance_km=round(dist_km, 2), offset_m=round(float(d[i]), 1),
            height_above_channel_m=round(max(hac, 0.0), 1),
            travel_time_min=round(dist_km * 1000.0 / wave_speed_m_s / 60.0, 1),
        ))
    out.sort(key=lambda e: e.path_distance_km)
    return out


def exposure_score(exposed: list[ExposedAsset], weights: dict[str, float], scale: float,
                   time_decay_min: float) -> tuple[float, dict]:
    """0..1 exposure: weighted assets, weighted more when the warning time is short."""
    total = 0.0
    counts: dict[str, int] = {}
    for e in exposed:
        w = weights.get(e.kind, 1.0)
        total += w * float(np.exp(-e.travel_time_min / time_decay_min))
        counts[e.kind] = counts.get(e.kind, 0) + 1
    score = float(1.0 - np.exp(-total / scale)) if scale > 0 else 0.0
    return score, {"weighted_total": round(total, 2), "counts": counts}


def load_curated_assets(path: Path) -> list[dict]:
    if not path.exists():
        return []
    fc = json.loads(path.read_text(encoding="utf-8"))
    return fc.get("features", [])


def lonlat_path(path: FlowPath) -> list[tuple[float, float]]:
    lon, lat = to_lonlat(path.x, path.y, path.grid.epsg)
    return list(zip(np.asarray(lon).tolist(), np.asarray(lat).tolist(), strict=True))
