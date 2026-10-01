"""Per-AOI processing context: grids, tiles, static terrain layers and product stores."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy import ndimage
from shapely.geometry import shape

from himsat.config import AOIConfig, Settings, get_settings
from himsat.detect.velocity import VelocityField
from himsat.geo.geometry import rasterize
from himsat.geo.grid import Grid, Tile, iter_tiles
from himsat.ingest.dem import load_dem, slope_aspect
from himsat.ingest.stac import Catalog
from himsat.sites.osm import fetch_glaciers

log = logging.getLogger(__name__)
EPOCH = datetime(2015, 1, 1, tzinfo=UTC)


@dataclass
class TileStatic:
    dem: np.ndarray
    slope: np.ndarray
    aspect: np.ndarray  # smoothed downslope direction (deg), for projecting velocities
    glacier: np.ndarray  # inventory glacier mask
    watch: np.ndarray  # where surface motion is tracked
    stable: np.ndarray  # terrain assumed motionless (co-registration reference)
    stable_sample: np.ndarray  # checkerboard subset of ``stable`` that is tracked alongside the watch zone


class AOIContext:
    def __init__(self, cfg: AOIConfig, settings: Settings | None = None, catalog: Catalog | None = None,
                 products_dir: Path | None = None):
        self.cfg = cfg
        self.settings = settings or get_settings()
        self.catalog = catalog or Catalog()
        self.grid = Grid.from_lonlat_bbox(cfg.bbox, cfg.resolution_m)
        self.tiles: list[Tile] = list(iter_tiles(self.grid, cfg.tile_size_px, cfg.tile_overlap_px))
        self.routing_grid = Grid.from_lonlat_bbox(cfg.routing_bbox, 30.0, epsg=self.grid.epsg)
        self.products = (products_dir or self.settings.products_dir) / cfg.id
        self._static: dict[tuple[int, int], TileStatic] = {}

    # -- inventories -----------------------------------------------------------------------
    @cached_property
    def glacier_features(self) -> list[dict]:
        try:
            return fetch_glaciers(self.cfg.bbox, self.cfg.id)["features"]
        except Exception as e:  # offline and no cache
            log.error("glacier inventory unavailable: %s", e)
            return []

    @cached_property
    def glacier_geoms(self) -> list:
        return [shape(f["geometry"]) for f in self.glacier_features]

    @cached_property
    def routing_dem(self) -> np.ndarray:
        log.info("loading routing DEM %s", self.routing_grid)
        return load_dem(self.routing_grid, self.catalog)

    # -- static layers ---------------------------------------------------------------------
    def tile_static(self, tile: Tile) -> TileStatic:
        if tile.index in self._static:
            return self._static[tile.index]
        g = tile.grid
        dem = load_dem(g, self.catalog)
        slope, aspect = slope_aspect(ndimage.gaussian_filter(dem, 2), g.res)
        s = ndimage.gaussian_filter(np.sin(np.radians(aspect)), 8)
        c = ndimage.gaussian_filter(np.cos(np.radians(aspect)), 8)
        aspect_s = (np.degrees(np.arctan2(s, c)) + 360) % 360
        tb = g.lonlat_bbox()
        geoms = [gm for gm in self.glacier_geoms
                 if not (gm.bounds[2] < tb[0] or gm.bounds[0] > tb[2] or gm.bounds[3] < tb[1] or gm.bounds[1] > tb[3])]
        glacier = rasterize(geoms, g) if geoms else np.zeros(g.shape, bool)
        high = dem >= self.cfg.watch_min_elevation_m
        watch = high & (glacier | (slope >= self.cfg.watch_min_slope_deg))
        near_ice = (ndimage.distance_transform_edt(~glacier) * g.res <= 300.0) if glacier.any() else glacier
        stable = ~near_ice & ~watch & (slope > 3) & (slope < 30)
        rr, cc = np.indices(g.shape, sparse=True)
        stable_sample = stable & (((rr // 48) + (cc // 48)) % 2 == 0)
        st = TileStatic(dem, slope, aspect_s.astype("float32"), glacier, watch, stable, stable_sample)
        if len(self._static) > 6:
            self._static.pop(next(iter(self._static)))
        self._static[tile.index] = st
        return st

    # -- S2 per-pixel "latest clear" composite (for optical change detection) ----------------
    def composite_path(self, tile: Tile) -> Path:
        return self.products / "s2composite" / f"{tile.index[0]}_{tile.index[1]}.npz"

    def load_composite(self, tile: Tile, bands: tuple[str, ...] = ()) -> dict[str, np.ndarray] | None:
        p = self.composite_path(tile)
        if not p.exists():
            return None
        with np.load(p) as z:
            comp = {k: z[k] for k in z.files}
        if any(b not in comp for b in bands) or comp.get("day", np.zeros(0)).shape != tile.grid.shape:
            log.warning("composite %s incompatible (bands/shape changed); starting a new one", p.name)
            return None
        return comp

    def save_composite(self, tile: Tile, comp: dict[str, np.ndarray]) -> None:
        p = self.composite_path(tile)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, **comp)  # uncompressed: rewritten after every clear scene, so speed matters more than size
        tmp.replace(p)

    # -- velocity fields ---------------------------------------------------------------------
    def velocity_dir(self, tile: Tile) -> Path:
        return self.products / "velocity" / f"{tile.index[0]}_{tile.index[1]}"

    def save_velocity(self, tile: Tile, vf: VelocityField, ref_at: datetime, sec_at: datetime, orbit: int | None,
                      meta: dict) -> Path:
        d = self.velocity_dir(tile)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{sec_at:%Y%m%dT%H%M}_R{orbit or 0:03d}.npz"
        np.savez_compressed(p, dx=vf.dx.astype("float16"), dy=vf.dy.astype("float16"),
                            corr=vf.corr.astype("float16"), rows=vf.rows, cols=vf.cols, dt=vf.dt_days,
                            ref=ref_at.timestamp(), sec=sec_at.timestamp(), meta=json.dumps(meta))
        return p

    def velocity_history(self, tile: Tile, before: datetime, lo_days: float, hi_days: float) -> list[dict]:
        d = self.velocity_dir(tile)
        if not d.exists():
            return []
        out = []
        for p in sorted(d.glob("*.npz")):
            with np.load(p) as z:
                sec = datetime.fromtimestamp(float(z["sec"]), UTC)
                age = (before - sec).total_seconds() / 86400
                if lo_days <= age <= hi_days:
                    out.append({"sec": sec, "dt": float(z["dt"]), "dx": z["dx"].astype("float32"),
                                "dy": z["dy"].astype("float32"), "meta": json.loads(str(z["meta"]))})
        return out


def days_since_epoch(dt: datetime) -> int:
    return int((dt - EPOCH).total_seconds() // 86400)
