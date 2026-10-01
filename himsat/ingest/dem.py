"""Copernicus GLO-30 DEM and derived terrain layers."""

from __future__ import annotations

import logging
import math

import numpy as np
from rasterio.enums import Resampling
from scipy import ndimage

from himsat.geo.grid import Grid
from himsat.ingest.loader import mosaic, read_to_grid
from himsat.ingest.stac import Catalog

log = logging.getLogger(__name__)


def load_dem(grid: Grid, catalog: Catalog | None = None) -> np.ndarray:
    catalog = catalog or Catalog()
    hrefs = catalog.search_dem(grid.lonlat_bbox())
    if not hrefs:
        raise RuntimeError("No DEM tiles found for grid")
    parts = [read_to_grid(h, grid, resampling=Resampling.bilinear, catalog=catalog) for h in hrefs]
    dem = mosaic(parts)
    if np.isnan(dem).any():
        # voids (rare in GLO-30) are filled by nearest valid neighbour
        idx = ndimage.distance_transform_edt(np.isnan(dem), return_distances=False, return_indices=True)
        dem = dem[tuple(idx)]
    return dem.astype("float32")


def slope_aspect(dem: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope (degrees) and aspect (degrees clockwise from north, downslope direction)."""
    dzdy, dzdx = np.gradient(dem.astype("float64"), res)
    # rows increase southward: north component = -dzdy
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) + 360.0) % 360.0
    return slope.astype("float32"), aspect.astype("float32")


def hillshade(dem: np.ndarray, res: float, sun_azimuth: float, sun_elevation: float) -> np.ndarray:
    """Cosine of the local solar incidence angle (<= 0 means self-shadowed)."""
    slope, aspect = slope_aspect(dem, res)
    zen = math.radians(90.0 - sun_elevation)
    s, a = np.radians(slope), np.radians(aspect)
    return (np.cos(zen) * np.cos(s) + np.sin(zen) * np.sin(s) * np.cos(math.radians(sun_azimuth) - a)).astype("float32")


def cast_shadow(dem: np.ndarray, res: float, sun_azimuth: float, sun_elevation: float,
                max_distance_m: float = 6000.0) -> np.ndarray:
    """Terrain cast-shadow mask by horizon marching towards the sun.

    A pixel is shadowed if any terrain along the sun direction rises above the line of sight.
    """
    az = math.radians(sun_azimuth)
    dx, dy = math.sin(az), -math.cos(az)  # columns eastward, rows southward (towards the sun)
    tan_el = math.tan(math.radians(max(sun_elevation, 1.0)))
    step = res
    n = int(max_distance_m / step)
    shadow = np.zeros(dem.shape, dtype=bool)
    h, w = dem.shape
    for k in range(1, n + 1):
        oc, orr = dx * k, dy * k
        c0, r0 = int(round(oc)), int(round(orr))
        if abs(c0) >= w or abs(r0) >= h:
            break
        shifted = np.full(dem.shape, -np.inf, dtype=np.float32)
        src = dem[max(0, r0):h + min(0, r0), max(0, c0):w + min(0, c0)]
        shifted[max(0, -r0):h - max(0, r0), max(0, -c0):w - max(0, c0)] = src
        shadow |= shifted - dem > k * step * tan_el
    return shadow


def upsample(arr: np.ndarray, shape: tuple[int, int], order: int = 1) -> np.ndarray:
    zy, zx = shape[0] / arr.shape[0], shape[1] / arr.shape[1]
    out = ndimage.zoom(arr, (zy, zx), order=order)
    # zoom may be off by one pixel
    out = out[: shape[0], : shape[1]]
    if out.shape != shape:
        out = np.pad(out, ((0, shape[0] - out.shape[0]), (0, shape[1] - out.shape[1])), mode="edge")
    return out
