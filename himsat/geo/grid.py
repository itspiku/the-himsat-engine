"""Raster grids in a projected (UTM) CRS and the transforms between them and lon/lat."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform


def utm_epsg(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180) / 6) + 1)
    return (32600 if lat >= 0 else 32700) + zone


@lru_cache(maxsize=64)
def _transformer(src: int, dst: int) -> Transformer:
    return Transformer.from_crs(CRS.from_epsg(src), CRS.from_epsg(dst), always_xy=True)


def reproject_geom(geom: BaseGeometry, src_epsg: int, dst_epsg: int) -> BaseGeometry:
    if src_epsg == dst_epsg:
        return geom
    t = _transformer(src_epsg, dst_epsg)
    return shp_transform(t.transform, geom)


def to_lonlat(x: np.ndarray | float, y: np.ndarray | float, epsg: int) -> tuple:
    return _transformer(epsg, 4326).transform(x, y)


def from_lonlat(lon: np.ndarray | float, lat: np.ndarray | float, epsg: int) -> tuple:
    return _transformer(4326, epsg).transform(lon, lat)


@dataclass(frozen=True)
class Grid:
    """A north-up raster grid: CRS (EPSG code), origin, pixel size, shape."""

    epsg: int
    x0: float  # west edge
    y0: float  # north edge
    res: float
    width: int
    height: int

    @classmethod
    def from_lonlat_bbox(cls, bbox: tuple[float, float, float, float], res: float, epsg: int | None = None) -> Grid:
        west, south, east, north = bbox
        epsg = epsg or utm_epsg((west + east) / 2, (south + north) / 2)
        xs, ys = from_lonlat(
            np.array([west, east, west, east]), np.array([south, south, north, north]), epsg
        )
        # snap to the resolution so grids built from overlapping boxes align pixel-for-pixel
        x0 = math.floor(min(xs) / res) * res
        x1 = math.ceil(max(xs) / res) * res
        y0 = math.ceil(max(ys) / res) * res
        y1 = math.floor(min(ys) / res) * res
        return cls(epsg, x0, y0, res, int(round((x1 - x0) / res)), int(round((y0 - y1) / res)))

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0.0, self.x0, 0.0, -self.res, self.y0)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0 - self.height * self.res, self.x0 + self.width * self.res, self.y0

    @property
    def pixel_area(self) -> float:
        return self.res * self.res

    def footprint(self) -> BaseGeometry:
        return box(*self.bounds)

    def lonlat_bbox(self) -> tuple[float, float, float, float]:
        g = reproject_geom(self.footprint(), self.epsg, 4326)
        return g.bounds

    def xy(self, row: np.ndarray | float, col: np.ndarray | float) -> tuple:
        """Pixel-centre map coordinates."""
        return self.x0 + (np.asarray(col) + 0.5) * self.res, self.y0 - (np.asarray(row) + 0.5) * self.res

    def rowcol(self, x: np.ndarray | float, y: np.ndarray | float) -> tuple:
        col = np.floor((np.asarray(x) - self.x0) / self.res).astype(int)
        row = np.floor((self.y0 - np.asarray(y)) / self.res).astype(int)
        return row, col

    def lonlat_rowcol(self, lon: float, lat: float) -> tuple[int, int]:
        x, y = from_lonlat(lon, lat, self.epsg)
        r, c = self.rowcol(x, y)
        return int(r), int(c)

    def contains_rc(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def window(self, row0: int, col0: int, height: int, width: int) -> Grid:
        """Sub-grid (clipped to this grid)."""
        row0, col0 = max(0, row0), max(0, col0)
        height = min(height, self.height - row0)
        width = min(width, self.width - col0)
        return Grid(self.epsg, self.x0 + col0 * self.res, self.y0 - row0 * self.res, self.res, width, height)

    def resampled(self, res: float) -> Grid:
        f = self.res / res
        return Grid(self.epsg, self.x0, self.y0, res, max(1, int(round(self.width * f))),
                    max(1, int(round(self.height * f))))

    def around(self, x: float, y: float, half_size_m: float) -> Grid:
        """Square grid centred on a map coordinate, snapped to this grid's pixel lattice."""
        n = int(math.ceil(half_size_m / self.res))
        r, c = self.rowcol(x, y)
        return Grid(self.epsg, self.x0 + (int(c) - n) * self.res, self.y0 - (int(r) - n) * self.res,
                    self.res, 2 * n, 2 * n)

    def key(self) -> str:
        return f"{self.epsg}_{self.x0:.0f}_{self.y0:.0f}_{self.res:g}_{self.width}x{self.height}"


@dataclass(frozen=True)
class Tile:
    """A processing tile: a core region plus an overlap margin (the ``grid``)."""

    index: tuple[int, int]
    grid: Grid  # with overlap
    core_row0: int  # core offset within the tile grid
    core_col0: int
    core_height: int
    core_width: int

    def in_core(self, row: float, col: float) -> bool:
        return (self.core_row0 <= row < self.core_row0 + self.core_height
                and self.core_col0 <= col < self.core_col0 + self.core_width)


def iter_tiles(grid: Grid, size: int, overlap: int) -> Iterator[Tile]:
    """Split a grid into overlapping tiles; each pixel belongs to exactly one tile core."""
    n_rows = max(1, math.ceil(grid.height / size))
    n_cols = max(1, math.ceil(grid.width / size))
    for i in range(n_rows):
        for j in range(n_cols):
            r0, c0 = i * size, j * size
            ch, cw = min(size, grid.height - r0), min(size, grid.width - c0)
            gr0, gc0 = max(0, r0 - overlap), max(0, c0 - overlap)
            gr1, gc1 = min(grid.height, r0 + ch + overlap), min(grid.width, c0 + cw + overlap)
            sub = grid.window(gr0, gc0, gr1 - gr0, gc1 - gc0)
            yield Tile((i, j), sub, r0 - gr0, c0 - gc0, ch, cw)
