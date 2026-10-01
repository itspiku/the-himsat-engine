"""Geometry helpers: GeoJSON (de)serialisation and raster → vector conversion."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from rasterio import features
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from himsat.geo.grid import Grid, reproject_geom


def to_geojson(geom: BaseGeometry | None, precision: int = 6) -> str | None:
    if geom is None or geom.is_empty:
        return None

    def _round(obj: Any) -> Any:
        if isinstance(obj, (list, tuple)):
            return [_round(o) for o in obj]
        if isinstance(obj, float):
            return round(obj, precision)
        return obj

    m = mapping(geom)
    return json.dumps({"type": m["type"], "coordinates": _round(m["coordinates"])}, separators=(",", ":"))


def from_geojson(text: str | dict | None) -> BaseGeometry | None:
    if not text:
        return None
    return shape(json.loads(text) if isinstance(text, str) else text)


def set_geom(obj: Any, geom_lonlat: BaseGeometry | None) -> None:
    """Assign a lon/lat geometry to a model using ``GeomMixin``."""
    obj.geom = to_geojson(geom_lonlat)
    if geom_lonlat is None or geom_lonlat.is_empty:
        obj.min_lon = obj.min_lat = obj.max_lon = obj.max_lat = None
    else:
        obj.min_lon, obj.min_lat, obj.max_lon, obj.max_lat = geom_lonlat.bounds


def mask_to_polygons(mask: np.ndarray, grid: Grid, min_pixels: int = 1) -> list[BaseGeometry]:
    """Vectorise a boolean mask into polygons in the grid CRS."""
    if not mask.any():
        return []
    polys = [
        shape(g)
        for g, v in features.shapes(mask.astype(np.uint8), mask=mask, transform=grid.transform, connectivity=8)
        if v == 1
    ]
    return [p for p in polys if p.area >= min_pixels * grid.pixel_area]


def mask_to_lonlat_geom(mask: np.ndarray, grid: Grid, simplify_m: float | None = None) -> BaseGeometry | None:
    polys = mask_to_polygons(mask, grid)
    if not polys:
        return None
    g = unary_union(polys)
    if simplify_m:
        g = g.simplify(simplify_m, preserve_topology=True)
    return reproject_geom(g, grid.epsg, 4326)


def rasterize(geoms: list[BaseGeometry], grid: Grid, geoms_epsg: int = 4326, all_touched: bool = False) -> np.ndarray:
    """Burn geometries (default lon/lat) into a boolean mask on ``grid``."""
    shapes = [reproject_geom(g, geoms_epsg, grid.epsg) for g in geoms if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(grid.shape, dtype=bool)
    return features.rasterize(
        [(s, 1) for s in shapes], out_shape=grid.shape, transform=grid.transform, fill=0,
        all_touched=all_touched, dtype=np.uint8,
    ).astype(bool)
