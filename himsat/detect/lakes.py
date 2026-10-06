"""Lake objects from segmentation masks (optical) and SAR backscatter (cloud-independent)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from shapely.geometry.base import BaseGeometry
from skimage import filters, measure

from himsat.detect.segmentation import DEBRIS_ICE, INVALID, SNOW_ICE, WATER, Segmentation
from himsat.geo.geometry import mask_to_polygons
from himsat.geo.grid import Grid, Tile, reproject_geom, to_lonlat


@dataclass
class LakeDetection:
    geom_utm: BaseGeometry
    geom_lonlat: BaseGeometry
    area_m2: float
    lon: float
    lat: float
    elevation_m: float
    quality: float  # fraction of the lake's surroundings that was observable (not cloud / no data)
    mean_prob: float
    elongation: float
    glacier_distance_m: float
    max_surrounding_slope_deg: float
    turbidity: float | None = None
    attrs: dict = field(default_factory=dict)


def extract_lakes(seg: Segmentation, grid: Grid, dem: np.ndarray, slope: np.ndarray, *,
                  min_area_m2: float, min_elevation_m: float, tile: Tile | None = None,
                  turbidity: np.ndarray | None = None, ice_mask: np.ndarray | None = None,
                  max_elongation: float = 7.0, glacier_search_m: float = 2000.0) -> list[LakeDetection]:
    """Connected water bodies → lake objects with the attributes the risk model needs.

    Only objects whose centroid falls in the tile core are returned (tiles overlap). Rivers are
    removed by shape: elongated ribbons are not lakes.
    """
    water = seg.classes == WATER
    invalid = seg.classes == INVALID
    ice = ice_mask if ice_mask is not None else np.isin(seg.classes, (SNOW_ICE, DEBRIS_ICE))
    labels = measure.label(water, connectivity=2)
    if labels.max() == 0:
        return []
    ice_dist = ndimage.distance_transform_edt(~ice) * grid.res if ice.any() else None
    p_water = seg.prob.get(WATER)
    min_px = max(1, int(min_area_m2 / grid.pixel_area))
    ring_r = max(2, int(round(60.0 / grid.res)))
    wall_r = max(2, int(round(500.0 / grid.res)))
    out: list[LakeDetection] = []
    for rp in measure.regionprops(labels):
        if rp.area < min_px:
            continue
        cy, cx = rp.centroid
        if tile is not None and not tile.in_core(cy, cx):
            continue
        r0, c0, r1, c1 = rp.bbox
        zs = dem[rp.coords[:, 0], rp.coords[:, 1]]
        elev = float(np.median(zs))
        # Copernicus DEM flattens lake surfaces; rivers keep their downstream gradient
        elev_range = float(np.percentile(zs, 90) - np.percentile(zs, 10))
        elong = float(rp.axis_major_length / max(rp.axis_minor_length, 1.0))
        if elong > max_elongation and rp.area < 50 * min_px:
            continue  # river reach
        # windows around the object
        pad = wall_r + 2
        wr0, wc0 = max(0, r0 - pad), max(0, c0 - pad)
        wr1, wc1 = min(water.shape[0], r1 + pad), min(water.shape[1], c1 + pad)
        obj = labels[wr0:wr1, wc0:wc1] == rp.label
        dist = ndimage.distance_transform_edt(~obj)
        ring = (dist > 0) & (dist <= ring_r)
        quality = 1.0 - float(invalid[wr0:wr1, wc0:wc1][ring | obj].mean()) if ring.any() else 1.0
        near = (dist > 0) & (dist <= wall_r)
        wall = float(np.nanmax(slope[wr0:wr1, wc0:wc1][near])) if near.any() else 0.0
        gdist = float(ice_dist[rp.coords[:, 0], rp.coords[:, 1]].min()) if ice_dist is not None else np.inf
        if elev < min_elevation_m and gdist > glacier_search_m:
            continue  # not a glacial lake
        polys = mask_to_polygons(obj, grid.window(wr0, wc0, wr1 - wr0, wc1 - wc0))
        if not polys:
            continue
        geom = max(polys, key=lambda p: p.area) if len(polys) > 1 else polys[0]
        geom = geom.simplify(grid.res * 0.5, preserve_topology=True)
        x, y = grid.xy(cy, cx)
        lon, lat = to_lonlat(float(x), float(y), grid.epsg)
        turb = None
        if turbidity is not None:
            core = ndimage.binary_erosion(obj, iterations=1)
            vals = turbidity[wr0:wr1, wc0:wc1][core if core.any() else obj]
            vals = vals[np.isfinite(vals)]
            turb = float(np.median(vals)) if vals.size else None
        mp = float(np.nanmean(p_water[rp.coords[:, 0], rp.coords[:, 1]])) if p_water is not None else 1.0
        out.append(LakeDetection(
            geom_utm=geom, geom_lonlat=reproject_geom(geom, grid.epsg, 4326), area_m2=float(rp.area * grid.pixel_area),
            lon=float(lon), lat=float(lat), elevation_m=elev, quality=quality, mean_prob=mp, elongation=elong,
            glacier_distance_m=gdist if np.isfinite(gdist) else 1e6, max_surrounding_slope_deg=wall, turbidity=turb,
            attrs={"elev_range_m": elev_range},
        ))
    return out


def sar_water_area(vv_db: np.ndarray, grid: Grid, lake_mask: np.ndarray, slope: np.ndarray,
                   buffer_m: float = 200.0) -> tuple[float, float, np.ndarray] | None:
    """Lake area from Sentinel-1 VV backscatter around a known lake (for cloudy periods).

    Water is a specular reflector (dark). The threshold is Otsu's, computed locally in a buffer
    around the reference outline and bounded to a physically plausible range. Water pixels connected
    to the reference lake count towards its area. Returns (area_m2, quality, mask) or None.
    """
    r = max(2, int(round(buffer_m / grid.res)))
    zone = ndimage.binary_dilation(lake_mask, iterations=r)
    vals = vv_db[zone]
    finite = np.isfinite(vals)
    if finite.mean() < 0.7 or finite.sum() < 50:
        return None
    try:
        thr = float(filters.threshold_otsu(vals[finite]))
    except ValueError:
        return None
    thr = float(np.clip(thr, -24.0, -14.0))
    wet = zone & np.isfinite(vv_db) & (vv_db < thr) & (slope < 15)
    wet = ndimage.binary_opening(wet, iterations=1)
    lab = measure.label(wet, connectivity=2)
    keep = np.unique(lab[lake_mask & wet])
    keep = keep[keep > 0]
    if keep.size == 0:
        return 0.0, 0.3, np.zeros_like(lake_mask)
    mask = np.isin(lab, keep)
    # separability of the two modes: a low-contrast histogram (wind, frozen/snow-covered lake) = low quality
    inside = vv_db[mask]
    outside = vv_db[zone & ~mask & np.isfinite(vv_db)]
    contrast = float(np.nanmedian(outside) - np.nanmedian(inside)) if outside.size and inside.size else 0.0
    quality = float(np.clip(contrast / 8.0, 0.0, 1.0)) * float(finite.mean())
    # wind-roughened or frozen lakes stop looking dark: if most of the known outline is no longer
    # classified as water the measurement is unreliable, not evidence of change
    overlap = float((mask & lake_mask).sum() / max(lake_mask.sum(), 1))
    quality *= float(np.clip(overlap / 0.7, 0.0, 1.0))
    return float(mask.sum() * grid.pixel_area), quality, mask
