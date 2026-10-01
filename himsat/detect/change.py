"""Change detection between acquisitions.

* SAR log-ratio: fresh mass-movement scars and deposits (rock/ice avalanches, landslides, debris
  flows) change radar backscatter strongly, and radar sees through monsoon cloud. Seasonal wet
  snow also darkens large areas; changes coherent across an elevation band are rejected as seasonal.
* Optical fracture index: crevasses and cracks show up as thin dark linear features in NIR.
  Newly appearing ridge energy on an ice/rock slope is summarised per site.
* Optical vegetation loss: NDVI drop with brightening (fresh scar, clearing, crop damage).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage import filters, measure

from himsat.geo.grid import Grid, to_lonlat


@dataclass
class ChangeRegion:
    kind: str
    lon: float
    lat: float
    area_m2: float
    mask_bbox: tuple[int, int, int, int]  # r0, c0, r1, c1 in grid pixels
    mean_change: float
    confidence: float
    attrs: dict = field(default_factory=dict)
    mask: np.ndarray | None = None  # local mask within mask_bbox


def lee_filter(img_db: np.ndarray, size: int = 5) -> np.ndarray:
    """Lee speckle filter in the linear domain; returns dB."""
    lin = np.power(10.0, img_db / 10.0)
    nan = ~np.isfinite(lin)
    lin = np.where(nan, np.nanmean(lin) if (~nan).any() else 0.0, lin)
    mean = ndimage.uniform_filter(lin, size)
    sq = ndimage.uniform_filter(lin * lin, size)
    var = np.maximum(sq - mean * mean, 0)
    noise = np.mean(var) if var.size else 0.0
    w = var / np.maximum(var + noise, 1e-12)
    out = mean + w * (lin - mean)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.maximum(out, 1e-6))
    db[nan] = np.nan
    return db.astype("float32")


def sar_log_ratio(pre_db: np.ndarray, post_db: np.ndarray, size: int = 5) -> np.ndarray:
    return (lee_filter(post_db, size) - lee_filter(pre_db, size)).astype("float32")


def seasonal_band_mask(lr: np.ndarray, dem: np.ndarray, thr_db: float, band_m: float = 200.0,
                       max_fraction: float = 0.35) -> np.ndarray:
    """Pixels whose change is shared by a large fraction of their elevation band (wet snow, melt)."""
    out = np.zeros(lr.shape, bool)
    finite = np.isfinite(lr) & np.isfinite(dem)
    if not finite.any():
        return out
    lo, hi = np.nanmin(dem[finite]), np.nanmax(dem[finite])
    for z in np.arange(lo, hi + band_m, band_m):
        band = finite & (dem >= z) & (dem < z + band_m)
        n = band.sum()
        if n < 500:
            continue
        for sign in (1, -1):
            ch = band & (sign * lr > thr_db)
            if ch.sum() / n > max_fraction:
                out |= ch
    return out


def detect_sar_changes(pre_db: np.ndarray, post_db: np.ndarray, grid: Grid, dem: np.ndarray, slope: np.ndarray,
                       *, thr_db: float = 3.0, min_area_m2: float = 50000.0,
                       exclude: np.ndarray | None = None, ice: np.ndarray | None = None,
                       max_disturbed_fraction: float = 0.04) -> list[ChangeRegion]:
    """Compact clusters of strong backscatter change → candidate mass movements.

    Melt-season glacier surfaces change backscatter all the time (wet snow, ponds). A change
    lying almost entirely on ice with little vertical extent is surface melt, not a mass
    movement. Rock/ice avalanches run off the ice and down hundreds of metres.
    """
    lr = sar_log_ratio(pre_db, post_db)
    seasonal = seasonal_band_mask(lr, dem, thr_db)
    strong = np.isfinite(lr) & (np.abs(lr) > thr_db) & ~seasonal
    if exclude is not None:
        strong &= ~exclude
    strong = ndimage.binary_opening(strong, iterations=1)
    strong = ndimage.binary_closing(strong, iterations=2)
    # mass movements are rare and local; when a large share of the scene changes at once (snowfall,
    # melt-freeze, soil moisture) individual blobs are not trustworthy
    finite = np.isfinite(lr)
    lab = measure.label(strong, connectivity=2)
    # a real cascade is a few large connected bodies; snow/moisture change is fragmented. Judge the
    # scene by the change remaining *outside* its three largest components.
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    top = np.argsort(sizes)[-3:]
    background = strong & ~np.isin(lab, top[sizes[top] > 0])
    strong_fraction = float(background[finite].mean()) if finite.any() else 0.0
    disturbed = strong_fraction > max_disturbed_fraction
    min_px = int(min_area_m2 / grid.pixel_area)
    out = []
    for rp in measure.regionprops(lab):
        if rp.area < min_px:
            continue
        rr, cc = rp.coords[:, 0], rp.coords[:, 1]
        vals = lr[rr, cc]
        cy, cx = rp.centroid
        x, y = grid.xy(cy, cx)
        lon, lat = to_lonlat(float(x), float(y), grid.epsg)
        z = dem[rr, cc]
        relief = float(np.percentile(z, 95) - np.percentile(z, 5))
        mean_slope = float(np.mean(slope[rr, cc]))
        on_ice = float(ice[rr, cc].mean()) if ice is not None else 0.0
        if on_ice > 0.8 and relief < 400.0:
            continue
        # a mass movement spans relief (source → deposit) or sits on steep terrain; flat, low-relief
        # blobs are more often agricultural/moisture change
        size = min(1.0, rp.area * grid.pixel_area / 500_000.0)
        conf = float(np.clip(0.2 + 0.35 * min(relief / 500.0, 1.0) + 0.25 * min(mean_slope / 30.0, 1.0)
                             + 0.2 * size, 0, 1))
        if disturbed:
            conf *= 0.5
        frac_inc = float((vals > 0).mean())
        # wet snow darkens radar uniformly; rock-ice avalanche deposits brighten (rough debris) and
        # source scars, which can darken, span far more relief than a snow patch
        snow_like = frac_inc <= 0.1 and float(np.mean(vals)) <= -3.0 and relief < 800.0
        r0, c0, r1, c1 = rp.bbox
        out.append(ChangeRegion(
            kind="mass_movement", lon=float(lon), lat=float(lat), area_m2=float(rp.area * grid.pixel_area),
            mask_bbox=(r0, c0, r1, c1), mean_change=float(np.mean(vals)), confidence=conf,
            attrs={"relief_m": relief, "mean_slope_deg": mean_slope, "elev_min_m": float(z.min()),
                   "elev_max_m": float(z.max()), "frac_increase": frac_inc, "on_ice": on_ice, "snow_like": snow_like,
                   "scene_change_fraction": round(strong_fraction, 4), "disturbed": disturbed, "sensor": "S1"},
            mask=rp.image.copy(),
        ))
    return out


def fracture_index(nir: np.ndarray, sigmas: tuple[float, ...] = (1.0, 2.0)) -> np.ndarray:
    """Dark thin linear features (crevasses / cracks): Sato ridge filter on NIR."""
    a = np.nan_to_num(nir, nan=float(np.nanmedian(nir)) if np.isfinite(nir).any() else 0.0)
    return filters.sato(a, sigmas=sigmas, black_ridges=True).astype("float32")


def new_fractures(pre_nir: np.ndarray, post_nir: np.ndarray, region: np.ndarray, res: float,
                  valid: np.ndarray, thr: float | None = None,
                  fi: tuple[np.ndarray, np.ndarray] | None = None) -> dict | None:
    """Newly appeared linear dark features within ``region``.

    Returns the length of new fracture skeleton (m), the fraction of region affected and the
    increase in ridge energy. ``valid`` = clear in both images.
    """
    reg = region & valid
    if reg.sum() < 200:
        return None
    # ``fi``: ridge indices precomputed once per scene (they are the expensive part)
    fi0, fi1 = fi if fi is not None else (fracture_index(pre_nir), fracture_index(post_nir))
    if thr is None:
        vals = fi1[reg]
        thr = float(np.percentile(vals, 97)) if vals.size else 0.0
        thr = max(thr, 0.02)
    new = reg & (fi1 > thr) & (fi0 < 0.5 * thr)
    new = ndimage.binary_opening(new, structure=np.ones((1, 2))) | ndimage.binary_opening(new, structure=np.ones((2, 1)))
    lab = measure.label(new, connectivity=2)
    length_px = 0
    min_len_px = max(3, int(60 / res))
    for rp in measure.regionprops(lab):
        if rp.axis_major_length >= min_len_px and rp.eccentricity > 0.9:
            length_px += rp.axis_major_length
    return {
        "new_fracture_length_m": float(length_px * res),
        "ridge_energy_change": float(np.mean(fi1[reg]) - np.mean(fi0[reg])),
        "observed_fraction": float(reg.sum() / max(region.sum(), 1)),
    }


def detect_vegetation_loss(pre: dict[str, np.ndarray], post: dict[str, np.ndarray], valid: np.ndarray,
                           grid: Grid, slope: np.ndarray, landcover: np.ndarray | None = None,
                           *, ndvi_drop: float = 0.25, min_area_m2: float = 10000.0) -> list[ChangeRegion]:
    """NDVI drop clusters, labelled by land-cover context (ESA WorldCover codes when available)."""
    from himsat.detect.indices import brightness, ndvi

    n0, n1 = ndvi(pre), ndvi(post)
    was_veg = n0 > 0.45
    drop = valid & was_veg & ((n0 - n1) > ndvi_drop) & (brightness(post) > brightness(pre))
    drop = ndimage.binary_opening(drop, iterations=1)
    lab = measure.label(drop, connectivity=2)
    min_px = int(min_area_m2 / grid.pixel_area)
    out = []
    for rp in measure.regionprops(lab):
        if rp.area < min_px:
            continue
        rr, cc = rp.coords[:, 0], rp.coords[:, 1]
        cy, cx = rp.centroid
        x, y = grid.xy(cy, cx)
        lon, lat = to_lonlat(float(x), float(y), grid.epsg)
        ms = float(np.mean(slope[rr, cc]))
        kind = "mass_movement" if ms > 25 and rp.eccentricity > 0.9 else "vegetation_loss"
        if landcover is not None:
            lc = np.bincount(landcover[rr, cc].astype(int), minlength=101)
            dominant = int(lc.argmax())
            if dominant == 40:  # WorldCover cropland
                kind = "cropland_change"
            elif dominant == 10 and kind != "mass_movement":  # tree cover
                kind = "vegetation_loss"
        r0, c0, r1, c1 = rp.bbox
        out.append(ChangeRegion(
            kind=kind, lon=float(lon), lat=float(lat), area_m2=float(rp.area * grid.pixel_area),
            mask_bbox=(r0, c0, r1, c1), mean_change=float(np.mean((n1 - n0)[rr, cc])),
            confidence=float(np.clip(0.4 + (np.mean((n0 - n1)[rr, cc]) - ndvi_drop), 0.2, 0.9)),
            attrs={"mean_slope_deg": ms, "sensor": "S2"}, mask=rp.image.copy(),
        ))
    return out
