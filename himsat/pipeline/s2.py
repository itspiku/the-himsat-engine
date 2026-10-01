"""Sentinel-2 acquisition processing: segmentation → lakes → sites/observations → optical change."""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
from scipy import ndimage
from sqlalchemy import select
from sqlalchemy.orm import Session

from himsat.db.models import ChangeEvent, Observation, Site
from himsat.detect import indices as ix
from himsat.detect.change import detect_vegetation_loss, new_fractures
from himsat.detect.lakes import LakeDetection, extract_lakes
from himsat.detect.segmentation import INVALID, SNOW_ICE, Segmenter, TerrainContext
from himsat.geo.geometry import rasterize, set_geom
from himsat.geo.grid import Tile, to_lonlat
from himsat.ingest.dem import cast_shadow, hillshade, upsample
from himsat.ingest.loader import S2_DEFAULT_BANDS, load_s2
from himsat.ingest.stac import Acquisition
from himsat.pipeline.context import AOIContext, days_since_epoch
from himsat.sites import inventory as inv

log = logging.getLogger(__name__)
COMPOSITE_BANDS = ("B02", "B03", "B04", "B08", "B11")
MAX_CHANGE_GAP_DAYS = 75
MAX_LAKE_ELEV_RANGE_M = 15.0


def terrain_for(ctx: AOIContext, tile: Tile, acq: Acquisition) -> TerrainContext:
    st = ctx.tile_static(tile)
    shadow = None
    if acq.sun_azimuth is not None and acq.sun_elevation is not None:
        f = 3  # shadows at 30 m (DEM native resolution), then upsampled
        dem30 = st.dem[::f, ::f]
        res30 = tile.grid.res * f
        sh = cast_shadow(dem30, res30, acq.sun_azimuth, acq.sun_elevation) | (
            hillshade(dem30, res30, acq.sun_azimuth, acq.sun_elevation) < 0.05)
        shadow = upsample(sh.astype("float32"), tile.grid.shape, order=0) > 0.5
    return TerrainContext(dem=st.dem, slope=st.slope, shadow=shadow, glacier_prior=st.glacier)


def process_s2(session: Session, ctx: AOIContext, acq: Acquisition, segmenter: Segmenter,
               named_places: list, touched: set[int]) -> dict:
    bands = S2_DEFAULT_BANDS if "prithvi" in segmenter.name else ("B02", "B03", "B04", "B08", "B11")
    stats = {"tiles": 0, "lakes": 0, "new_sites": 0, "events": 0}
    from concurrent.futures import ThreadPoolExecutor

    def fetch(t):
        return load_s2(acq, t.grid, bands=bands, catalog=ctx.catalog, min_valid_fraction=0.05)

    # download the next tile while the current one is being processed (I/O and compute overlap)
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        nxt = prefetch.submit(fetch, ctx.tiles[0]) if ctx.tiles else None
        for i, tile in enumerate(ctx.tiles):
            scene = nxt.result()
            nxt = prefetch.submit(fetch, ctx.tiles[i + 1]) if i + 1 < len(ctx.tiles) else None
            if scene is None:
                continue
            _process_s2_tile(session, ctx, acq, segmenter, named_places, touched, tile, scene, stats)
    return stats


def _process_s2_tile(session: Session, ctx: AOIContext, acq: Acquisition, segmenter: Segmenter,
                     named_places: list, touched: set[int], tile: Tile, scene, stats: dict) -> None:
    stats["tiles"] += 1
    st = ctx.tile_static(tile)
    terrain = terrain_for(ctx, tile, acq)
    seg = segmenter.segment(scene, terrain)
    valid = seg.classes != INVALID
    turb = ix.turbidity_index(scene.bands)
    ice = st.glacier | (seg.classes == SNOW_ICE)
    lakes = extract_lakes(seg, tile.grid, st.dem, st.slope, min_area_m2=ctx.cfg.min_lake_area_m2,
                          min_elevation_m=0.0, tile=tile, turbidity=turb, ice_mask=ice)
    inv_dist = (ndimage.distance_transform_edt(~st.glacier) * tile.grid.res) if st.glacier.any() else None
    for lk in lakes:
        # below the glacial-lake elevation only lakes close to *inventoried* glaciers count
        # (seasonal snow would otherwise make every valley pond "glacial")
        near_inv = False
        if inv_dist is not None:
            r, c = tile.grid.lonlat_rowcol(lk.lon, lk.lat)
            near_inv = tile.grid.contains_rc(r, c) and inv_dist[r, c] < 1000
        glacial = lk.elevation_m >= ctx.cfg.min_glacial_lake_elevation_m or near_inv
        site = _record_lake(session, ctx, lk, acq, seg.method, glacial, named_places)
        if site is not None:
            touched.add(site.id)
            stats["lakes"] += 1
            if site.first_seen == acq.datetime:
                stats["new_sites"] += 1
    stats["events"] += _optical_change(session, ctx, tile, acq, scene.bands, valid, st, touched)
    _update_composite(ctx, tile, acq.datetime, scene.bands, valid)


def _record_lake(session: Session, ctx: AOIContext, lk: LakeDetection, acq: Acquisition, method: str,
                 glacial: bool, named_places: list) -> Site | None:
    site = inv.match_site(session, ctx.cfg.id, lk.geom_lonlat, ("glacial_lake", "barrier_lake"))
    if site is None:
        # only create sites from confident, fully observed detections
        if lk.quality < 0.9 or lk.mean_prob < 0.5:
            return None
        barrier = _near_recent_mass_movement(session, ctx, lk, acq.datetime)
        if not glacial and not barrier:
            return None
        if not barrier and lk.attrs.get("elev_range_m", 0) > MAX_LAKE_ELEV_RANGE_M:
            return None  # sloping water surface: a river reach, not a lake (barrier lakes postdate the DEM)
        kind = "barrier_lake" if barrier and not glacial else "glacial_lake"
        attrs = {"area_m2": lk.area_m2, "glacier_distance_m": lk.glacier_distance_m,
                 "max_surrounding_slope_deg": lk.max_surrounding_slope_deg, "barrier": bool(barrier)}
        site = inv.create_site(session, ctx.cfg, kind=kind, geom_lonlat=lk.geom_lonlat, source=f"discovered:{method}",
                               attrs=attrs, elevation_m=lk.elevation_m, first_seen=acq.datetime,
                               named_places=named_places)
    obs = session.scalars(select(Observation).where(Observation.site_id == site.id, Observation.kind == "lake_area",
                                                    Observation.scene_key == acq.key)).first()
    if obs is None:
        obs = Observation(site_id=site.id, kind="lake_area", sensor="S2", scene_key=acq.key,
                          observed_at=acq.datetime, method=method)
        session.add(obs)
    obs.values = {"area_m2": round(lk.area_m2, 1), "turbidity": lk.turbidity, "glacier_distance_m": lk.glacier_distance_m,
                  "max_surrounding_slope_deg": lk.max_surrounding_slope_deg, "mean_prob": round(lk.mean_prob, 3),
                  "elongation": round(lk.elongation, 2)}
    obs.quality = round(lk.quality, 3)
    set_geom(obs, lk.geom_lonlat)
    site.last_seen = max(site.last_seen or acq.datetime, acq.datetime)
    if lk.quality >= 0.95 and lk.mean_prob >= 0.6:
        # keep the site outline current with the latest fully observed detection
        set_geom(site, lk.geom_lonlat)
        site.lon, site.lat = lk.lon, lk.lat
        attrs = dict(site.attrs or {})
        attrs.update(area_m2=lk.area_m2, glacier_distance_m=lk.glacier_distance_m,
                     max_surrounding_slope_deg=lk.max_surrounding_slope_deg)
        site.attrs = attrs
    return site


def _near_recent_mass_movement(session: Session, ctx: AOIContext, lk: LakeDetection, when: datetime,
                               days: int = 45, radius_deg: float = 0.03) -> bool:
    from datetime import timedelta

    q = select(ChangeEvent).where(ChangeEvent.aoi_id == ctx.cfg.id, ChangeEvent.kind == "mass_movement",
                                  ChangeEvent.detected_at >= when - timedelta(days=days),
                                  ChangeEvent.detected_at <= when, ChangeEvent.confidence >= 0.6,
                                  ChangeEvent.lon.between(lk.lon - radius_deg, lk.lon + radius_deg),
                                  ChangeEvent.lat.between(lk.lat - radius_deg, lk.lat + radius_deg))
    return session.scalars(q).first() is not None


def _update_composite(ctx: AOIContext, tile: Tile, when: datetime, bands: dict, valid: np.ndarray) -> None:
    comp = ctx.load_composite(tile, COMPOSITE_BANDS)
    day = days_since_epoch(when)
    if comp is None:
        comp = {b: np.full(tile.grid.shape, np.nan, "float16") for b in COMPOSITE_BANDS}
        comp["day"] = np.full(tile.grid.shape, -1, "int32")
    for b in COMPOSITE_BANDS:
        arr = comp[b]
        arr[valid] = bands[b][valid].astype("float16")
    comp["day"][valid] = day
    ctx.save_composite(tile, comp)


def _optical_change(session: Session, ctx: AOIContext, tile: Tile, acq: Acquisition, bands: dict,
                    valid: np.ndarray, st, touched: set[int]) -> int:
    """Compare the new scene with each pixel's latest clear observation."""
    comp = ctx.load_composite(tile, COMPOSITE_BANDS)
    if comp is None:
        return 0
    day = days_since_epoch(acq.datetime)
    gap = day - comp["day"]
    both = valid & (comp["day"] >= 0) & (gap > 0) & (gap <= MAX_CHANGE_GAP_DAYS)
    if both.sum() < 1000:
        return 0
    pre = {b: comp[b].astype("float32") for b in COMPOSITE_BANDS}
    n_events = 0
    # 1) new fractures on monitored ice / rock slopes (ridge filter computed once, only where watched)
    fi = None
    for site in inv.sites_near(session, ctx.cfg.id, _tile_box(tile), ("glacier", "slope")):
        region = inv.site_mask(site, tile.grid) & st.watch
        if region.sum() < 200:
            continue
        cy, cx = ndimage.center_of_mass(region)
        if not tile.in_core(cy, cx):
            continue
        if fi is None:
            fi = _fracture_indices(pre["B08"], bands["B08"], st.watch)
        res = new_fractures(pre["B08"], bands["B08"], region, tile.grid.res, both, fi=fi)
        if res is None or res["observed_fraction"] < 0.5:
            continue
        _upsert_obs(session, site.id, "surface_change", "S2", acq.key, acq.datetime, res,
                    quality=res["observed_fraction"], method="sato-ridge-diff")
        touched.add(site.id)
    # 2) vegetation loss / fresh scars below the treeline
    regions = detect_vegetation_loss(pre, bands, both, tile.grid, st.slope)
    for r in regions:
        cy = (r.mask_bbox[0] + r.mask_bbox[2]) / 2
        cx = (r.mask_bbox[1] + r.mask_bbox[3]) / 2
        if not tile.in_core(cy, cx):
            continue
        _add_event(session, ctx, tile, r, acq, pre_day=int(np.median(comp["day"][both])))
        n_events += 1
    return n_events


def _fracture_indices(pre_nir: np.ndarray, post_nir: np.ndarray, watch: np.ndarray):
    """Ridge (crack) indices for both images, computed only over the watched terrain's bounding box."""
    from himsat.detect.change import fracture_index

    f0 = np.zeros(pre_nir.shape, "float32")
    f1 = np.zeros(pre_nir.shape, "float32")
    rows, cols = np.nonzero(watch.any(axis=1))[0], np.nonzero(watch.any(axis=0))[0]
    if rows.size == 0:
        return f0, f1
    sl = (slice(max(rows[0] - 8, 0), rows[-1] + 9), slice(max(cols[0] - 8, 0), cols[-1] + 9))
    f0[sl] = fracture_index(pre_nir[sl], sigmas=(1.5,))
    f1[sl] = fracture_index(post_nir[sl], sigmas=(1.5,))
    return f0, f1


def _tile_box(tile: Tile):
    from shapely.geometry import box

    return box(*tile.grid.lonlat_bbox())


def _upsert_obs(session: Session, site_id: int, kind: str, sensor: str, scene_key: str, when: datetime,
                values: dict, quality: float, method: str, reference_at: datetime | None = None,
                geom=None) -> Observation:
    obs = session.scalars(select(Observation).where(Observation.site_id == site_id, Observation.kind == kind,
                                                    Observation.scene_key == scene_key)).first()
    if obs is None:
        obs = Observation(site_id=site_id, kind=kind, sensor=sensor, scene_key=scene_key, observed_at=when)
        session.add(obs)
    obs.values, obs.quality, obs.method, obs.reference_at = values, round(float(quality), 3), method, reference_at
    if geom is not None:
        set_geom(obs, geom)
    return obs


def _add_event(session: Session, ctx: AOIContext, tile: Tile, r, acq: Acquisition, pre_day: int | None = None,
               sensor: str = "S2", pre_at: datetime | None = None) -> ChangeEvent:
    from datetime import timedelta

    from himsat.geo.geometry import mask_to_lonlat_geom
    from himsat.pipeline.context import EPOCH

    r0, c0, r1, c1 = r.mask_bbox
    geom = mask_to_lonlat_geom(r.mask, tile.grid.window(r0, c0, r1 - r0, c1 - c0), simplify_m=tile.grid.res)
    ev = ChangeEvent(aoi_id=ctx.cfg.id, kind=r.kind, sensor=sensor, detected_at=acq.datetime,
                     pre_at=pre_at or (EPOCH + timedelta(days=pre_day) if pre_day is not None else None),
                     lon=r.lon, lat=r.lat, area_m2=r.area_m2, confidence=r.confidence,
                     attrs={**r.attrs, "mean_change": r.mean_change}, scene_key=acq.key)
    set_geom(ev, geom)
    session.add(ev)
    return ev


__all__ = ["process_s2", "to_lonlat", "rasterize"]
