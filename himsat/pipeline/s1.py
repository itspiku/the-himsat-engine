"""Sentinel-1 acquisition processing (cloud-independent):

* offset tracking against the previous pass on the same relative orbit gives site velocities and
  instability hotspots
* backscatter change gives mass-movement events and landslide sites
* dark-water mapping measures glacial-lake area under monsoon cloud
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
from scipy import ndimage
from shapely.geometry import MultiPoint
from skimage import measure
from sqlalchemy import select
from sqlalchemy.orm import Session

from himsat.db.models import CellVelocity, Site
from himsat.detect.cells import anomaly, cell_polygon, cell_stats, neighbours, weighted_mean
from himsat.detect.change import detect_sar_changes
from himsat.detect.lakes import sar_water_area
from himsat.detect.velocity import (
    _nan_median_filter,
    correct_stable_ground,
    downslope,
    offset_tracking,
    summarize_region,
)
from himsat.geo.grid import Tile, reproject_geom, to_lonlat
from himsat.ingest.loader import load_s1
from himsat.ingest.stac import Acquisition
from himsat.pipeline.context import AOIContext
from himsat.pipeline.s2 import _add_event, _tile_box, _upsert_obs
from himsat.sites import inventory as inv

log = logging.getLogger(__name__)

# Point-wise hotspots catch only *fast* motion (surges, imminent failure) where speckle noise can't
# reach. Slow anomalies are left to the statistically gated watch cells. In testing, a 0.15 m/day
# threshold produced dozens of spurious sites from spring snow decorrelation.
HOTSPOT_MIN_NODES = 6
HOTSPOT_MIN_V = 1.0  # m/day, downslope
HOTSPOT_MIN_RATIO = 3.0
HOTSPOT_MIN_CORR = 0.2
EVENT_MIN_AREA = 50_000.0  # m²: smaller backscatter changes are not recorded
ATTACH_MIN_AREA = 100_000.0  # m²: and must be this large / confident to count as evidence at a site
ATTACH_MIN_CONF = 0.7
LANDSLIDE_MIN_AREA = 250_000.0  # m²
LANDSLIDE_MIN_CONF = 0.75
LANDSLIDE_MIN_RELIEF = 300.0


def find_references(acq: Acquisition, history: list[Acquisition], max_days: int, max_refs: int = 3
                    ) -> list[Acquisition]:
    """Earlier acquisitions on the same relative orbit (identical viewing geometry), newest first.

    Pairing each image with several predecessors (≈12, 24, 36 days) gives longer baselines:
    velocity noise scales with 1/Δt, so slow creep becomes measurable.
    """
    cands = [a for a in history if a.relative_orbit == acq.relative_orbit and a.datetime < acq.datetime
             and timedelta(days=5) <= (acq.datetime - a.datetime) <= timedelta(days=max_days)]
    return sorted(cands, key=lambda a: a.datetime, reverse=True)[:max_refs]


def process_s1(session: Session, ctx: AOIContext, acq: Acquisition, refs: list[Acquisition],
               named_places: list, touched: set[int]) -> dict:
    stats = {"tiles": 0, "pairs": 0, "velocity_obs": 0, "hotspots": 0, "cells": 0, "anomalous_cells": 0,
             "events": 0, "lake_obs": 0}
    ref = refs[0] if refs else None
    updated_cells: set[str] = set()
    for tile in ctx.tiles:
        st = ctx.tile_static(tile)
        lake_sites = [s for s in inv.sites_near(session, ctx.cfg.id, _tile_box(tile), ("glacial_lake", "barrier_lake"))
                      if _in_core(tile, s)]
        if ref is None and not lake_sites:
            continue
        # fetch this pass and its references concurrently (each load parallelises its own items)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1 + len(refs)) as ex:
            loaded = list(ex.map(lambda a, g=tile.grid: load_s1(a, g, ctx.catalog, with_vh=False), [acq, *refs]))
        sec, ref_imgs = loaded[0], loaded[1:]
        if sec is None or np.isfinite(sec.vv_db).mean() < 0.2:
            continue
        stats["tiles"] += 1
        # --- lake area under cloud -------------------------------------------------------
        for s in lake_sites:
            m = inv.site_mask(s, tile.grid)
            if m.sum() < 20:
                continue
            res = sar_water_area(sec.vv_db, tile.grid, m, st.slope)
            if res is None:
                continue
            area, q, _ = res
            _upsert_obs(session, s.id, "lake_area", "S1", acq.key, acq.datetime,
                        {"area_m2": round(area, 1), "reference_area_m2": round(float(m.sum() * tile.grid.pixel_area), 1)},
                        quality=q, method="s1-otsu-dark-water")
            touched.add(s.id)
            stats["lake_obs"] += 1
        if ref is None:
            continue
        # --- velocity: every reference on this orbit ---------------------------------------
        pre = None
        for k, (rf, p_img) in enumerate(zip(refs, ref_imgs, strict=True)):
            if p_img is None:
                continue
            if k == 0:
                pre = p_img
            if st.watch.any():
                _velocity_pair(session, ctx, tile, st, p_img, sec, rf, acq, named_places, touched, stats,
                               updated_cells, fast_hotspots=(k == 0))
        if pre is None:
            continue
        # --- backscatter change (shortest pair only) -----------------------------------------------------------
        regions = detect_sar_changes(pre.vv_db, sec.vv_db, tile.grid, st.dem, st.slope, thr_db=3.0,
                                     min_area_m2=EVENT_MIN_AREA, ice=st.glacier)
        for r in regions:
            cy = (r.mask_bbox[0] + r.mask_bbox[2]) / 2
            cx = (r.mask_bbox[1] + r.mask_bbox[3]) / 2
            if not tile.in_core(cy, cx):
                continue
            r.attrs["orbit"] = acq.relative_orbit
            ev = _add_event(session, ctx, tile, r, acq, sensor="S1", pre_at=ref.datetime)
            stats["events"] += 1
            if (r.area_m2 < ATTACH_MIN_AREA or r.confidence < ATTACH_MIN_CONF or r.attrs.get("disturbed")
                    or r.attrs.get("snow_like")):
                continue
            if _after_wet_snow(session, ctx, ev, acq):
                stats["snow_reversal"] = stats.get("snow_reversal", 0) + 1
                continue  # refreeze / melt-out of an earlier wet-snow patch
            # independent confirmation: the same change seen from a different viewing geometry
            partner = _confirming_event(session, ctx, ev, acq)
            if partner is None:
                stats["unconfirmed"] = stats.get("unconfirmed", 0) + 1
                continue
            stats["confirmed"] = stats.get("confirmed", 0) + 1
            ev.attrs = {**ev.attrs, "confirmed_by": partner.id}
            partner.attrs = {**partner.attrs, "confirmed_by": ev.id}
            _attach_change(session, ctx, ev, acq, touched)
            if (r.area_m2 >= LANDSLIDE_MIN_AREA and r.confidence >= LANDSLIDE_MIN_CONF
                    and r.attrs.get("relief_m", 0) >= LANDSLIDE_MIN_RELIEF):
                _landslide_site(session, ctx, tile, r, ev, acq, named_places, touched)
    if updated_cells:
        stats["anomalous_cells"] = _cell_anomalies(session, ctx, acq, updated_cells, named_places, touched)
    return stats


def _velocity_pair(session: Session, ctx: AOIContext, tile: Tile, st, pre, sec, ref: Acquisition,
                   acq: Acquisition, named_places: list, touched: set[int], stats: dict, updated_cells: set[str],
                   fast_hotspots: bool) -> None:
    dt = (acq.datetime - ref.datetime).total_seconds() / 86400.0
    # track the watch zone plus a sample of stable terrain (needed for co-registration correction)
    vf = offset_tracking(pre.vv_db, sec.vv_db, tile.grid, dt, where=st.watch | st.stable_sample, chip=64, step=16,
                         min_corr=0.1, min_snr=3.0)
    info = correct_stable_ground(vf, st.stable)
    nmad = info["stable_nmad_px"]
    if nmad is None or nmad >= 1.0 or vf.valid.sum() <= 20:
        log.info("S1 %s/%s tile %s: velocity rejected (stable NMAD %s px)", ref.key, acq.key, tile.index, nmad)
        return
    stats["pairs"] += 1
    pair_key = f"{acq.key}|{ref.datetime:%Y%m%dT%H%M}"
    ctx.save_velocity(tile, vf, ref.datetime, acq.datetime, acq.relative_orbit,
                      {"nmad_px": nmad, "orbit": acq.relative_orbit, "key": pair_key})
    for c in cell_stats(vf, st.aspect, st.watch, tile, ctx.grid, nmad):
        _upsert_cell(session, ctx.cfg.id, c, pair_key, ref.datetime, acq.datetime, acq.relative_orbit)
        updated_cells.add(c.cell)
        stats["cells"] += 1
    stats["velocity_obs"] += _site_velocities(session, ctx, tile, st, vf, nmad, acq, ref, touched, pair_key)
    if fast_hotspots:
        stats["hotspots"] += _hotspots(session, ctx, tile, st, vf, nmad, acq, ref, named_places, touched)


def _upsert_cell(session: Session, aoi_id: str, c, pair_key: str, start: datetime, end: datetime,
                 orbit: int | None) -> None:
    row = session.scalars(select(CellVelocity).where(CellVelocity.aoi_id == aoi_id, CellVelocity.cell == c.cell,
                                                     CellVelocity.pair_key == pair_key)).first()
    if row is None:
        session.add(CellVelocity(aoi_id=aoi_id, cell=c.cell, pair_key=pair_key, pair_start=start, pair_end=end,
                                 orbit=orbit, v=c.v, se=c.se, n=c.n))
    else:
        row.v, row.se, row.n = c.v, c.se, c.n


def _cell_rows(session: Session, aoi_id: str, cells: list[str], until: datetime, days: int = 400):
    q = select(CellVelocity).where(CellVelocity.aoi_id == aoi_id, CellVelocity.cell.in_(cells),
                                   CellVelocity.pair_end <= until,
                                   CellVelocity.pair_end >= until - timedelta(days=days))
    return list(session.scalars(q))


def _cell_anomalies(session: Session, ctx: AOIContext, acq: Acquisition, cells: set[str], named_places: list,
                    touched: set[int]) -> int:
    """Test updated cells; promote significant ones (merged with anomalous neighbours) to slope sites."""
    session.flush()
    by_cell: dict[str, list] = {}
    for r in _cell_rows(session, ctx.cfg.id, sorted(cells), acq.datetime, days=400):  # incl. last season
        by_cell.setdefault(r.cell, []).append((r.pair_end, r.v, r.se, r.orbit))
    hot = {}
    for cell, rr in by_cell.items():
        a = anomaly(rr, acq.datetime)
        if a is not None and a.significant:
            hot[cell] = a
    cell_sites = [s for s in session.scalars(select(Site).where(Site.aoi_id == ctx.cfg.id, Site.kind == "slope",
                                                                Site.status == "active"))
                  if (s.attrs or {}).get("cells")]
    covered = {c for s in cell_sites for c in s.attrs["cells"]}
    seen: set[str] = set()
    for cell in sorted(hot):
        if cell in seen or cell in covered:
            continue
        cluster, stack = [], [cell]
        while stack:
            c = stack.pop()
            if c in seen or c not in hot or c in covered:
                continue
            seen.add(c)
            cluster.append(c)
            stack.extend(neighbours(c))
        cell_sites.append(_create_cell_site(session, ctx, cluster, hot, acq, named_places))
    for s in cell_sites:
        if set(s.attrs["cells"]) & cells and _cell_site_observations(session, ctx, s, acq.datetime):
            touched.add(s.id)
    return len(hot)


def _create_cell_site(session: Session, ctx: AOIContext, cluster: list[str], hot: dict, acq: Acquisition,
                      named_places: list) -> Site:
    from shapely.ops import unary_union

    from himsat.geo.geometry import rasterize
    from himsat.ingest.dem import slope_aspect

    geom = unary_union([cell_polygon(c, ctx.grid) for c in cluster])
    dem, g = ctx.routing_dem, ctx.routing_grid
    low = inv.lowest_point(geom, dem, g)
    m = rasterize([geom], g)
    slope, _ = slope_aspect(dem, g.res)
    a = max((hot[c] for c in cluster), key=lambda x: x.z)
    attrs = {"cells": sorted(cluster), "lowest_point": [low[0], low[1]] if low else None,
             "mean_slope_deg": float(slope[m].mean()) if m.any() else None,
             "elev_min_m": float(dem[m].min()) if m.any() else None,
             "elev_max_m": float(dem[m].max()) if m.any() else None,
             "discovered_z": round(a.z, 2), "discovered_ratio": round(a.ratio, 2),
             "discovered_v_m_day": round(a.v_recent, 4)}
    return inv.create_site(session, ctx.cfg, kind="slope", geom_lonlat=geom, source="discovered:watch-cell-anomaly",
                           attrs=attrs, elevation_m=float(np.median(dem[m])) if m.any() else None,
                           first_seen=acq.datetime, named_places=named_places)


def _cell_site_observations(session: Session, ctx: AOIContext, site: Site, until: datetime) -> bool:
    """(Back)fill a cell-based site's velocity observations: one per pair, combining its cells."""
    by_pair: dict[str, list] = {}
    for r in _cell_rows(session, ctx.cfg.id, site.attrs["cells"], until):
        by_pair.setdefault(r.pair_key, []).append(r)
    for key, rs in by_pair.items():
        v, se = weighted_mean([r.v for r in rs], [r.se for r in rs], clip=99)
        r0 = rs[0]
        _upsert_obs(session, site.id, "velocity", "S1", key, r0.pair_end,
                    {"v_down_median": v, "v_down_se": se, "n_points": int(sum(r.n for r in rs)), "coverage": 1.0,
                     "dt_days": (r0.pair_end - r0.pair_start).total_seconds() / 86400, "orbit": r0.orbit,
                     "cells": len(rs)},
                    quality=1.0, method="s1-watch-cells", reference_at=r0.pair_start)
    return bool(by_pair)


def _in_core(tile: Tile, site: Site) -> bool:
    r, c = tile.grid.lonlat_rowcol(site.lon, site.lat)
    return tile.in_core(r, c)


def _site_velocities(session: Session, ctx: AOIContext, tile: Tile, st, vf, nmad: float, acq: Acquisition,
                     ref: Acquisition, touched: set[int], pair_key: str) -> int:
    n = 0
    for s in inv.sites_near(session, ctx.cfg.id, _tile_box(tile), ("glacier", "slope")):
        if not _in_core(tile, s) or (s.attrs or {}).get("cells"):
            continue  # cell-based slope sites are observed through their cells
        region = inv.site_mask(s, tile.grid) & st.watch
        if region.sum() < 50:
            continue
        summ = summarize_region(vf, region, st.aspect, noise_px=nmad)
        if summ is None or summ.get("n_points", 0) < 4:
            continue
        pw = _pointwise_ratio(ctx, tile, st, vf, region, acq.datetime)
        if pw is not None:
            summ["ratio_pointwise"] = pw
        summ["orbit"] = acq.relative_orbit
        _upsert_obs(session, s.id, "velocity", "S1", pair_key, acq.datetime, summ,
                    quality=min(1.0, summ["coverage"]), method="s1-offset-tracking-ncc", reference_at=ref.datetime)
        touched.add(s.id)
        n += 1
    return n


def _baseline_down(ctx: AOIContext, tile: Tile, st, vf, when: datetime) -> np.ndarray | None:
    """Per-node median downslope velocity over earlier pairs (18–150 days before)."""
    hist = ctx.velocity_history(tile, when, 18, 150)
    if len(hist) < 2:
        return None
    a = np.radians(st.aspect[np.ix_(vf.rows, vf.cols)])
    stack = []
    for h in hist:
        if h["dx"].shape != vf.dx.shape:
            continue
        d = (h["dx"] * np.sin(a) + h["dy"] * np.cos(a)) * tile.grid.res / h["dt"]
        stack.append(_nan_median_filter(d, 3))
    if len(stack) < 2:
        return None
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.stack(stack), axis=0)


def _pointwise_ratio(ctx, tile, st, vf, region, when) -> float | None:
    base = _baseline_down(ctx, tile, st, vf, when)
    if base is None:
        return None
    now = _nan_median_filter(downslope(vf, st.aspect), 3)
    m = vf.lattice_mask(region) & np.isfinite(now) & np.isfinite(base)
    if m.sum() < 4:
        return None
    v_now, v_base = float(np.median(now[m])), float(np.median(base[m]))
    return v_now / max(v_base, 0.05) if v_now > 0.05 else 1.0


def _hotspots(session: Session, ctx: AOIContext, tile: Tile, st, vf, nmad: float, acq: Acquisition,
              ref: Acquisition, named_places: list, touched: set[int]) -> int:
    """Clusters of nodes moving much faster than their own history → new/updated 'slope' sites."""
    base = _baseline_down(ctx, tile, st, vf, acq.datetime)
    if base is None:
        return 0
    now = _nan_median_filter(downslope(vf, st.aspect), 3)
    noise = 0.7 * nmad * tile.grid.res / vf.dt_days
    thr = max(HOTSPOT_MIN_V, 3.0 * noise)
    hot = np.isfinite(now) & np.isfinite(base) & (now >= thr) & (now >= HOTSPOT_MIN_RATIO * np.maximum(base, 0.05))
    hot &= vf.lattice_mask(st.watch)
    lab = measure.label(hot, connectivity=2)
    n = 0
    for rp in measure.regionprops(lab):
        if rp.area < HOTSPOT_MIN_NODES:
            continue
        rr, cc = rp.coords[:, 0], rp.coords[:, 1]
        prow, pcol = vf.rows[rr], vf.cols[cc]
        if not tile.in_core(float(np.mean(prow)), float(np.mean(pcol))):
            continue
        x, y = tile.grid.xy(prow, pcol)
        hull = MultiPoint(np.column_stack([x, y])).convex_hull.buffer(vf.grid.res * 8)
        geom = reproject_geom(hull, tile.grid.epsg, 4326)
        ratio = float(np.median(now[rr, cc] / np.maximum(base[rr, cc], 0.05)))
        v = float(np.median(now[rr, cc]))
        corr = float(np.nanmean(vf.corr[rr, cc]))
        if corr < HOTSPOT_MIN_CORR:
            continue  # decorrelated (wet snow, vegetation, layover): offsets are not trustworthy
        site = inv.match_site(session, ctx.cfg.id, geom, ("slope",), buffer_deg=0.002)
        if site is None:
            # a new fast-moving area must be seen from two independent viewing geometries
            ev = _hotspot_event(session, ctx, geom, hull.area, acq, v, ratio, corr)
            if _confirming_event(session, ctx, ev, acq, kind="velocity_hotspot") is None:
                continue
            rows_px = np.clip(prow, 0, st.dem.shape[0] - 1)
            cols_px = np.clip(pcol, 0, st.dem.shape[1] - 1)
            k = int(np.argmin(st.dem[rows_px, cols_px]))
            low_lon, low_lat = to_lonlat(float(x[k]), float(y[k]), tile.grid.epsg)
            from himsat.geo.geometry import rasterize

            m = rasterize([geom], tile.grid)
            attrs = {"mean_slope_deg": float(st.slope[m].mean()) if m.any() else None,
                     "lowest_point": [float(low_lon), float(low_lat)], "area_m2": float(hull.area),
                     "discovered_ratio": ratio, "discovered_v_m_day": v}
            site = inv.create_site(session, ctx.cfg, kind="slope", geom_lonlat=geom, source="discovered:velocity-hotspot",
                                   attrs=attrs, elevation_m=float(np.median(st.dem[rows_px, cols_px])),
                                   first_seen=acq.datetime, named_places=named_places)
        region = inv.site_mask(site, tile.grid) & st.watch
        summ = summarize_region(vf, region, st.aspect, noise_px=nmad) or {}
        summ.update({"ratio_pointwise": ratio, "hotspot_v_m_day": v, "hotspot_nodes": int(rp.area),
                     "orbit": acq.relative_orbit})
        _upsert_obs(session, site.id, "velocity", "S1", f"{acq.key}|{ref.datetime:%Y%m%dT%H%M}", acq.datetime, summ,
                    quality=min(1.0, summ.get("coverage", 0.5)), method="s1-offset-tracking-hotspot",
                    reference_at=ref.datetime, geom=geom)
        touched.add(site.id)
        n += 1
    return n


def _hotspot_event(session: Session, ctx: AOIContext, geom, area_m2: float, acq: Acquisition, v: float,
                   ratio: float, corr: float):
    from himsat.db.models import ChangeEvent
    from himsat.geo.geometry import set_geom

    c = geom.centroid
    ev = ChangeEvent(aoi_id=ctx.cfg.id, kind="velocity_hotspot", sensor="S1", detected_at=acq.datetime,
                     lon=c.x, lat=c.y, area_m2=float(area_m2), confidence=min(1.0, corr * 2),
                     attrs={"orbit": acq.relative_orbit, "v_m_day": v, "ratio": ratio, "mean_corr": corr},
                     scene_key=acq.key)
    set_geom(ev, geom)
    session.add(ev)
    session.flush()
    return ev


def _overlapping_events(session: Session, ctx: AOIContext, ev, since: datetime, until: datetime):
    from himsat.db.models import ChangeEvent
    from himsat.geo.geometry import from_geojson

    session.flush()
    g = from_geojson(ev.geom)
    if g is None:
        return []
    q = select(ChangeEvent).where(
        ChangeEvent.aoi_id == ctx.cfg.id, ChangeEvent.sensor == "S1", ChangeEvent.id != ev.id,
        ChangeEvent.detected_at >= since, ChangeEvent.detected_at <= until,
        ChangeEvent.max_lon >= g.bounds[0], ChangeEvent.min_lon <= g.bounds[2],
        ChangeEvent.max_lat >= g.bounds[1], ChangeEvent.min_lat <= g.bounds[3])
    out = []
    for other in session.scalars(q):
        og = from_geojson(other.geom)
        if og is not None and og.intersects(g):
            out.append(other)
    return out


def _after_wet_snow(session: Session, ctx: AOIContext, ev, acq: Acquisition, days: int = 45) -> bool:
    return any(o.kind == "mass_movement" and (o.attrs or {}).get("snow_like") for o in _overlapping_events(
        session, ctx, ev, acq.datetime - timedelta(days=days), acq.datetime))


def _confirming_event(session: Session, ctx: AOIContext, ev, acq: Acquisition, days: int = 15,
                      kind: str = "mass_movement"):
    """A prior strong change from a *different* relative orbit overlapping this one (within ``days``)."""
    for other in _overlapping_events(session, ctx, ev, acq.datetime - timedelta(days=days), acq.datetime):
        a = other.attrs or {}
        if other.kind != kind or a.get("orbit") == acq.relative_orbit or a.get("disturbed") or a.get("snow_like"):
            continue
        if kind == "mass_movement" and other.area_m2 < ATTACH_MIN_AREA * 0.5:
            continue
        return other
    return None


def _attach_change(session: Session, ctx: AOIContext, ev, acq: Acquisition, touched: set[int],
                   radius_deg: float = 0.015) -> None:
    """Link a backscatter-change event to monitored sites within ~1.5 km."""
    from shapely.geometry import Point

    from himsat.geo.geometry import from_geojson

    p = Point(ev.lon, ev.lat)
    for s in inv.sites_near(session, ctx.cfg.id, p.buffer(radius_deg),
                            ("glacier", "slope", "glacial_lake", "landslide", "barrier_lake")):
        sg = from_geojson(s.geom)
        if sg is None or sg.distance(p) > radius_deg:
            continue
        key = f"{acq.key}#ev{ev.id or 0}"
        session.flush()
        key = f"{acq.key}#ev{ev.id}"
        _upsert_obs(session, s.id, "sar_change", "S1", key, acq.datetime,
                    {"area_m2": ev.area_m2, "mean_change_db": ev.attrs.get("mean_change"), "event_id": ev.id,
                     "confidence": ev.confidence}, quality=ev.confidence, method="s1-log-ratio",
                    reference_at=ev.pre_at)
        touched.add(s.id)


def _landslide_site(session: Session, ctx: AOIContext, tile: Tile, r, ev, acq: Acquisition,
                    named_places: list, touched: set[int]) -> None:
    from himsat.geo.geometry import from_geojson

    geom = from_geojson(ev.geom)
    if geom is None:
        return
    existing = inv.match_site(session, ctx.cfg.id, geom, ("landslide",), buffer_deg=0.003)
    st = ctx.tile_static(tile)
    r0, c0, r1, c1 = r.mask_bbox
    sub = st.dem[r0:r1, c0:c1]
    z = np.where(r.mask, sub, np.inf)
    lr, lc = np.unravel_index(np.argmin(z), z.shape)
    x, y = tile.grid.xy(r0 + lr, c0 + lc)
    low_lon, low_lat = to_lonlat(float(x), float(y), tile.grid.epsg)
    if existing is None:
        attrs = {"area_m2": r.area_m2, "relief_m": r.attrs.get("relief_m"), "mean_slope_deg": r.attrs.get("mean_slope_deg"),
                 "lowest_point": [float(low_lon), float(low_lat)], "event_id": ev.id}
        existing = inv.create_site(session, ctx.cfg, kind="landslide", geom_lonlat=geom, source="discovered:sar-change",
                                   attrs=attrs, elevation_m=float(ndimage.median(sub, labels=r.mask.astype(int))),
                                   first_seen=acq.datetime, named_places=named_places)
    ev.site_id = existing.id
    session.flush()
    _upsert_obs(session, existing.id, "sar_change", "S1", f"{acq.key}#ev{ev.id}", acq.datetime,
                {"area_m2": ev.area_m2, "mean_change_db": ev.attrs.get("mean_change"), "event_id": ev.id,
                 "confidence": ev.confidence}, quality=ev.confidence, method="s1-log-ratio", reference_at=ev.pre_at)
    touched.add(existing.id)
