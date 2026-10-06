"""Monitoring cycle orchestration (live and hindcast).

A cycle:
  1. make sure the AOI, assets and the glacier inventory exist
  2. find new Sentinel-1/-2 acquisitions and register them (idempotent)
  3. process pending acquisitions in time order (S2: optical; S1: radar)
  4. after each acquisition, reassess every site it touched as of that acquisition's time
     (alerts are generated inside the assessment)

Processing is strictly chronological and uses only data up to each acquisition. A hindcast is
therefore the same code run over an archive window against a scratch database.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from rasterio.errors import RasterioIOError
from shapely.geometry import shape
from sqlalchemy import select
from sqlalchemy.orm import Session

from himsat.alerts.llm import LLMClient
from himsat.config import AOIConfig, Settings, get_aoi, get_settings
from himsat.db.models import PipelineRun, Scene, Site, utcnow
from himsat.db.session import init_db, session_scope
from himsat.detect.segmentation import get_segmenter
from himsat.geo.geometry import rasterize
from himsat.ingest.stac import Acquisition, group_acquisitions
from himsat.pipeline.assess import Assessor
from himsat.pipeline.context import AOIContext
from himsat.pipeline.s1 import find_references, process_s1
from himsat.pipeline.s2 import process_s2
from himsat.sites import inventory as inv

log = logging.getLogger(__name__)
SCENE_ATTEMPTS = 3


@dataclass
class CycleOptions:
    start: datetime | None = None
    end: datetime | None = None
    dispatch: bool = True
    sensors: tuple[str, ...] = ("S1", "S2", "INSAR")  # INSAR runs only if settings.insar_enabled
    data_latency: timedelta = timedelta(hours=6)  # hindcast: when results would have been available
    hindcast: bool = False
    min_glacier_km2: float = 0.5
    progress: object | None = None  # callable(msg)
    products_dir: Path | None = None  # hindcasts keep their own composites / velocity fields


@dataclass
class CycleResult:
    run_id: int | None = None
    scenes_processed: int = 0
    scenes_skipped: int = 0
    scenes_failed: int = 0
    alerts: list[int] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _readable(name: str) -> bool:
    return bool(name) and all(ord(ch) < 0x0250 or 0x0900 <= ord(ch) <= 0x097F or ch in "’‘–—" for ch in name)


def ensure_glacier_sites(session: Session, ctx: AOIContext, min_km2: float) -> int:
    """Create 'glacier' sites from the inventory (OSM natural=glacier ≥ min_km2 within the AOI)."""
    existing = {s.attrs.get("inventory_id") for s in session.scalars(
        select(Site).where(Site.aoi_id == ctx.cfg.id, Site.kind == "glacier"))}
    w, s_, e, n = ctx.cfg.bbox
    dem, g = ctx.routing_dem, ctx.routing_grid
    from himsat.ingest.dem import slope_aspect

    slope, _ = slope_aspect(dem, g.res)
    created = 0
    for f in ctx.glacier_features:
        p = f["properties"]
        if p.get("area_km2", 0) < min_km2 or p["external_id"] in existing:
            continue
        geom = shape(f["geometry"])
        c = geom.representative_point()
        if not (w <= c.x <= e and s_ <= c.y <= n):
            continue
        m = rasterize([geom], g)
        if not m.any():
            continue
        low = inv.lowest_point(geom, dem, g)
        attrs = {"inventory_id": p["external_id"], "area_m2": p["area_km2"] * 1e6,
                 "mean_slope_deg": float(slope[m].mean()), "elev_min_m": float(dem[m].min()),
                 "elev_max_m": float(dem[m].max()), "lowest_point": [low[0], low[1]] if low else None}
        own = (p["name"], p.get("name_ne", "")) if p.get("name") and _readable(p["name"]) else None
        inv.create_site(session, ctx.cfg, kind="glacier", geom_lonlat=geom, source="inventory:osm", attrs=attrs,
                        elevation_m=float(np.median(dem[m])), own_name=own)
        created += 1
    return created


def run_insar(ctx: AOIContext, db_url: str | None, start: datetime, end: datetime, assessor: Assessor,
              opts: CycleOptions, res: CycleResult, say) -> dict:
    """Submit new HyP3 InSAR pairs, then ingest every finished product not seen before."""
    from himsat.ingest import insar

    cfg = ctx.cfg
    client = insar.InsarClient(ctx.settings)
    project = insar.project_name(cfg.id)
    pairs = insar.find_pairs(insar.search_slc(cfg, start - timedelta(days=30), end))
    submitted = client.submit(pairs, project)
    products = [p for p in (insar.read_product(d) for d in client.collect(project, ctx.products / "insar")) if p]
    with session_scope(db_url) as session:
        done = {k for (k,) in session.execute(select(Scene.key).where(Scene.aoi_id == cfg.id, Scene.sensor == "INSAR"))}
    ingested = 0
    for prod in sorted(products, key=lambda p: p.sec_time):
        if prod.key in done or not (start <= prod.sec_time <= end + timedelta(days=1)):
            continue
        with session_scope(db_url) as session:
            touched: set[int] = set()
            n = insar.ingest_product(session, ctx, prod, touched)
            session.add(Scene(aoi_id=cfg.id, sensor="INSAR", key=prod.key, acquired_at=prod.sec_time,
                              platform="S1", status="processed", stats={"site_obs": n}, processed_at=utcnow(),
                              item_ids=[prod.path.name]))
            now = prod.sec_time + opts.data_latency if opts.hindcast else utcnow()
            for sid in sorted(touched):
                _, alert = assessor.assess(session, session.get(Site, sid), as_of=prod.sec_time, now=now)
                if alert is not None:
                    res.alerts.append(alert.id)
        ingested += 1
        say(f"InSAR {prod.path.name}: {n} site observations")
    return {"pairs": len(pairs), "submitted": submitted, "products": len(products), "ingested": ingested}


def _register(session: Session, cfg: AOIConfig, acqs: list[Acquisition]) -> None:
    have = {(s.sensor, s.key) for s in session.scalars(select(Scene).where(Scene.aoi_id == cfg.id))}
    for a in acqs:
        if (a.sensor, a.key) in have:
            continue
        session.add(Scene(aoi_id=cfg.id, sensor=a.sensor, key=a.key, acquired_at=a.datetime, platform=a.platform,
                          relative_orbit=a.relative_orbit, cloud_cover=a.cloud_cover,
                          item_ids=[i.id for i in a.items]))


def run_cycle(aoi_id: str, opts: CycleOptions | None = None, db_url: str | None = None,
              settings: Settings | None = None) -> CycleResult:
    opts = opts or CycleOptions()
    settings = settings or get_settings()
    cfg = get_aoi(aoi_id, settings)
    init_db(db_url)
    ctx = AOIContext(cfg, settings, products_dir=opts.products_dir)
    segmenter = get_segmenter(settings)
    llm = LLMClient.from_settings(settings)
    assessor = Assessor(ctx, settings, llm, dispatch=opts.dispatch and not opts.hindcast)
    res = CycleResult()
    say = opts.progress or (lambda m: log.info(m))

    with session_scope(db_url) as session:
        run = PipelineRun(aoi_id=cfg.id, kind="hindcast" if opts.hindcast else "monitor")
        session.add(run)
        inv.ensure_aoi(session, cfg)
        n_assets = inv.sync_assets(session, cfg)
        n_gl = ensure_glacier_sites(session, ctx, opts.min_glacier_km2)
        session.flush()
        res.run_id = run.id
        say(f"AOI {cfg.id}: {len(ctx.tiles)} tiles, {n_assets} assets, {n_gl} new glacier sites")
        glacier_named = inv.glacier_named_places(ctx.glacier_features)
        settlement_named = [(a["name"], a["name_ne"], a["lon"], a["lat"]) for a in assessor.assets(session)
                            if a["kind"] in ("town", "settlement") and a["name"]]
        # alerts are read in Nepali and English: only use names written in Latin or Devanagari script
        named = [n for n in glacier_named + settlement_named if _readable(n[0]) or _readable(n[1])]

    end = opts.end or utcnow()
    if opts.start is None:
        with session_scope(db_url) as session:
            last = session.scalar(select(Scene.acquired_at).where(Scene.aoi_id == cfg.id, Scene.status == "processed")
                                  .order_by(Scene.acquired_at.desc()).limit(1))
        start = last - timedelta(days=1) if last else end - timedelta(days=settings.lookback_days)
    else:
        start = opts.start
    search_start = start - timedelta(days=cfg.s1_max_pair_days + 1)
    say(f"searching {search_start:%Y-%m-%d} … {end:%Y-%m-%d}")
    acqs: list[Acquisition] = []
    if "S2" in opts.sensors:
        acqs += group_acquisitions(ctx.catalog.search_s2(cfg.bbox, search_start, end))
    s1_all: list[Acquisition] = []
    if "S1" in opts.sensors:
        s1_all = group_acquisitions(ctx.catalog.search_s1(cfg.bbox, search_start, end))
        acqs += s1_all
    acqs.sort(key=lambda a: a.datetime)
    with session_scope(db_url) as session:
        _register(session, cfg, acqs)

    with session_scope(db_url) as session:
        done = {(s.sensor, s.key) for s in session.scalars(
            select(Scene).where(Scene.aoi_id == cfg.id, Scene.status.in_(("processed", "skipped"))))}
    todo = [a for a in acqs if a.datetime >= start and (a.sensor, a.key) not in done]
    say(f"{len(todo)} acquisitions to process ({sum(a.sensor == 'S1' for a in todo)} S1, "
        f"{sum(a.sensor == 'S2' for a in todo)} S2)")

    def process_one(acq: Acquisition) -> dict:
        with session_scope(db_url) as session:
            touched: set[int] = set()
            if acq.sensor == "S2":
                stats = process_s2(session, ctx, acq, segmenter, named, touched)
            else:
                refs = find_references(acq, s1_all, cfg.s1_max_pair_days)
                stats = process_s1(session, ctx, acq, refs, named, touched)
                stats["references"] = len(refs)
            scene = session.scalars(select(Scene).where(Scene.aoi_id == cfg.id, Scene.sensor == acq.sensor,
                                                        Scene.key == acq.key)).one()
            scene.status = "processed" if stats.get("tiles") else "skipped"
            scene.stats, scene.processed_at = stats, utcnow()
            now = acq.datetime + opts.data_latency if opts.hindcast else utcnow()
            for sid in sorted(touched):
                site = session.get(Site, sid)
                _, alert = assessor.assess(session, site, as_of=acq.datetime, now=now)
                if alert is not None:
                    res.alerts.append(alert.id)
        return stats

    for i, acq in enumerate(todo, 1):
        t0 = datetime.now(UTC)
        try:
            for attempt in range(1, SCENE_ATTEMPTS + 1):
                try:
                    stats = process_one(acq)
                    break
                except (RasterioIOError, OSError) as e:  # transient remote-read failure: retry the scene
                    if attempt == SCENE_ATTEMPTS:
                        raise
                    log.warning("scene %s attempt %d failed (%s); retrying", acq.key, attempt, e)
            if stats.get("tiles"):
                res.scenes_processed += 1
            else:
                res.scenes_skipped += 1
            say(f"[{i}/{len(todo)}] {acq.key}: {stats} ({(datetime.now(UTC) - t0).total_seconds():.0f}s)")
        except Exception as e:  # keep going: one bad scene must not stop monitoring
            res.scenes_failed += 1
            log.error("scene %s failed: %s\n%s", acq.key, e, traceback.format_exc())
            with session_scope(db_url) as session:
                sc = session.scalars(select(Scene).where(Scene.aoi_id == cfg.id, Scene.sensor == acq.sensor,
                                                         Scene.key == acq.key)).first()
                if sc:
                    sc.status, sc.stats = "failed", {"error": str(e)[:500]}

    if settings.insar_enabled and "INSAR" in opts.sensors:
        try:
            res.stats["insar"] = run_insar(ctx, db_url, start, end, assessor, opts, res, say)
        except Exception as e:  # InSAR is an add-on: its failure must never stop monitoring
            log.error("InSAR step failed: %s\n%s", e, traceback.format_exc())

    with session_scope(db_url) as session:
        run = session.get(PipelineRun, res.run_id)
        run.finished_at = utcnow()
        run.status = "ok" if res.scenes_failed == 0 else "partial"
        res.stats = {**res.stats, "processed": res.scenes_processed, "skipped": res.scenes_skipped,
                     "failed": res.scenes_failed, "alerts": len(res.alerts), "start": start.isoformat(),
                     "end": end.isoformat()}
        run.stats = res.stats
    return res
