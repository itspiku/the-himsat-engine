"""Site inventory: creation, spatial matching, naming, asset sync and exposure computation."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from himsat.config import AOIConfig, get_settings
from himsat.db.models import AOI, Asset, Site, SiteExposure
from himsat.geo.geometry import from_geojson, rasterize, set_geom
from himsat.geo.grid import reproject_geom
from himsat.risk.exposure import ExposureParams, exposed_assets, flow_path_from_lonlat
from himsat.risk.model import load_risk_config
from himsat.sites.osm import fetch_assets

log = logging.getLogger(__name__)

KIND_LETTER = {"glacial_lake": "L", "glacier": "G", "slope": "S", "barrier_lake": "B", "landslide": "M"}
KIND_LABEL = {"glacial_lake": ("Glacial lake", "हिमताल"), "glacier": ("Glacier", "हिमनदी"),
              "slope": ("Unstable slope", "अस्थिर भिर"), "barrier_lake": ("Landslide-dammed lake", "पहिरोले थुनिएको ताल"),
              "landslide": ("Landslide", "पहिरो")}


def curated_assets_path() -> Path:
    return get_settings().data_dir / "assets" / "curated.geojson"


def ensure_aoi(session: Session, cfg: AOIConfig) -> AOI:
    aoi = session.get(AOI, cfg.id)
    from shapely.geometry import box

    if aoi is None:
        aoi = AOI(id=cfg.id, name=cfg.name, name_ne=cfg.name_ne, config=cfg.model_dump(mode="json"))
        session.add(aoi)
    aoi.name, aoi.name_ne, aoi.config = cfg.name, cfg.name_ne, cfg.model_dump(mode="json")
    set_geom(aoi, box(*cfg.bbox))
    session.flush()
    return aoi


def next_code(session: Session, cfg: AOIConfig, kind: str) -> str:
    prefix = f"{cfg.code_prefix}-{KIND_LETTER[kind]}"
    n = session.scalar(select(func.count()).select_from(Site).where(Site.code.like(f"{prefix}%"))) or 0
    while True:
        n += 1
        code = f"{prefix}{n:04d}"
        if session.scalar(select(Site.id).where(Site.code == code)) is None:
            return code


def sites_near(session: Session, aoi_id: str, geom_lonlat: BaseGeometry, kinds: tuple[str, ...],
               pad_deg: float = 0.002) -> list[Site]:
    minx, miny, maxx, maxy = geom_lonlat.bounds
    q = select(Site).where(Site.aoi_id == aoi_id, Site.kind.in_(kinds), Site.status == "active",
                           Site.max_lon >= minx - pad_deg, Site.min_lon <= maxx + pad_deg,
                           Site.max_lat >= miny - pad_deg, Site.min_lat <= maxy + pad_deg)
    return list(session.scalars(q))


def match_site(session: Session, aoi_id: str, geom_lonlat: BaseGeometry, kinds: tuple[str, ...],
               buffer_deg: float = 0.0004) -> Site | None:
    """Existing site whose outline overlaps the detection (≈40 m tolerance)."""
    best, best_area = None, 0.0
    g = geom_lonlat.buffer(buffer_deg)
    for s in sites_near(session, aoi_id, geom_lonlat, kinds):
        sg = from_geojson(s.geom)
        if sg is None:
            continue
        inter = sg.intersection(g).area
        if inter > best_area:
            best, best_area = s, inter
    return best


def nearest_name(lon: float, lat: float, named: list[tuple[str, str, float, float]], max_km: float = 6.0):
    best, best_d = None, max_km
    for name, name_ne, x, y in named:
        d = math.hypot((x - lon) * 111.32 * math.cos(math.radians(lat)), (y - lat) * 110.57)
        if d < best_d:
            best, best_d = (name, name_ne), d
    return best


def make_names(kind: str, code: str, near: tuple[str, str] | None, own: tuple[str, str] | None = None) -> tuple[str, str]:
    en_label, ne_label = KIND_LABEL[kind]
    if own and own[0]:
        return own[0], own[1] or own[0]
    if near:
        return f"{en_label} near {near[0]} ({code})", f"{near[1] or near[0]} नजिकको {ne_label} ({code})"
    return f"{en_label} {code}", f"{ne_label} {code}"


def create_site(session: Session, cfg: AOIConfig, *, kind: str, geom_lonlat: BaseGeometry, source: str,
                attrs: dict, elevation_m: float | None, first_seen=None, own_name: tuple[str, str] | None = None,
                named_places: list | None = None) -> Site:
    code = next_code(session, cfg, kind)
    c = geom_lonlat.centroid
    near = nearest_name(c.x, c.y, named_places or []) if not own_name else None
    name, name_ne = make_names(kind, code, near, own_name)
    site = Site(code=code, aoi_id=cfg.id, kind=kind, name=name[:200], name_ne=name_ne[:200], lon=c.x, lat=c.y,
                elevation_m=elevation_m, source=source, attrs=attrs, first_seen=first_seen, last_seen=first_seen)
    set_geom(site, geom_lonlat)
    session.add(site)
    session.flush()
    log.info("new site %s %s (%s)", site.code, site.name, source)
    return site


def sync_assets(session: Session, cfg: AOIConfig, refresh: bool = False) -> int:
    """Load OSM + curated assets inside the AOI's exposure box into the DB (upsert)."""
    feats = []
    try:
        feats += fetch_assets(cfg.routing_bbox, cfg.id, refresh=refresh)["features"]
    except Exception as e:
        log.error("OSM assets unavailable: %s", e)
    curated = curated_assets_path()
    if curated.exists():
        import json

        feats += json.loads(curated.read_text(encoding="utf-8"))["features"]
    w, s, e, n = cfg.routing_bbox
    existing = {a.external_id: a for a in session.scalars(select(Asset))}
    count = 0
    for f in feats:
        lon, lat = f["geometry"]["coordinates"][:2]
        if not (w <= lon <= e and s <= lat <= n):
            continue
        p = f["properties"]
        ext = p["external_id"]
        a = existing.get(ext)
        if a is None:
            a = Asset(external_id=ext)
            session.add(a)
            existing[ext] = a
        a.kind, a.name, a.name_ne, a.lon, a.lat = p["kind"], (p.get("name") or "")[:200], (p.get("name_ne") or "")[:200], lon, lat
        a.source = ext.split(":", 1)[0]
        a.attrs = {k: v for k, v in p.items() if k not in ("external_id", "kind", "name", "name_ne")}
        count += 1
    session.flush()
    return count


def assets_in_box(session: Session, bbox: tuple[float, float, float, float]) -> list[dict]:
    w, s, e, n = bbox
    rows = session.scalars(select(Asset).where(Asset.lon >= w, Asset.lon <= e, Asset.lat >= s, Asset.lat <= n))
    return [{"id": a.id, "kind": a.kind, "name": a.name, "name_ne": a.name_ne, "lon": a.lon, "lat": a.lat,
             "at_channel": bool((a.attrs or {}).get("at_channel"))} for a in rows]


def compute_site_exposure(session: Session, site: Site, routing_dem: np.ndarray, routing_grid,
                          assets: list[dict]) -> int:
    """Trace the downstream path from the site and store exposed assets with arrival times."""
    cfg = load_risk_config()
    flow = cfg["flow"]
    params = ExposureParams(
        corridor_max_offset_m=flow["corridor_max_offset_m"], height_max_near_m=flow["height_max_near_m"],
        height_min_far_m=flow["height_min_far_m"], height_decay_m_per_km=flow["height_decay_m_per_km"],
        max_distance_km=flow["max_distance_km"])
    lon, lat = _outlet(site)
    path = flow_path_from_lonlat(routing_dem, routing_grid, lon, lat, max_km=params.max_distance_km)
    session.execute(delete(SiteExposure).where(SiteExposure.site_id == site.id))
    if path is None:
        site.flow_path = None
        return 0
    from himsat.geo.geometry import to_geojson

    site.flow_path = to_geojson(path.lonlat_linestring(), precision=5)
    speed = float(flow["wave_speed_m_s"].get(site.kind, 8.0))
    ex = exposed_assets(path, assets, routing_dem, params, speed)
    for e in ex:
        session.add(SiteExposure(site_id=site.id, asset_id=e.asset_id, path_distance_km=e.path_distance_km,
                                 offset_m=e.offset_m, height_above_channel_m=e.height_above_channel_m,
                                 travel_time_min=e.travel_time_min))
    attrs = dict(site.attrs or {})
    attrs["flow_path_km"] = round(path.length_km, 1)
    site.attrs = attrs
    session.flush()
    return len(ex)


def _outlet(site: Site) -> tuple[float, float]:
    """Where a flood would leave the site: the lowest point of its outline (approx. by the centroid
    for lakes, whose outlet is found by the tracer's pit escape)."""
    lowest = (site.attrs or {}).get("lowest_point")
    if lowest:
        return float(lowest[0]), float(lowest[1])
    return site.lon, site.lat


def site_mask(site: Site, grid) -> np.ndarray:
    g = from_geojson(site.geom)
    if g is None:
        return np.zeros(grid.shape, bool)
    m = rasterize([g], grid)
    if not m.any():  # tiny site: burn its centroid neighbourhood
        m = rasterize([Point(site.lon, site.lat).buffer(0.0003)], grid)
    return m


def lowest_point(geom_lonlat: BaseGeometry, dem: np.ndarray, grid) -> tuple[float, float, float] | None:
    m = rasterize([geom_lonlat], grid, all_touched=True)
    if not m.any():
        return None
    z = np.where(m, dem, np.inf)
    r, c = np.unravel_index(np.argmin(z), z.shape)
    x, y = grid.xy(r, c)
    from himsat.geo.grid import to_lonlat

    lon, lat = to_lonlat(float(x), float(y), grid.epsg)
    return float(lon), float(lat), float(dem[r, c])


def geom_utm(site: Site, epsg: int) -> BaseGeometry | None:
    g = from_geojson(site.geom)
    return reproject_geom(g, 4326, epsg) if g is not None else None


def load_geojson_sites(path: Path) -> list[dict]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))["features"]


def glacier_named_places(glacier_features: list[dict]) -> list[tuple[str, str, float, float]]:
    out = []
    for f in glacier_features:
        p = f["properties"]
        if p.get("name"):
            c = shape(f["geometry"]).representative_point()
            out.append((p["name"], p.get("name_ne", ""), c.x, c.y))
    return out
