"""End-to-end monitoring cycle on a synthetic AOI (no network).

Catalogue, imagery, DEM and OSM are replaced by synthetic stand-ins; everything else — tiling,
segmentation, lake extraction, site discovery, S1 velocity, exposure routing, risk, alert
composition and policy — is the production code path.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from shapely.geometry import box, mapping
from sqlalchemy import select

from himsat.config import AOIConfig, Settings
from himsat.db.models import Alert, Observation, RiskAssessment, Site, SiteExposure
from himsat.geo.grid import Grid
from himsat.ingest.loader import S1Scene, S2Scene
from himsat.ingest.stac import Acquisition, SceneItem

BBOX = (85.50, 28.30, 85.56, 28.35)
LAKE_CENTRE = (85.53, 28.335)  # lon, lat
VILLAGE = (85.53, 28.305)
T0 = datetime(2026, 6, 1, 5, 0, tzinfo=UTC)


def _dem(grid: Grid) -> np.ndarray:
    """Valley draining south, a flat lake basin at 4500 m, steep glacier slope above it."""
    rows, cols = np.indices(grid.shape)
    x, y = grid.xy(rows, cols)
    from himsat.geo.grid import to_lonlat

    lon, lat = to_lonlat(np.asarray(x), np.asarray(y), grid.epsg)
    dem = 2500 + (lat - 28.30) * 60000 + np.abs(lon - 85.53) * 40000  # ~ +600 m per 1 km north
    basin = np.hypot((lon - LAKE_CENTRE[0]) * 98000, (lat - LAKE_CENTRE[1]) * 111000) < 400
    dem = np.where(basin, 4500.0, dem)
    return dem.astype("float32")


def _lake_mask(grid: Grid, radius_m: float) -> np.ndarray:
    rows, cols = np.indices(grid.shape)
    x, y = grid.xy(rows, cols)
    from himsat.geo.grid import from_lonlat

    cx, cy = from_lonlat(*LAKE_CENTRE, grid.epsg)
    return np.hypot(np.asarray(x) - cx, np.asarray(y) - cy) < radius_m


def _s2(acq: Acquisition, grid: Grid, bands=(), catalog=None, min_valid_fraction=0.0) -> S2Scene:
    days = (acq.datetime - T0).days
    radius = 250 if days < 20 else 300  # area grows ~44 %
    rng = np.random.default_rng(days)
    b = {k: np.full(grid.shape, v, "float32") + rng.normal(0, 0.004, grid.shape).astype("float32")
         for k, v in {"B02": 0.09, "B03": 0.11, "B04": 0.13, "B08": 0.26, "B8A": 0.27, "B11": 0.24,
                      "B12": 0.18}.items()}
    lake = _lake_mask(grid, radius)
    for k, v in {"B02": 0.10, "B03": 0.12, "B04": 0.08, "B08": 0.03, "B8A": 0.03, "B11": 0.02, "B12": 0.01}.items():
        b[k][lake] = v
    scl = np.full(grid.shape, 5, np.uint8)
    return S2Scene(acq.datetime, acq.key, b, scl, 150.0, 70.0)


def _s1(acq: Acquisition, grid: Grid, catalog=None, with_vh=True) -> S1Scene:
    rng = np.random.default_rng(42)
    from scipy import ndimage

    base = ndimage.gaussian_filter(rng.normal(-10, 2, grid.shape), 1.5).astype("float32")
    noise = np.random.default_rng(int(acq.datetime.timestamp())).normal(0, 0.3, grid.shape).astype("float32")
    vv = base + noise
    vv[_lake_mask(grid, 280)] = -23.0
    return S1Scene(acq.datetime, acq.key, acq.relative_orbit, "descending", vv, None)


def _acq(sensor: str, when: datetime, orbit: int) -> Acquisition:
    it = SceneItem(id=f"{sensor}-{when:%Y%m%d}", collection="x", sensor=sensor, datetime=when, platform=f"{sensor}A",
                   assets={}, relative_orbit=orbit, bbox=BBOX, cloud_cover=1.0, sun_azimuth=150.0, sun_elevation=70.0)
    key = f"{sensor}_{sensor}A_{when:%Y%m%dT%H%M}_R{orbit:03d}"
    return Acquisition(sensor, key, when, f"{sensor}A", orbit, "descending", [it])


class FakeCatalog:
    def __init__(self, *_, **__):
        self.s2 = [_acq("S2", T0 + timedelta(days=d), 19) for d in (0, 5, 30, 35)]
        self.s1 = [_acq("S1", T0 + timedelta(days=d, hours=1), 19) for d in (2, 14, 26, 38)]

    def search_s2(self, bbox, start=None, end=None, max_cloud=None):
        return [i for a in self.s2 if start <= a.datetime <= end for i in a.items]

    def search_s1(self, bbox, start=None, end=None):
        return [i for a in self.s1 if start <= a.datetime <= end for i in a.items]

    def sign(self, href):
        return href


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    import himsat.pipeline.context as ctxmod
    import himsat.pipeline.monitor as mon
    import himsat.pipeline.s1 as s1mod
    import himsat.pipeline.s2 as s2mod
    import himsat.sites.inventory as invmod

    aoi = AOIConfig(id="synthetic", name="Synthetic valley", name_ne="कृत्रिम उपत्यका", bbox=BBOX,
                    exposure_bbox=(85.49, 28.29, 85.57, 28.36), code_prefix="SYN", tile_size_px=1024,
                    min_glacial_lake_elevation_m=3000)
    settings = Settings(data_dir=tmp_path, database_url=f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}",
                        dispatch_enabled=False, llm_backend="none", cache_enabled=False)
    glacier = box(85.52, 28.340, 85.54, 28.349)
    monkeypatch.setattr(mon, "get_aoi", lambda aoi_id, s=None: aoi)
    monkeypatch.setattr(ctxmod, "Catalog", FakeCatalog)
    monkeypatch.setattr(mon, "AOIContext", lambda cfg, s, products_dir=None: ctxmod.AOIContext(cfg, s, FakeCatalog(), products_dir))
    monkeypatch.setattr(ctxmod, "load_dem", lambda grid, catalog=None: _dem(grid))
    monkeypatch.setattr(ctxmod, "fetch_glaciers", lambda bbox, name, refresh=False: {"features": [
        {"type": "Feature", "geometry": mapping(glacier),
         "properties": {"external_id": "osm:way/1", "name": "Test Glacier", "name_ne": "परीक्षण हिमनदी", "area_km2": 2.0}}]})
    def asset(i, lon, lat, kind, name, name_ne, **kw):
        return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {"external_id": f"osm:node/{i}", "kind": kind, "name": name, "name_ne": name_ne, **kw}}

    monkeypatch.setattr(invmod, "fetch_assets", lambda bbox, name, refresh=False: {"features": [
        asset(7, *VILLAGE, "settlement", "Testgaun", "टेस्टगाउँ"),
        asset(8, 85.53, 28.300, "town", "Testbazar", "टेस्टबजार"),
        asset(9, 85.53, 28.296, "hydropower", "Test Khola Hydropower intake", "टेस्ट खोला जलविद्युत", at_channel=True)]})
    monkeypatch.setattr(s2mod, "load_s2", _s2)
    monkeypatch.setattr(s1mod, "load_s1", _s1)
    return aoi, settings


def test_monitoring_cycle_end_to_end(synthetic):
    from himsat.db.session import session_scope
    from himsat.pipeline.monitor import CycleOptions, run_cycle

    aoi, settings = synthetic
    res = run_cycle(aoi.id, CycleOptions(start=T0 - timedelta(days=1), end=T0 + timedelta(days=40), hindcast=True),
                    db_url=settings.database_url, settings=settings)
    assert res.scenes_failed == 0 and res.scenes_processed == 8
    with session_scope(settings.database_url) as s:
        lakes = s.scalars(select(Site).where(Site.kind == "glacial_lake")).all()
        assert len(lakes) == 1, [x.name for x in lakes]
        lake = lakes[0]
        assert lake.code == "SYN-L0001" and "Test Glacier" in lake.name
        areas = [o.values["area_m2"] for o in s.scalars(select(Observation).where(
            Observation.site_id == lake.id, Observation.kind == "lake_area", Observation.sensor == "S2"))]
        assert len(areas) == 4 and max(areas) / min(areas) > 1.3
        assert s.scalars(select(Observation).where(Observation.site_id == lake.id, Observation.sensor == "S1")).first()
        assert s.scalars(select(Site).where(Site.kind == "glacier")).one().name == "Test Glacier"
        exp = s.scalars(select(SiteExposure).where(SiteExposure.site_id == lake.id)).all()
        assert len(exp) == 3 and min(e.travel_time_min for e in exp) < 15
        last = s.scalars(select(RiskAssessment).where(RiskAssessment.site_id == lake.id)
                         .order_by(RiskAssessment.assessed_at.desc())).first()
        assert "lake_growth_recent_pct" in [r["code"] for r in last.reasons]
        alerts = s.scalars(select(Alert).where(Alert.site_id == lake.id)).all()
        assert alerts and alerts[0].level in ("medium", "high")
        assert "टेस्टगाउँ" in alerts[0].body_ne and "Testgaun" in alerts[0].body_en
        assert alerts[0].status == "pending_review"  # hindcast: never dispatched
