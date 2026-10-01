"""Lake-area benchmark against independently published values for well-studied Nepal lakes.

The published areas are literature values (field and very-high-resolution surveys), rounded,
and from slightly different years. Lakes change, so treat differences below ~5 % as noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from himsat.detect.lakes import extract_lakes
from himsat.detect.segmentation import SpectralSegmenter, TerrainContext
from himsat.geo.grid import Grid
from himsat.ingest.dem import cast_shadow, hillshade, load_dem, slope_aspect, upsample
from himsat.ingest.loader import S2_DEFAULT_BANDS, load_s2
from himsat.ingest.stac import Catalog, group_acquisitions


@dataclass
class Lake:
    name: str
    lon: float
    lat: float
    published_km2: float
    source: str


LAKES = [
    Lake("Gosainkunda", 85.4145, 28.0826, 0.138, "DoHM / ICIMOD surveys (~13.8 ha)"),
    Lake("Tsho Rolpa", 86.475, 27.863, 1.55, "ICIMOD 2020; bathymetry surveys 2017-2019 (~1.5-1.6 km²)"),
    Lake("Imja Tsho", 86.925, 27.900, 1.30, "post-2016 lowering surveys (~1.3 km²)"),
    Lake("Thulagi (Dona)", 84.485, 28.490, 0.95, "ICIMOD 2020 (~0.9-1.0 km²)"),
]


def _terrain(grid: Grid, catalog: Catalog, acq) -> TerrainContext:
    dem = load_dem(grid, catalog)
    slope, _ = slope_aspect(dem, grid.res)
    d30 = dem[::3, ::3]
    sh = cast_shadow(d30, 30, acq.sun_azimuth, acq.sun_elevation) | (
        hillshade(d30, 30, acq.sun_azimuth, acq.sun_elevation) < 0.05)
    return TerrainContext(dem, slope, upsample(sh.astype("float32"), grid.shape, 0) > 0.5)


def run(segmenters: dict, start=datetime(2025, 10, 10, tzinfo=UTC), end=datetime(2025, 12, 10, tzinfo=UTC),
        half_deg: float = 0.03, progress=print) -> list[dict]:
    cat = Catalog()
    rows = []
    for lk in LAKES:
        bbox = (lk.lon - half_deg, lk.lat - half_deg, lk.lon + half_deg, lk.lat + half_deg)
        grid = Grid.from_lonlat_bbox(bbox, 10)
        acqs = sorted(group_acquisitions(cat.search_s2(bbox, start, end, max_cloud=10)),
                      key=lambda a: a.cloud_cover or 0)
        scene, acq = None, None
        for a in acqs[:4]:
            scene = load_s2(a, grid, bands=S2_DEFAULT_BANDS, catalog=cat, min_valid_fraction=0.9)
            if scene is not None:
                acq = a
                break
        if scene is None:
            progress(f"{lk.name}: no clear scene")
            continue
        terrain = _terrain(grid, cat, acq)
        row = {"lake": lk.name, "published_km2": lk.published_km2, "scene": acq.key}
        for name, seg in segmenters.items():
            s = seg.segment(scene, terrain)
            lakes = extract_lakes(s, grid, terrain.dem, terrain.slope, min_area_m2=5000, min_elevation_m=0)
            best = min(lakes, key=lambda x: (x.lon - lk.lon) ** 2 + (x.lat - lk.lat) ** 2, default=None)
            area = best.area_m2 / 1e6 if best and abs(best.lon - lk.lon) < 0.01 and abs(best.lat - lk.lat) < 0.01 else 0.0
            row[name] = round(area, 4)
            row[f"{name}_err_pct"] = round(100 * (area - lk.published_km2) / lk.published_km2, 1)
        progress(str(row))
        rows.append(row)
    return rows


def default_segmenters(checkpoint=None, sam_model: str | None = None, device: str = "cpu") -> dict:
    segs: dict = {"rules": SpectralSegmenter()}
    if checkpoint:
        from himsat.ml.prithvi import PrithviSegmenter

        segs["prithvi"] = PrithviSegmenter.from_checkpoint(checkpoint, device)
    if sam_model:
        from himsat.ml.sam import SamRefiner

        segs["rules+sam"] = SamRefiner(SpectralSegmenter(), sam_model, device)
        if checkpoint:
            segs["prithvi+sam"] = SamRefiner(segs["prithvi"], sam_model, device)
    return segs
