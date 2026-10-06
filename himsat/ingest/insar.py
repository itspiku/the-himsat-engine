"""Interferometric SAR (InSAR) deformation via ASF HyP3 on-demand processing (optional).

Sentinel-1 amplitude offset tracking resolves a few cm/day. Many slope-failure precursors are
slower: the published precursor of the 26 August 2026 Lhende Khola collapse was ~10 mm/month.
Interferometry measures line-of-sight (LOS) motion at the mm level. Running it means
processing SLC data, which HyP3 does for free in the cloud (it needs a NASA Earthdata login).

Workflow:
1. ``find_pairs``: Sentinel-1 IW SLC scenes over the AOI; consecutive same-orbit pairs (≤ 24 days).
2. ``InsarClient.submit``: one INSAR_GAMMA job per new pair, grouped under a HyP3 project name.
3. ``InsarClient.collect``: download finished products (LOS displacement + coherence GeoTIFFs).
4. ``ingest_product``: reference each interferogram to stable terrain, then record per-site LOS
   velocity (m/day) with standard errors as ``insar_los`` observations. The risk model treats them
   like offset-tracking velocities: significance-gated acceleration relative to the site's own history.

LOS sign: positive = towards the satellite. Downslope motion projects onto LOS with either sign
depending on aspect and look direction, so anomalies use the LOS *speed* |v|.
"""

from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from rasterio.enums import Resampling
from sqlalchemy.orm import Session

from himsat.config import AOIConfig, Settings, get_settings

log = logging.getLogger(__name__)
_DATES = re.compile(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_")


@dataclass(frozen=True)
class SlcScene:
    name: str
    start: datetime
    relative_orbit: int
    flight_direction: str
    frame: int | None = None


def search_slc(aoi: AOIConfig, start: datetime, end: datetime) -> list[SlcScene]:
    import asf_search as asf
    from shapely.geometry import box

    res = asf.search(platform=asf.PLATFORM.SENTINEL1, processingLevel=asf.PRODUCT_TYPE.SLC, beamMode="IW",
                     intersectsWith=box(*aoi.bbox).wkt, start=start, end=end)
    out = []
    for r in res:
        p = r.properties
        out.append(SlcScene(p["sceneName"], datetime.fromisoformat(p["startTime"].replace("Z", "+00:00")),
                            int(p["pathNumber"]), p.get("flightDirection", ""), p.get("frameNumber")))
    return sorted(out, key=lambda s: s.start)


def find_pairs(scenes: list[SlcScene], max_days: int = 24) -> list[tuple[SlcScene, SlcScene]]:
    """Consecutive pairs on the same relative orbit and frame (shortest temporal baseline = best coherence)."""
    by_track: dict[tuple, list[SlcScene]] = {}
    for s in scenes:
        by_track.setdefault((s.relative_orbit, s.frame), []).append(s)
    pairs = []
    for track in by_track.values():
        track.sort(key=lambda s: s.start)
        for a, b in zip(track[:-1], track[1:], strict=True):
            if timedelta(days=5) <= b.start - a.start <= timedelta(days=max_days):
                pairs.append((a, b))
    return sorted(pairs, key=lambda p: p[1].start)


class InsarClient:
    """Thin wrapper around ``hyp3_sdk`` (credentials from HIMSAT_EARTHDATA_USERNAME/PASSWORD)."""

    def __init__(self, settings: Settings | None = None, hyp3=None):
        s = settings or get_settings()
        if hyp3 is None:
            from hyp3_sdk import HyP3

            if not (s.earthdata_username and s.earthdata_password):
                raise RuntimeError("InSAR needs HIMSAT_EARTHDATA_USERNAME and HIMSAT_EARTHDATA_PASSWORD")
            hyp3 = HyP3(username=s.earthdata_username, password=s.earthdata_password)
        self.hyp3 = hyp3
        self.looks = s.insar_looks

    def submitted_pairs(self, project: str) -> set[tuple[str, str]]:
        return {tuple(j.job_parameters["granules"][:2]) for j in self.hyp3.find_jobs(name=project)
                if j.job_parameters and "granules" in j.job_parameters}

    def submit(self, pairs: list[tuple[SlcScene, SlcScene]], project: str, max_jobs: int = 50) -> int:
        done = self.submitted_pairs(project)
        n = 0
        for a, b in pairs:
            if (a.name, b.name) in done or n >= max_jobs:
                continue
            self.hyp3.submit_insar_job(a.name, b.name, name=project, looks=self.looks,
                                       include_displacement_maps=True, apply_water_mask=False)
            n += 1
        log.info("InSAR: submitted %d new jobs to HyP3 project %s", n, project)
        return n

    def collect(self, project: str, out_dir: Path) -> list[Path]:
        """Download succeeded, not-yet-downloaded products; return product directories."""
        out_dir.mkdir(parents=True, exist_ok=True)
        batch = self.hyp3.find_jobs(name=project).filter_jobs(succeeded=True, running=False, failed=False)
        products = []
        for job in batch:
            stem = (job.files or [{}])[0].get("filename", "").removesuffix(".zip")
            target = out_dir / stem
            if stem and not target.exists():
                for z in job.download_files(out_dir):
                    with zipfile.ZipFile(z) as zf:
                        zf.extractall(out_dir)
                    Path(z).unlink(missing_ok=True)
            if target.exists():
                products.append(target)
        return products


@dataclass
class InsarProduct:
    path: Path
    ref_time: datetime
    sec_time: datetime
    los: Path
    corr: Path | None

    @property
    def dt_days(self) -> float:
        return (self.sec_time - self.ref_time).total_seconds() / 86400.0

    @property
    def key(self) -> str:
        return f"INSAR_{self.path.name}"


def read_product(path: Path) -> InsarProduct | None:
    m = _DATES.search(path.name + "_")
    los = next(iter(sorted(path.glob("*_los_disp.tif"))), None)
    if not m or los is None:
        return None
    t0, t1 = (datetime.strptime(x, "%Y%m%dT%H%M%S").replace(tzinfo=UTC) for x in m.groups())
    corr = next(iter(sorted(path.glob("*_corr.tif"))), None)
    return InsarProduct(path, min(t0, t1), max(t0, t1), los, corr)


def los_velocity_stats(los_m: np.ndarray, corr: np.ndarray | None, region: np.ndarray, stable: np.ndarray,
                       dt_days: float, min_corr: float = 0.35, min_px: int = 30) -> dict | None:
    """LOS velocity of ``region`` relative to stable terrain, with a standard error.

    Interferograms are relative measurements: the stable-terrain median is the zero reference, and
    the spread of stable pixels (robust SD) sets the noise. ~40 m pixels are treated as independent
    after the 20×4 multilooking HyP3 applies, so the error is scaled conservatively by √(n/4).
    """
    ok = np.isfinite(los_m)
    if corr is not None:
        ok &= np.nan_to_num(corr, nan=0.0) >= min_corr
    st, rg = ok & stable, ok & region
    if st.sum() < min_px or rg.sum() < max(5, min_px // 3):
        return None
    ref = float(np.median(los_m[st]))
    noise = 1.4826 * float(np.median(np.abs(los_m[st] - ref)))
    vals = los_m[rg] - ref
    v = float(np.median(vals)) / dt_days
    se = 1.2533 * noise / np.sqrt(max(rg.sum() / 4.0, 1.0)) / dt_days
    return {"v_los_m_day": v, "speed_los_m_day": abs(v), "v_los_se": max(se, 1e-5),
            "coverage": float(rg.sum() / max(region.sum(), 1)), "n_points": int(rg.sum()),
            "noise_mm": noise * 1000, "dt_days": dt_days}


def ingest_product(session: Session, ctx, product: InsarProduct, touched: set[int]) -> int:
    """Record per-site LOS velocities from one interferogram (sites = glaciers and slopes)."""
    from himsat.ingest.loader import read_to_grid
    from himsat.pipeline.s2 import _tile_box, _upsert_obs
    from himsat.sites import inventory as inv

    n = 0
    for tile in ctx.tiles:
        sites = [s for s in inv.sites_near(session, ctx.cfg.id, _tile_box(tile), ("glacier", "slope"))]
        if not sites:
            continue
        los = read_to_grid(str(product.los), tile.grid, resampling=Resampling.bilinear, cache=False)
        los[los == 0] = np.nan  # HyP3 writes 0 outside the processed footprint
        if not np.isfinite(los).any():
            continue
        corr = (read_to_grid(str(product.corr), tile.grid, resampling=Resampling.bilinear, cache=False)
                if product.corr else None)
        st = ctx.tile_static(tile)
        for s in sites:
            r, c = tile.grid.lonlat_rowcol(s.lon, s.lat)
            if not tile.in_core(r, c):
                continue
            region = inv.site_mask(s, tile.grid) & st.watch
            stats = los_velocity_stats(los, corr, region, st.stable, product.dt_days)
            if stats is None:
                continue
            _upsert_obs(session, s.id, "insar_los", "S1", product.key, product.sec_time, stats,
                        quality=min(1.0, stats["coverage"]), method="hyp3-insar-gamma",
                        reference_at=product.ref_time)
            touched.add(s.id)
            n += 1
    return n


def project_name(aoi_id: str) -> str:
    return f"himsat-{aoi_id}"
