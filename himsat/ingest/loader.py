"""Read cloud-optimised GeoTIFFs onto a target :class:`Grid`.

GDAL range requests fetch only the blocks each tile needs. Results are cached on disk as
GeoTIFFs, keyed by asset and grid, so re-runs and hindcasts don't download again. The cache
files can be opened in QGIS for inspection.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.vrt import WarpedVRT

from himsat.config import get_settings
from himsat.geo.grid import Grid
from himsat.ingest.stac import Acquisition, Catalog
from himsat.util import retry

log = logging.getLogger(__name__)

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff,.TIFF",
    "GDAL_HTTP_MAX_RETRY": 5,
    "GDAL_HTTP_RETRY_DELAY": 2,
    "GDAL_HTTP_TIMEOUT": 120,  # never let a stalled connection hang a monitoring cycle
    "GDAL_HTTP_CONNECTTIMEOUT": 30,
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": 64 * 1024 * 1024,
    "GDAL_CACHEMAX": 512,
    "AWS_NO_SIGN_REQUEST": "YES",
}


def _cache_path(href: str, grid: Grid, resampling: Resampling, tag: str = "") -> Path:
    base = href.split("?", 1)[0]
    h = hashlib.sha1(f"{base}|{grid.key()}|{resampling.name}|{tag}".encode()).hexdigest()
    return get_settings().cache_dir / "rasters" / h[:2] / f"{h}.tif"


def _write_cache(path: Path, arr: np.ndarray, grid: Grid) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.tif")
    profile = {
        "driver": "GTiff", "width": grid.width, "height": grid.height, "count": 1, "dtype": arr.dtype.name,
        "crs": f"EPSG:{grid.epsg}", "transform": grid.transform, "compress": "deflate", "tiled": True,
        "blockxsize": 256, "blockysize": 256,
    }
    if arr.dtype.kind == "f":
        profile["predictor"] = 3
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(arr, 1)
    tmp.replace(path)


def read_to_grid(href: str, grid: Grid, *, resampling: Resampling = Resampling.bilinear,
                 dtype: str = "float32", nodata: float | None = None, catalog: Catalog | None = None,
                 cache: bool | None = None) -> np.ndarray:
    """Warp one raster asset onto ``grid``. Pixels outside the source are NaN (float) or ``nodata``."""
    cache = get_settings().cache_enabled if cache is None else cache
    cpath = _cache_path(href, grid, resampling)
    if cache and cpath.exists():
        with rasterio.open(cpath) as src:
            return src.read(1)

    fill = np.nan if np.dtype(dtype).kind == "f" else (nodata if nodata is not None else 0)

    def _read() -> np.ndarray:
        # sign on every attempt: Planetary Computer SAS tokens are short-lived, and a retry must
        # not reuse a URL whose signature expired during a long run
        url = catalog.sign(href) if catalog else href
        with rasterio.Env(**GDAL_ENV), rasterio.open(url) as src:
            src_nodata = src.nodata if nodata is None else nodata
            with WarpedVRT(src, crs=f"EPSG:{grid.epsg}", transform=grid.transform, width=grid.width,
                           height=grid.height, resampling=resampling, src_nodata=src_nodata,
                           nodata=src_nodata if src_nodata is not None else 0) as vrt:
                data = vrt.read(1, masked=True)
        out = data.astype(dtype).filled(fill)
        return out

    arr = retry(_read, attempts=get_settings().http_retries, exceptions=(RasterioIOError, OSError))()
    if cache:
        _write_cache(cpath, arr, grid)
    return arr


def mosaic(arrays: list[np.ndarray]) -> np.ndarray:
    """First-valid-wins mosaic of float arrays (NaN = no data)."""
    out = arrays[0].copy()
    for a in arrays[1:]:
        m = np.isnan(out) & ~np.isnan(a)
        out[m] = a[m]
    return out


def _item_covers(item_bbox, grid: Grid) -> bool:
    if item_bbox is None:
        return True
    w, s, e, n = grid.lonlat_bbox()
    return not (item_bbox[2] < w or item_bbox[0] > e or item_bbox[3] < s or item_bbox[1] > n)


@dataclass
class S2Scene:
    datetime: datetime
    key: str
    bands: dict[str, np.ndarray]  # reflectance (0..~1.5), NaN = no data
    scl: np.ndarray  # uint8 scene classification, 0 = no data
    sun_azimuth: float | None
    sun_elevation: float | None


S2_DEFAULT_BANDS = ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")

_POOL: ThreadPoolExecutor | None = None


def io_pool() -> ThreadPoolExecutor:
    """Shared pool for concurrent COG reads: remote reads are latency-bound, and GDAL releases the GIL."""
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=get_settings().io_workers, thread_name_prefix="cog")
    return _POOL


def load_s2(acq: Acquisition, grid: Grid, bands: tuple[str, ...] = S2_DEFAULT_BANDS,
            catalog: Catalog | None = None, min_valid_fraction: float = 0.0) -> S2Scene | None:
    """Load an S2 acquisition as reflectance. Returns None if too little of the grid is covered/clear."""
    catalog = catalog or Catalog()
    items = [i for i in acq.items if _item_covers(i.bbox, grid)]
    if not items:
        return None
    # read SCL first (single 20 m band) to skip cloudy tiles cheaply
    scl_parts = [read_to_grid(i.assets["SCL"], grid, resampling=Resampling.nearest, dtype="float32",
                              nodata=0, catalog=catalog) for i in items if "SCL" in i.assets]
    scl_f = mosaic(scl_parts) if scl_parts else np.full(grid.shape, np.nan, "float32")
    scl = np.nan_to_num(scl_f, nan=0).astype(np.uint8)
    clear = np.isin(scl, (2, 4, 5, 6, 7, 11))
    if clear.mean() < min_valid_fraction:
        log.info("S2 %s: %.1f%% clear < %.1f%%, skipped", acq.key, 100 * clear.mean(), 100 * min_valid_fraction)
        return None
    jobs = [(b, i) for b in bands for i in items if b in i.assets]
    futures = [io_pool().submit(read_to_grid, i.assets[b], grid, resampling=Resampling.bilinear, nodata=0,
                                catalog=catalog) for b, i in jobs]
    parts: dict[str, list[np.ndarray]] = {}
    for (b, i), fut in zip(jobs, futures, strict=True):
        parts.setdefault(b, []).append(((fut.result() + i.dn_offset) * i.dn_scale).astype("float32"))
    out = {b: mosaic(ps) for b, ps in parts.items()}
    return S2Scene(acq.datetime, acq.key, out, scl, acq.sun_azimuth, acq.sun_elevation)


@dataclass
class S1Scene:
    datetime: datetime
    key: str
    relative_orbit: int | None
    orbit_state: str | None
    vv_db: np.ndarray  # gamma0 (terrain flattened) in dB, NaN = no data / invalid
    vh_db: np.ndarray | None


def _to_db(lin: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(lin)
    db[~np.isfinite(db)] = np.nan
    return db.astype("float32")


def load_s1(acq: Acquisition, grid: Grid, catalog: Catalog | None = None, with_vh: bool = True) -> S1Scene | None:
    catalog = catalog or Catalog()
    items = [i for i in acq.items if _item_covers(i.bbox, grid)]
    if not items:
        return None
    pols = ["VV"] + (["VH"] if with_vh and all("VH" in i.assets for i in items) else [])
    futs = {pol: [io_pool().submit(read_to_grid, i.assets[pol], grid, resampling=Resampling.bilinear, catalog=catalog)
                  for i in items] for pol in pols}
    vv = mosaic([f.result() for f in futs["VV"]])
    if np.isnan(vv).all():
        return None
    vh = _to_db(mosaic([f.result() for f in futs["VH"]])) if "VH" in futs else None
    return S1Scene(acq.datetime, acq.key, acq.relative_orbit, acq.orbit_state, _to_db(vv), vh)
