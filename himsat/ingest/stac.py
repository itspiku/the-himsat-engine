"""STAC catalogue access, normalised across providers.

Sentinel-2 L2A, Sentinel-1 RTC and Copernicus DEM are free on Microsoft Planetary Computer
(default). Element84 Earth Search (AWS) is supported for Sentinel-2 and the DEM. Items are
normalised into :class:`SceneItem` with canonical band names, and items from the same satellite
pass are grouped into one :class:`Acquisition` (a pass is usually split into several tiles or frames).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from himsat.config import get_settings
from himsat.util import retry

log = logging.getLogger(__name__)

PROVIDERS = {
    "planetary-computer": "https://planetarycomputer.microsoft.com/api/stac/v1",
    "earth-search": "https://earth-search.aws.element84.com/v1",
}

COLLECTIONS = {
    "planetary-computer": {"S2": "sentinel-2-l2a", "S1": "sentinel-1-rtc", "DEM": "cop-dem-glo-30",
                           "LANDCOVER": "esa-worldcover"},
    "earth-search": {"S2": "sentinel-2-l2a", "DEM": "cop-dem-glo-30"},
}

# provider asset key -> canonical band name
_S2_EARTH_SEARCH = {"blue": "B02", "green": "B03", "red": "B04", "rededge1": "B05", "rededge2": "B06",
                    "rededge3": "B07", "nir": "B08", "nir08": "B8A", "swir16": "B11", "swir22": "B12",
                    "scl": "SCL"}
_S2_BANDS = {"B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "SCL"}


@dataclass
class SceneItem:
    id: str
    collection: str
    sensor: str
    datetime: datetime
    platform: str
    assets: dict[str, str]
    cloud_cover: float | None = None
    relative_orbit: int | None = None
    orbit_state: str | None = None
    epsg: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    # Sentinel-2 digital number -> reflectance: refl = (DN + dn_offset) * dn_scale
    dn_offset: float = 0.0
    dn_scale: float = 1e-4
    sun_azimuth: float | None = None
    sun_elevation: float | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Acquisition:
    sensor: str
    key: str
    datetime: datetime
    platform: str
    relative_orbit: int | None
    orbit_state: str | None
    items: list[SceneItem]

    @property
    def cloud_cover(self) -> float | None:
        cc = [i.cloud_cover for i in self.items if i.cloud_cover is not None]
        return sum(cc) / len(cc) if cc else None

    @property
    def sun_azimuth(self) -> float | None:
        v = [i.sun_azimuth for i in self.items if i.sun_azimuth is not None]
        return sum(v) / len(v) if v else None

    @property
    def sun_elevation(self) -> float | None:
        v = [i.sun_elevation for i in self.items if i.sun_elevation is not None]
        return sum(v) / len(v) if v else None


def _platform_code(p: str) -> str:
    """'sentinel-2c' / 'Sentinel-1A' / 'SENTINEL-1D' -> 'S2C' / 'S1A' / 'S1D'."""
    m = re.search(r"(\d)([a-dA-D])\s*$", p or "")
    return f"S{m.group(1)}{m.group(2).upper()}" if m else (p or "unknown")


def _parse_dt(v: Any) -> datetime:
    dt = v if isinstance(v, datetime) else datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


class Catalog:
    def __init__(self, provider: str | None = None):
        self.provider = provider or get_settings().stac_provider
        if self.provider not in PROVIDERS:
            raise ValueError(f"Unknown STAC provider {self.provider}")
        self._client = None

    # -- client ----------------------------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            import pystac_client

            modifier = None
            if self.provider == "planetary-computer":
                import planetary_computer

                modifier = planetary_computer.sign_inplace
            self._client = retry(lambda: pystac_client.Client.open(PROVIDERS[self.provider], modifier=modifier))()
        return self._client

    def collection(self, kind: str) -> str:
        try:
            return COLLECTIONS[self.provider][kind]
        except KeyError as e:
            raise ValueError(f"Provider {self.provider} has no {kind} collection") from e

    def sign(self, href: str) -> str:
        """Return a readable URL (Planetary Computer needs short-lived SAS tokens)."""
        if self.provider == "planetary-computer" and "blob.core.windows.net" in href:
            import planetary_computer

            return planetary_computer.sign(href.split("?", 1)[0])
        return href

    # -- search ----------------------------------------------------------------------------
    def _search(self, kind: str, bbox, start: datetime | None, end: datetime | None, query=None) -> list:
        coll = self.collection(kind)
        dt = None
        if start or end:
            s = start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if start else ".."
            e = end.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if end else ".."
            dt = f"{s}/{e}"
        kwargs: dict[str, Any] = {"collections": [coll], "bbox": list(bbox),
                                  "max_items": get_settings().stac_max_items}
        if dt:
            kwargs["datetime"] = dt
        if query:
            kwargs["query"] = query

        def _do():
            return list(self.client.search(**kwargs).items())

        items = retry(_do, attempts=get_settings().http_retries)()
        log.info("STAC %s %s: %d items", self.provider, coll, len(items))
        return items

    def search_s2(self, bbox, start=None, end=None, max_cloud: float | None = None) -> list[SceneItem]:
        max_cloud = get_settings().s2_max_scene_cloud if max_cloud is None else max_cloud
        items = self._search("S2", bbox, start, end, query={"eo:cloud_cover": {"lte": max_cloud}})
        return [self._norm_s2(i) for i in items]

    def search_s1(self, bbox, start=None, end=None) -> list[SceneItem]:
        items = self._search("S1", bbox, start, end)
        out = []
        for i in items:
            p = i.properties
            if p.get("sar:instrument_mode", "IW") != "IW":
                continue
            assets = {k.upper(): a.href for k, a in i.assets.items() if k.lower() in ("vv", "vh")}
            if "VV" not in assets:
                continue
            out.append(SceneItem(
                id=i.id, collection=i.collection_id, sensor="S1", datetime=_parse_dt(i.datetime or p["datetime"]),
                platform=_platform_code(p.get("platform", "")), assets=assets,
                relative_orbit=p.get("sat:relative_orbit"), orbit_state=p.get("sat:orbit_state"),
                epsg=p.get("proj:epsg"), bbox=tuple(i.bbox) if i.bbox else None, properties=dict(p),
            ))
        return out

    def search_dem(self, bbox) -> list[str]:
        items = self._search("DEM", bbox, None, None)
        hrefs = []
        for i in items:
            a = i.assets.get("data") or next(iter(i.assets.values()))
            hrefs.append(a.href)
        return hrefs

    def search_landcover(self, bbox) -> list[str]:
        items = self._search("LANDCOVER", bbox, None, None)
        items.sort(key=lambda i: i.datetime or _parse_dt(i.properties.get("start_datetime", "2000-01-01")),
                   reverse=True)
        latest_year = None
        hrefs = []
        for i in items:
            y = (i.datetime or _parse_dt(i.properties.get("start_datetime"))).year
            latest_year = latest_year or y
            if y == latest_year:
                hrefs.append(i.assets["map"].href)
        return hrefs

    # -- normalisation ---------------------------------------------------------------------
    def _norm_s2(self, i) -> SceneItem:
        p = i.properties
        assets: dict[str, str] = {}
        scale, offset = 1e-4, 0.0
        for k, a in i.assets.items():
            band = k.upper() if k.upper() in _S2_BANDS else _S2_EARTH_SEARCH.get(k)
            if not band or band in assets:
                continue
            if self.provider == "earth-search" and not a.href.endswith(".tif"):
                continue
            assets[band] = a.href
            rb = (a.extra_fields or {}).get("raster:bands")
            if band == "B04" and rb and isinstance(rb, list) and "scale" in rb[0]:
                scale = float(rb[0].get("scale", 1e-4))
                offset = float(rb[0].get("offset", 0.0)) / scale  # expressed in DN
        if self.provider == "planetary-computer":
            baseline = p.get("s2:processing_baseline")
            try:
                offset = -1000.0 if baseline and float(baseline) >= 4.0 else 0.0
            except ValueError:
                offset = 0.0
        sun_el = p.get("view:sun_elevation")
        if sun_el is None and p.get("s2:mean_solar_zenith") is not None:
            sun_el = 90.0 - float(p["s2:mean_solar_zenith"])
        sun_az = p.get("view:sun_azimuth", p.get("s2:mean_solar_azimuth"))
        orbit = p.get("sat:relative_orbit") or p.get("s2:relative_orbit")
        if orbit is None:
            m = re.search(r"_R(\d{3})_", i.id)
            orbit = int(m.group(1)) if m else None
        return SceneItem(
            id=i.id, collection=i.collection_id, sensor="S2", datetime=_parse_dt(i.datetime or p["datetime"]),
            platform=_platform_code(p.get("platform", "")), assets=assets, cloud_cover=p.get("eo:cloud_cover"),
            relative_orbit=orbit, epsg=p.get("proj:epsg"), bbox=tuple(i.bbox) if i.bbox else None,
            dn_offset=offset, dn_scale=scale, sun_azimuth=sun_az, sun_elevation=sun_el, properties=dict(p),
        )


def group_acquisitions(items: list[SceneItem]) -> list[Acquisition]:
    """Group items from one satellite pass into acquisitions, sorted by time."""
    groups: dict[tuple, list[SceneItem]] = defaultdict(list)
    for it in items:
        groups[(it.sensor, it.platform, it.datetime.strftime("%Y%m%d"), it.relative_orbit)].append(it)
    acqs = []
    for (sensor, platform, _day, orbit), its in groups.items():
        # the same product can be listed twice (reprocessing): keep the latest id per tile/frame
        uniq: dict[str, SceneItem] = {}
        for it in sorted(its, key=lambda x: x.id):
            tile = re.sub(r"_\d{8}T\d{6}$", "", it.id) if sensor == "S2" else it.id
            uniq[tile] = it
        its = sorted(uniq.values(), key=lambda x: x.datetime)
        t0 = its[0].datetime
        mean_dt = t0 + (its[-1].datetime - t0) / 2
        orbit_s = f"R{int(orbit):03d}" if orbit is not None else "Rxxx"
        key = f"{sensor}_{platform}_{mean_dt.strftime('%Y%m%dT%H%M')}_{orbit_s}"
        acqs.append(Acquisition(sensor, key, mean_dt, platform, orbit, its[0].orbit_state, its))
    return sorted(acqs, key=lambda a: a.datetime)
