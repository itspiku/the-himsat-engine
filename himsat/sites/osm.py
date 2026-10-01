"""OpenStreetMap data via the Overpass API: exposed assets and glacier outlines.

Data © OpenStreetMap contributors, ODbL 1.0. Results are cached as GeoJSON under
``data/osm/`` so the system keeps working when Overpass is unreachable. Refresh them with
``himsat assets sync``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
from shapely.geometry import LineString, Polygon, mapping
from shapely.ops import unary_union

from himsat.config import get_settings
from himsat.util import retry

log = logging.getLogger(__name__)

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

ASSET_QUERY = """
[out:json][timeout:240];
(
  node["place"~"^(city|town|village|hamlet|isolated_dwelling|locality)$"]({s},{w},{n},{e});
  nwr["power"~"^(plant|generator)$"]["plant:source"="hydro"]({s},{w},{n},{e});
  nwr["power"="generator"]["generator:source"="hydro"]({s},{w},{n},{e});
  nwr["waterway"~"^(dam|weir)$"]({s},{w},{n},{e});
  nwr["barrier"="border_control"]({s},{w},{n},{e});
  way["bridge"="yes"]["highway"~"^(trunk|primary|secondary|tertiary)$"]({s},{w},{n},{e});
  nwr["amenity"~"^(school|hospital|clinic|police|townhall)$"]({s},{w},{n},{e});
  nwr["healthcare"~"^(hospital|clinic|centre)$"]({s},{w},{n},{e});
  nwr["tourism"~"^(alpine_hut|guest_house|hotel|hostel|camp_site)$"]({s},{w},{n},{e});
);
out center tags;
"""

GLACIER_QUERY = """
[out:json][timeout:240];
(
  way["natural"="glacier"]({s},{w},{n},{e});
  relation["natural"="glacier"]({s},{w},{n},{e});
);
out geom tags;
"""


def _overpass(query: str) -> dict:
    last: Exception | None = None
    for url in OVERPASS_URLS:
        try:
            def _post(url=url):
                r = httpx.post(url, data={"data": query}, timeout=300,
                               headers={"User-Agent": "HimSat-Engine/1.0 (glacier hazard early warning)"})
                r.raise_for_status()
                return r.json()

            return retry(_post, attempts=3, base_delay=5)()
        except Exception as e:  # try the next mirror
            last = e
            log.warning("Overpass %s failed: %s", url, e)
    raise RuntimeError(f"All Overpass endpoints failed: {last}")


def classify_asset(tags: dict) -> str | None:
    place = tags.get("place")
    if place in ("city", "town"):
        return "town"
    if place in ("village", "hamlet", "isolated_dwelling", "locality"):
        return "settlement"
    if tags.get("power") in ("plant", "generator") and "hydro" in (
            tags.get("plant:source", "") + tags.get("generator:source", "")):
        return "hydropower"
    if tags.get("waterway") in ("dam", "weir"):
        return "dam"
    if tags.get("barrier") == "border_control":
        return "border_crossing"
    if tags.get("bridge") == "yes":
        return "bridge"
    amenity = tags.get("amenity")
    if amenity == "school":
        return "school"
    if amenity in ("hospital", "clinic") or tags.get("healthcare"):
        return "health"
    if amenity == "police":
        return "police"
    if amenity == "townhall":
        return "government"
    if tags.get("tourism") in ("alpine_hut", "guest_house", "hotel", "hostel", "camp_site"):
        return "tourism"
    return None


def cache_path(name: str) -> Path:
    return get_settings().data_dir / "osm" / f"{name}.geojson"


def fetch_assets(bbox: tuple[float, float, float, float], name: str, refresh: bool = False) -> dict:
    """Assets in bbox as a GeoJSON FeatureCollection of points (cached)."""
    path = cache_path(f"assets_{name}")
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    w, s, e, n = bbox
    data = _overpass(ASSET_QUERY.format(w=w, s=s, e=e, n=n))
    feats = []
    seen = set()
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        kind = classify_asset(tags)
        if not kind:
            continue
        if el["type"] == "node":
            lon, lat = el["lon"], el["lat"]
        elif "center" in el:
            lon, lat = el["center"]["lon"], el["center"]["lat"]
        else:
            continue
        ext = f"osm:{el['type']}/{el['id']}"
        if ext in seen:
            continue
        seen.add(ext)
        name_en = tags.get("name:en") or tags.get("name") or tags.get("int_name") or ""
        name_ne = tags.get("name:ne") or (tags.get("name") if any("ऀ" <= ch <= "ॿ" for ch in tags.get("name", "")) else "")
        props = {"external_id": ext, "kind": kind, "name": name_en, "name_ne": name_ne,
                 "tags": {k: v for k, v in tags.items() if k in (
                     "place", "power", "plant:output:electricity", "operator", "population", "waterway",
                     "barrier", "amenity", "tourism", "highway", "name", "name:ne", "name:en")}}
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": props})
    fc = {"type": "FeatureCollection", "features": feats,
          "attribution": "© OpenStreetMap contributors, ODbL 1.0"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    log.info("OSM assets %s: %d features", name, len(feats))
    return fc


def fetch_glaciers(bbox: tuple[float, float, float, float], name: str, refresh: bool = False,
                   min_area_km2: float = 0.1) -> dict:
    """Glacier outlines (natural=glacier) as GeoJSON polygons (cached)."""
    path = cache_path(f"glaciers_{name}")
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    w, s, e, n = bbox
    data = _overpass(GLACIER_QUERY.format(w=w, s=s, e=e, n=n))
    feats = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        poly = None
        if el["type"] == "way" and el.get("geometry"):
            pts = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if len(pts) >= 4 and pts[0] == pts[-1]:
                poly = Polygon(pts)
        elif el["type"] == "relation":
            outers = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
                      for m in el.get("members", []) if m.get("role") == "outer" and m.get("geometry")]
            from shapely.ops import polygonize

            polys = list(polygonize(unary_union(outers))) if outers else []
            poly = unary_union(polys) if polys else None
        if poly is None or poly.is_empty:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        # rough area in km² at this latitude
        import math

        lat0 = poly.centroid.y
        area_km2 = poly.area * (111.32 ** 2) * math.cos(math.radians(lat0))
        if area_km2 < min_area_km2:
            continue
        feats.append({"type": "Feature", "geometry": mapping(poly), "properties": {
            "external_id": f"osm:{el['type']}/{el['id']}", "name": tags.get("name:en") or tags.get("name") or "",
            "name_ne": tags.get("name:ne", ""), "area_km2": round(area_km2, 3)}})
    fc = {"type": "FeatureCollection", "features": feats, "attribution": "© OpenStreetMap contributors, ODbL 1.0"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    log.info("OSM glaciers %s: %d polygons", name, len(feats))
    return fc
