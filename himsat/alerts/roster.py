"""Bulk subscriber management from a reviewed YAML roster (idempotent by name).

```yaml
subscribers:
  - name: Gosaikunda Rural Municipality (disaster focal point)
    org_type: municipality
    language: ne
    phone: "+97798XXXXXXXX"
    channels: [sms]
    min_level: medium
    aoi_ids: [rasuwa-trishuli, rasuwa-lhende]
    bbox: [85.25, 28.05, 85.65, 28.35]      # or: area: path/to/polygon.geojson
```
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from shapely.geometry import box, shape
from sqlalchemy import select
from sqlalchemy.orm import Session

from himsat.api.schemas import SubscriberIn
from himsat.db.models import Subscriber
from himsat.geo.geometry import set_geom


def _area(entry: dict, base: Path):
    if entry.get("bbox"):
        return box(*entry["bbox"])
    if entry.get("area"):
        gj = json.loads((base / entry["area"]).read_text(encoding="utf-8"))
        if gj.get("type") == "FeatureCollection":
            gj = gj["features"][0]["geometry"]
        elif gj.get("type") == "Feature":
            gj = gj["geometry"]
        g = shape(gj)
        if g.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"{entry['name']}: area must be a polygon")
        return g
    return None


def import_roster(session: Session, path: Path, deactivate_missing: bool = False) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("subscribers", [])
    existing = {s.name: s for s in session.scalars(select(Subscriber))}
    stats = {"created": 0, "updated": 0, "deactivated": 0}
    seen = set()
    for raw in entries:
        geom = _area(raw, path.parent)
        fields = {k: v for k, v in raw.items() if k not in ("bbox", "area")}
        body = SubscriberIn(**fields)  # validates channels, levels, languages
        missing = [ch for ch, addr in (("sms", body.phone), ("email", body.email), ("webhook", body.webhook_url))
                   if ch in body.channels and not addr]
        if missing:
            raise ValueError(f"{body.name}: no contact address for channel(s) {', '.join(missing)}")
        sub = existing.get(body.name)
        if sub is None:
            sub = Subscriber(name=body.name)
            session.add(sub)
            stats["created"] += 1
        else:
            stats["updated"] += 1
        for f in ("org_type", "language", "phone", "email", "webhook_url", "min_level", "active"):
            setattr(sub, f, getattr(body, f))
        sub.channels, sub.aoi_ids, sub.site_ids = list(body.channels), list(body.aoi_ids), list(body.site_ids)
        set_geom(sub, geom)
        seen.add(body.name)
    if deactivate_missing:
        for name, sub in existing.items():
            if name not in seen and sub.active:
                sub.active = False
                stats["deactivated"] += 1
    session.flush()
    return stats
