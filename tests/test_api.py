import hashlib
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Point

from himsat.config import Settings
from himsat.db.models import AOI, Alert, RiskAssessment, Site
from himsat.db.session import session_scope
from himsat.geo.geometry import set_geom

KEY = "hs_test_key"


@pytest.fixture
def client(db_url, tmp_path):
    from himsat.api.app import create_app

    s = Settings(database_url=db_url, admin_api_keys=[hashlib.sha256(KEY.encode()).hexdigest()],
                 web_dist_dir=tmp_path / "nodist", data_dir=tmp_path, dispatch_enabled=False)
    now = datetime.now(UTC)
    with session_scope(db_url) as db:
        db.add(AOI(id="rasuwa-lhende", name="Lhende"))
        site = Site(code="LHD-S0001", aoi_id="rasuwa-lhende", kind="slope", name="Slope", name_ne="भिर",
                    lon=85.53, lat=28.29, latest_level="high", latest_score=0.7, latest_assessed_at=now)
        set_geom(site, Point(85.53, 28.29).buffer(0.005))
        db.add(site)
        db.flush()
        db.add(RiskAssessment(site_id=site.id, assessed_at=now, level="high", score=0.7, hazard=0.8, exposure=0.9,
                              confidence=0.8, reasons=[{"code": "velocity_ratio", "text_en": "x", "text_ne": "y"}],
                              indicators={"_dynamic": 0.7}))
        for status, title in (("dispatched", "Public"), ("pending_review", "Hidden")):
            db.add(Alert(site_id=site.id, level="high", kind="alert", status=status, issued_at=now,
                         expires_at=now + timedelta(days=1), title_en=title, title_ne="शीर्षक", body_en="b",
                         body_ne="ब", sms_en="s", sms_ne="स", facts={"score": 0.7}))
    return TestClient(create_app(s))


def test_public_endpoints(client):
    h = client.get("/api/health").json()
    assert h["status"] == "degraded" and h["monitoring"]["stale"]  # no monitoring run has finished yet
    fc = client.get("/api/sites").json()
    assert fc["type"] == "FeatureCollection" and fc["features"][0]["properties"]["level"] == "high"
    d = client.get("/api/sites/LHD-S0001").json()
    assert d["risk"]["level"] == "high" and d["geometry"]["type"] == "Polygon"
    assert client.get("/api/sites/NOPE").status_code == 404
    alerts = client.get("/api/alerts").json()
    assert [a["title_en"] for a in alerts] == ["Public"]  # unapproved alerts are never public
    uid = alerts[0]["uid"]
    cap = client.get(f"/api/alerts/{uid}/cap.xml")
    assert cap.headers["content-type"].startswith("application/cap+xml")
    ET.fromstring(cap.content)
    feed = client.get("/api/alerts/feed.xml")
    assert uid in feed.text
    assert client.get("/api/stats").json()["sites_by_level"] == {"high": 1}
    assert client.get("/api/does-not-exist").status_code == 404
    assert "himsat_sites" in client.get("/metrics").text


def test_admin_requires_key(client):
    assert client.get("/api/admin/alerts").status_code == 401
    assert client.get("/api/admin/alerts", headers={"X-API-Key": "wrong"}).status_code == 403
    q = client.get("/api/admin/alerts", headers={"X-API-Key": KEY}).json()
    assert [a["title_en"] for a in q] == ["Hidden"]


def test_admin_approve_and_subscribers(client):
    h = {"X-API-Key": KEY}
    pending = client.get("/api/admin/alerts", headers=h).json()[0]
    r = client.post(f"/api/admin/alerts/{pending['id']}/approve", json={"user": "duty officer"}, headers=h)
    assert r.status_code == 200
    assert len(client.get("/api/alerts").json()) == 2
    assert client.post(f"/api/admin/alerts/{pending['id']}/approve", json={"user": "x2"}, headers=h).status_code == 409
    body = {"name": "Gosaikunda Rural Municipality", "org_type": "municipality", "language": "ne",
            "phone": "+9779800000000", "channels": ["sms"], "min_level": "medium",
            "area": {"type": "Polygon", "coordinates": [[[85.2, 28.0], [85.6, 28.0], [85.6, 28.4], [85.2, 28.4],
                                                         [85.2, 28.0]]]}}
    created = client.post("/api/admin/subscribers", json=body, headers=h)
    assert created.status_code == 201 and created.json()["area"]["type"] == "Polygon"
    bad = dict(body, area={"type": "Point", "coordinates": [85, 28]})
    assert client.post("/api/admin/subscribers", json=bad, headers=h).status_code == 422
    assert client.post("/api/admin/runs", json={"aoi_id": "nope"}, headers=h).status_code == 404


def test_hindcast_endpoints_and_cache_headers(client, tmp_path):
    import json as _json

    d = tmp_path / "hindcasts" / "demo"
    d.mkdir(parents=True)
    (d / "report.json").write_text(_json.dumps({
        "aoi": {"id": "rasuwa-lhende", "name": "Lhende"}, "period": ["2026-08-01", "2026-09-01"],
        "event_time": "2026-08-26T02:52:00+00:00", "alerts": [{"id": 1}]}), encoding="utf-8")
    lst = client.get("/api/hindcasts").json()
    assert lst == [{"name": "demo", "aoi": "rasuwa-lhende", "aoi_name": "Lhende", "period": ["2026-08-01", "2026-09-01"],
                    "event_time": "2026-08-26T02:52:00+00:00", "alerts": 1}]
    assert client.get("/api/hindcasts/demo").json()["aoi"]["id"] == "rasuwa-lhende"
    assert client.get("/api/hindcasts/..%2Fsecret").status_code in (400, 404)
    assert client.get("/api/hindcasts/nope").status_code == 404
    assert client.get("/api/alerts").headers["cache-control"] == "no-cache"
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff"


def test_prune_removes_only_old_files(tmp_path):
    import os
    import time

    from himsat.maintenance import prune

    s = Settings(data_dir=tmp_path)
    old = s.cache_dir / "rasters" / "ab" / "old.tif"
    new = s.cache_dir / "rasters" / "ab" / "new.tif"
    vel = s.products_dir / "aoi" / "velocity" / "0_0" / "x.npz"
    for f in (old, new, vel):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x" * 10)
    t = time.time() - 200 * 86400
    os.utime(old, (t, t))
    os.utime(vel, (t, t))
    res = prune(s, cache_days=120, velocity_days=400)
    assert res["cache_files"] == 1 and res["velocity_files"] == 0
    assert not old.exists() and new.exists() and vel.exists()
