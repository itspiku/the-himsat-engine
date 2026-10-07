"""Public, read-only endpoints (map, alerts, CAP feeds, hindcast reports)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from xml.sax.saxutils import escape

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from himsat import __version__
from himsat.alerts.cap import alert_to_cap
from himsat.alerts.service import site_exposures
from himsat.api.deps import get_db, get_settings_dep
from himsat.api.schemas import AlertOut
from himsat.config import Settings, load_aois
from himsat.db.models import Alert, ChangeEvent, Observation, PipelineRun, RiskAssessment, Site

router = APIRouter(prefix="/api", tags=["public"])
PUBLIC_ALERT_STATUSES = ("approved", "dispatched")
LEVEL_RANK = {"unknown": -1, "low": 0, "medium": 1, "high": 2}


def _feature(geom: str | None, props: dict, fallback: tuple[float, float] | None = None) -> dict:
    g = json.loads(geom) if geom else ({"type": "Point", "coordinates": list(fallback)} if fallback else None)
    return {"type": "Feature", "geometry": g, "properties": props}


def _site_props(s: Site) -> dict:
    a = s.attrs or {}
    return {
        "code": s.code, "kind": s.kind, "name": s.name, "name_ne": s.name_ne, "lon": s.lon, "lat": s.lat,
        "elevation_m": s.elevation_m, "source": s.source, "level": s.latest_level, "score": s.latest_score,
        "assessed_at": s.latest_assessed_at.isoformat() if s.latest_assessed_at else None, "aoi": s.aoi_id,
        "area_m2": a.get("area_m2"), "exposed_assets": a.get("exposed_assets"),
    }


@router.get("/health")
def health(db: Session = Depends(get_db), settings: Settings = Depends(get_settings_dep)) -> dict:
    """Liveness plus data freshness: 'degraded' when monitoring has not finished a run recently."""
    db.execute(select(1))
    now = datetime.now(UTC)
    last = db.scalar(select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(1))
    last_ok = db.scalar(select(func.max(PipelineRun.finished_at)).where(PipelineRun.kind.in_(("monitor", "backfill")),
                                                                        PipelineRun.status.in_(("ok", "partial"))))
    max_age = timedelta(minutes=2 * settings.schedule_interval_minutes)
    stale = last_ok is None or now - last_ok > max_age
    return {"status": "degraded" if stale else "ok", "version": __version__, "time": now.isoformat(),
            "monitoring": {"last_finished": last_ok.isoformat() if last_ok else None, "stale": stale,
                           "max_age_minutes": int(max_age.total_seconds() // 60)},
            "last_run": {"aoi": last.aoi_id, "status": last.status, "started_at": last.started_at,
                         "finished_at": last.finished_at} if last else None}


@router.get("/aois")
def aois() -> list[dict]:
    return [{"id": a.id, "name": a.name, "name_ne": a.name_ne, "bbox": a.bbox} for a in load_aois().values()]


@router.get("/sites")
def sites(aoi: str | None = None, kind: str | None = None, min_level: str = "low",
          include_low: bool = True, db: Session = Depends(get_db)) -> dict:
    """All active sites as GeoJSON (outlines; points for tiny sites) with their latest risk."""
    q = select(Site).where(Site.status == "active")
    if aoi:
        q = q.where(Site.aoi_id == aoi)
    if kind:
        q = q.where(Site.kind.in_(kind.split(",")))
    rank = LEVEL_RANK.get(min_level, 0)
    feats = [_feature(s.geom, _site_props(s), (s.lon, s.lat)) for s in db.scalars(q)
             if LEVEL_RANK.get(s.latest_level, -1) >= rank or (include_low and rank <= 0)]
    return {"type": "FeatureCollection", "features": feats}


def _get_site(db: Session, code: str) -> Site:
    s = db.scalar(select(Site).where(Site.code == code))
    if s is None:
        raise HTTPException(404, f"site {code} not found")
    return s


@router.get("/sites/{code}")
def site_detail(code: str, db: Session = Depends(get_db)) -> dict:
    s = _get_site(db, code)
    ra = db.scalar(select(RiskAssessment).where(RiskAssessment.site_id == s.id)
                   .order_by(RiskAssessment.assessed_at.desc()).limit(1))
    alerts = db.scalars(select(Alert).where(Alert.site_id == s.id, Alert.status.in_(PUBLIC_ALERT_STATUSES))
                        .order_by(Alert.issued_at.desc()).limit(10)).all()
    return {
        **_site_props(s),
        "geometry": json.loads(s.geom) if s.geom else None,
        "flow_path": json.loads(s.flow_path) if s.flow_path else None,
        "attrs": {k: v for k, v in (s.attrs or {}).items() if k not in ("cells",)},
        "first_seen": s.first_seen, "last_seen": s.last_seen,
        "risk": {"level": ra.level, "score": ra.score, "hazard": ra.hazard, "exposure": ra.exposure,
                 "confidence": ra.confidence, "assessed_at": ra.assessed_at, "reasons": ra.reasons,
                 "model_version": ra.model_version,
                 "potentially_dangerous_only": not (ra.indicators or {}).get("_dynamic", 0)} if ra else None,
        "exposed": site_exposures(db, s.id),
        "alerts": [{"uid": a.uid, "level": a.level, "kind": a.kind, "issued_at": a.issued_at, "title_en": a.title_en,
                    "title_ne": a.title_ne} for a in alerts],
    }


@router.get("/sites/{code}/observations")
def site_observations(code: str, kind: str | None = None, days: int = Query(400, le=3650),
                      db: Session = Depends(get_db)) -> list[dict]:
    s = _get_site(db, code)
    q = select(Observation).where(Observation.site_id == s.id,
                                  Observation.observed_at >= datetime.now(UTC) - timedelta(days=days))
    if kind:
        q = q.where(Observation.kind == kind)
    return [{"at": o.observed_at, "ref_at": o.reference_at, "kind": o.kind, "sensor": o.sensor, "quality": o.quality,
             "method": o.method, "values": o.values}
            for o in db.scalars(q.order_by(Observation.observed_at))]


@router.get("/sites/{code}/assessments")
def site_assessments(code: str, limit: int = Query(200, le=2000), db: Session = Depends(get_db)) -> list[dict]:
    s = _get_site(db, code)
    rows = db.scalars(select(RiskAssessment).where(RiskAssessment.site_id == s.id)
                      .order_by(RiskAssessment.assessed_at.desc()).limit(limit)).all()
    return [{"at": r.assessed_at, "level": r.level, "score": r.score, "hazard": r.hazard, "exposure": r.exposure,
             "confidence": r.confidence, "reasons": [x["code"] for x in r.reasons]} for r in reversed(rows)]


@router.get("/events")
def events(aoi: str | None = None, kind: str | None = None, days: int = Query(60, le=730),
           min_area_m2: float = 50_000, db: Session = Depends(get_db)) -> dict:
    q = select(ChangeEvent).where(ChangeEvent.detected_at >= datetime.now(UTC) - timedelta(days=days),
                                  ChangeEvent.area_m2 >= min_area_m2)
    if aoi:
        q = q.where(ChangeEvent.aoi_id == aoi)
    if kind:
        q = q.where(ChangeEvent.kind.in_(kind.split(",")))
    feats = [_feature(e.geom, {"kind": e.kind, "sensor": e.sensor, "detected_at": e.detected_at.isoformat(),
                               "area_m2": e.area_m2, "confidence": e.confidence, "site_id": e.site_id},
                      (e.lon, e.lat)) for e in db.scalars(q.limit(5000))]
    return {"type": "FeatureCollection", "features": feats}


def _alert_out(a: Alert, s: Site) -> AlertOut:
    return AlertOut(uid=a.uid, id=a.id, site_code=s.code, site_name=s.name, site_name_ne=s.name_ne,
                    site_kind=s.kind, lon=s.lon, lat=s.lat, level=a.level, kind=a.kind, status=a.status,
                    issued_at=a.issued_at, created_at=a.created_at, expires_at=a.expires_at,
                    dispatched_at=a.dispatched_at, title_en=a.title_en, title_ne=a.title_ne, body_en=a.body_en,
                    body_ne=a.body_ne, sms_en=a.sms_en, sms_ne=a.sms_ne, generator=a.generator,
                    reasons=(a.facts or {}).get("reasons", []), exposed=(a.facts or {}).get("exposed", []))


@router.get("/alerts", response_model=list[AlertOut])
def alerts(aoi: str | None = None, active: bool = False, days: int = Query(30, le=3650),
           limit: int = Query(100, le=1000), db: Session = Depends(get_db)) -> list[AlertOut]:
    """Published alerts (approved or dispatched)."""
    now = datetime.now(UTC)
    q = (select(Alert, Site).join(Site, Alert.site_id == Site.id)
         .where(Alert.status.in_(PUBLIC_ALERT_STATUSES), Alert.issued_at >= now - timedelta(days=days))
         .order_by(Alert.issued_at.desc()).limit(limit))
    if aoi:
        q = q.where(Site.aoi_id == aoi)
    if active:
        q = q.where((Alert.expires_at.is_(None)) | (Alert.expires_at >= now), Alert.kind != "all_clear")
    return [_alert_out(a, s) for a, s in db.execute(q)]


def _get_alert(db: Session, uid: str) -> tuple[Alert, Site]:
    row = db.execute(select(Alert, Site).join(Site, Alert.site_id == Site.id)
                     .where(Alert.uid == uid, Alert.status.in_(PUBLIC_ALERT_STATUSES + ("cancelled",)))).first()
    if row is None:
        raise HTTPException(404, "alert not found")
    return row[0], row[1]


@router.get("/alerts/feed.xml")
def cap_feed(db: Session = Depends(get_db), settings: Settings = Depends(get_settings_dep)) -> Response:
    """Atom feed of CAP alerts (for alert aggregators)."""
    base = settings.public_base_url.rstrip("/")
    rows = db.execute(select(Alert, Site).join(Site, Alert.site_id == Site.id)
                      .where(Alert.status.in_(PUBLIC_ALERT_STATUSES)).order_by(Alert.issued_at.desc()).limit(50)).all()
    updated = rows[0][0].issued_at if rows else datetime.now(UTC)
    entries = "".join(f"""
  <entry>
    <id>urn:uuid:{a.uid}</id>
    <title>{escape(a.title_en)}</title>
    <updated>{a.issued_at.isoformat()}</updated>
    <link rel="alternate" type="application/cap+xml" href="{base}/api/alerts/{a.uid}/cap.xml"/>
    <summary>{escape(a.title_ne)}</summary>
  </entry>""" for a, _ in rows)
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <id>{base}/api/alerts/feed.xml</id>
  <title>HimSat Engine – glacier hazard alerts</title>
  <updated>{updated.isoformat()}</updated>
  <link rel="self" href="{base}/api/alerts/feed.xml"/>{entries}
</feed>
"""
    return Response(xml, media_type="application/atom+xml")


@router.get("/alerts/{uid}", response_model=AlertOut)
def alert_detail(uid: str, db: Session = Depends(get_db)) -> AlertOut:
    a, s = _get_alert(db, uid)
    return _alert_out(a, s)


@router.get("/alerts/{uid}/cap.xml")
def alert_cap(uid: str, db: Session = Depends(get_db), settings: Settings = Depends(get_settings_dep)) -> Response:
    a, s = _get_alert(db, uid)
    prev = db.get(Alert, a.supersedes_id) if a.supersedes_id else None
    refs = f"{settings.alert_sender_name},urn:uuid:{prev.uid},{prev.issued_at.isoformat()}" if prev else None
    return Response(alert_to_cap(a, s, settings.alert_sender_name, settings.public_base_url, refs),
                    media_type="application/cap+xml")


@router.get("/stats")
def stats(aoi: str | None = None, db: Session = Depends(get_db)) -> dict:
    q = select(Site.latest_level, func.count()).where(Site.status == "active").group_by(Site.latest_level)
    if aoi:
        q = q.where(Site.aoi_id == aoi)
    by_level = dict(db.execute(q).all())
    q2 = select(Site.kind, func.count()).where(Site.status == "active").group_by(Site.kind)
    if aoi:
        q2 = q2.where(Site.aoi_id == aoi)
    return {"sites_by_level": by_level, "sites_by_kind": dict(db.execute(q2).all())}


@router.get("/hindcasts")
def hindcasts(settings: Settings = Depends(get_settings_dep)) -> list[dict]:
    root = settings.data_dir / "hindcasts"
    out = []
    for p in sorted(root.glob("*/report.json")) if root.exists() else []:
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({"name": p.parent.name, "aoi": r["aoi"]["id"], "aoi_name": r["aoi"]["name"],
                    "period": r["period"], "event_time": r["event_time"], "alerts": len(r["alerts"])})
    return out


@router.get("/hindcasts/{name}")
def hindcast(name: str, settings: Settings = Depends(get_settings_dep)) -> Response:
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "bad name")
    p = settings.data_dir / "hindcasts" / name / "report.json"
    if not p.exists():
        raise HTTPException(404, "hindcast not found")
    return Response(p.read_text(encoding="utf-8"), media_type="application/json")
