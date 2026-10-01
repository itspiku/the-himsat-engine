"""Admin endpoints (API key): alert review queue, subscribers, pipeline runs."""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from himsat.alerts.service import approve_alert, cancel_alert, dispatch_alert
from himsat.api.deps import get_db, get_settings_dep, require_admin
from himsat.api.routes.public import _alert_out
from himsat.api.schemas import AlertOut, ApproveIn, RunIn, SubscriberIn, SubscriberOut
from himsat.config import Settings, load_aois
from himsat.db.models import Alert, Delivery, PipelineRun, Site, Subscriber
from himsat.geo.geometry import from_geojson, set_geom

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])
log = logging.getLogger(__name__)
_run_lock = threading.Lock()


@router.get("/alerts", response_model=list[AlertOut])
def review_queue(status: str = "pending_review", db: Session = Depends(get_db)) -> list[AlertOut]:
    rows = db.execute(select(Alert, Site).join(Site, Alert.site_id == Site.id).where(Alert.status == status)
                      .order_by(Alert.created_at.desc()).limit(200)).all()
    return [_alert_out(a, s) for a, s in rows]


def _alert(db: Session, alert_id: int) -> Alert:
    a = db.get(Alert, alert_id)
    if a is None:
        raise HTTPException(404, "alert not found")
    return a


@router.post("/alerts/{alert_id}/approve")
def approve(alert_id: int, body: ApproveIn, db: Session = Depends(get_db),
            settings: Settings = Depends(get_settings_dep), who: str = Depends(require_admin)) -> dict:
    try:
        return approve_alert(db, _alert(db, alert_id), f"{body.user} ({who})", settings)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@router.post("/alerts/{alert_id}/cancel")
def cancel(alert_id: int, body: ApproveIn, db: Session = Depends(get_db), who: str = Depends(require_admin)) -> dict:
    cancel_alert(db, _alert(db, alert_id), f"{body.user} ({who})")
    return {"status": "cancelled"}


@router.post("/alerts/{alert_id}/redispatch")
def redispatch(alert_id: int, db: Session = Depends(get_db), settings: Settings = Depends(get_settings_dep)) -> dict:
    try:
        return dispatch_alert(db, _alert(db, alert_id), settings)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@router.get("/alerts/{alert_id}/deliveries")
def deliveries(alert_id: int, db: Session = Depends(get_db)) -> list[dict]:
    rows = db.execute(select(Delivery, Subscriber).join(Subscriber).where(Delivery.alert_id == alert_id)).all()
    return [{"subscriber": s.name, "channel": d.channel, "status": d.status, "attempts": d.attempts,
             "error": d.last_error, "sent_at": d.sent_at} for d, s in rows]


def _sub_out(s: Subscriber) -> SubscriberOut:
    import json

    return SubscriberOut(id=s.id, created_at=s.created_at, name=s.name, org_type=s.org_type, language=s.language,
                         phone=s.phone, email=s.email, webhook_url=s.webhook_url, channels=s.channels,
                         min_level=s.min_level, aoi_ids=s.aoi_ids, site_ids=[str(x) for x in s.site_ids],
                         area=json.loads(s.geom) if s.geom else None, active=s.active)


def _apply(s: Subscriber, body: SubscriberIn) -> None:
    for f in ("name", "org_type", "language", "phone", "email", "webhook_url", "min_level", "active"):
        setattr(s, f, getattr(body, f))
    s.channels, s.aoi_ids, s.site_ids = list(body.channels), list(body.aoi_ids), list(body.site_ids)
    if body.area:
        geom = from_geojson(body.area)
        if geom is None or geom.geom_type not in ("Polygon", "MultiPolygon"):
            raise HTTPException(422, "area must be a GeoJSON Polygon or MultiPolygon")
        set_geom(s, geom)
    else:
        set_geom(s, None)


@router.get("/subscribers", response_model=list[SubscriberOut])
def list_subscribers(db: Session = Depends(get_db)) -> list[SubscriberOut]:
    return [_sub_out(s) for s in db.scalars(select(Subscriber).order_by(Subscriber.id))]


@router.post("/subscribers", response_model=SubscriberOut, status_code=201)
def create_subscriber(body: SubscriberIn, db: Session = Depends(get_db)) -> SubscriberOut:
    s = Subscriber()
    _apply(s, body)
    db.add(s)
    db.flush()
    return _sub_out(s)


@router.put("/subscribers/{sub_id}", response_model=SubscriberOut)
def update_subscriber(sub_id: int, body: SubscriberIn, db: Session = Depends(get_db)) -> SubscriberOut:
    s = db.get(Subscriber, sub_id)
    if s is None:
        raise HTTPException(404, "subscriber not found")
    _apply(s, body)
    db.flush()
    return _sub_out(s)


@router.delete("/subscribers/{sub_id}", status_code=204)
def deactivate_subscriber(sub_id: int, db: Session = Depends(get_db)) -> None:
    s = db.get(Subscriber, sub_id)
    if s is None:
        raise HTTPException(404, "subscriber not found")
    s.active = False


def _run_in_background(body: RunIn, settings: Settings) -> None:
    from himsat.pipeline.monitor import CycleOptions, run_cycle

    if not _run_lock.acquire(blocking=False):
        log.warning("a run is already in progress; skipping")
        return
    try:
        run_cycle(body.aoi_id, CycleOptions(start=body.start, end=body.end, dispatch=body.dispatch),
                  db_url=settings.database_url, settings=settings)
    except Exception:
        log.exception("background run failed")
    finally:
        _run_lock.release()


@router.post("/runs", status_code=202)
def start_run(body: RunIn, tasks: BackgroundTasks, settings: Settings = Depends(get_settings_dep)) -> dict:
    if body.aoi_id not in load_aois(settings):
        raise HTTPException(404, "unknown AOI")
    if _run_lock.locked():
        raise HTTPException(409, "a run is already in progress")
    tasks.add_task(_run_in_background, body, settings)
    return {"status": "accepted", "aoi_id": body.aoi_id}


@router.get("/runs")
def runs(limit: int = 20, db: Session = Depends(get_db)) -> list[dict]:
    return [{"id": r.id, "aoi": r.aoi_id, "kind": r.kind, "status": r.status, "started_at": r.started_at,
             "finished_at": r.finished_at, "stats": r.stats, "error": r.error}
            for r in db.scalars(select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(limit))]
