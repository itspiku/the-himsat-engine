"""Alert lifecycle: policy → composition → review/approval → dispatch → delivery tracking."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from shapely.geometry import Point
from sqlalchemy import select
from sqlalchemy.orm import Session

from himsat.alerts.channels import ChannelError, Message, get_channels
from himsat.alerts.compose import compose
from himsat.alerts.facts import build_facts
from himsat.alerts.llm import LLMClient
from himsat.config import Settings, get_settings
from himsat.db.models import Alert, Asset, Delivery, RiskAssessment, Site, SiteExposure, Subscriber, utcnow
from himsat.geo.geometry import from_geojson
from himsat.risk.model import RiskResult

log = logging.getLogger(__name__)

RANK = {"unknown": -1, "low": 0, "medium": 1, "high": 2}
COOLDOWN = {"high": timedelta(hours=24), "medium": timedelta(hours=72)}
EXPIRY = {"high": timedelta(hours=72), "medium": timedelta(days=7), "low": timedelta(days=2)}
ACTIVE = ("pending_review", "approved", "dispatched")


def latest_active_alert(session: Session, site_id: int, now: datetime) -> Alert | None:
    a = session.scalars(
        select(Alert).where(Alert.site_id == site_id, Alert.status.in_(ACTIVE))
        .order_by(Alert.issued_at.desc()).limit(1)).first()
    if a and a.expires_at and a.expires_at < now and a.level != "high":
        return None
    return a


def decide(prev: Alert | None, result: RiskResult, now: datetime) -> str | None:
    """Return the alert kind to issue ('alert' | 'update' | 'all_clear') or None."""
    lvl = result.level
    if lvl in ("medium", "high") and not result.alertable:
        return None  # "potentially dangerous" status only: no change has been observed
    if lvl in ("medium", "high"):
        if prev is None or prev.kind == "all_clear" or RANK[prev.level] < RANK[lvl]:
            return "alert"
        if RANK[prev.level] > RANK[lvl]:
            return "update"  # de-escalation high → medium
        prev_score = float(prev.facts.get("score", 0.0))
        if now - prev.issued_at >= COOLDOWN[lvl] and result.score >= prev_score + 0.1:
            return "update"  # same level, materially worse, after cooldown
        return None
    if prev is not None and prev.kind != "all_clear" and prev.status == "dispatched":
        return "all_clear"
    return None


def site_exposures(session: Session, site_id: int) -> list[dict]:
    rows = session.execute(
        select(SiteExposure, Asset).join(Asset, SiteExposure.asset_id == Asset.id)
        .where(SiteExposure.site_id == site_id).order_by(SiteExposure.travel_time_min)).all()
    return [{"asset_id": a.id, "kind": a.kind, "name": a.name, "name_ne": a.name_ne, "lon": a.lon, "lat": a.lat,
             "travel_time_min": e.travel_time_min, "path_distance_km": e.path_distance_km,
             "height_above_channel_m": e.height_above_channel_m} for e, a in rows]


def process_assessment(session: Session, site: Site, assessment: RiskAssessment, result: RiskResult, *,
                       llm: LLMClient | None, settings: Settings | None = None, now: datetime | None = None,
                       dispatch: bool = True) -> Alert | None:
    settings = settings or get_settings()
    now = now or utcnow()
    prev = latest_active_alert(session, site.id, now)
    kind = decide(prev, result, now)
    if kind is None:
        return None
    exposures = site_exposures(session, site.id)
    facts = build_facts(
        alert_kind=kind, level=result.level, previous_level=prev.level if prev else None, site=site,
        reasons=result.reasons, exposures=exposures, evidence_time=assessment.assessed_at,
        needs_confirmation=result.needs_confirmation, hotline=settings.emergency_hotline,
        base_url=settings.public_base_url)
    comp = compose(facts, llm, languages=tuple(settings.llm_languages))
    auto = (settings.dispatch_enabled and dispatch and result.level in settings.auto_dispatch_levels
            and not result.needs_confirmation and kind != "all_clear")
    alert = Alert(
        site_id=site.id, assessment_id=assessment.id, level=result.level, kind=kind,
        status="approved" if auto else "pending_review", issued_at=assessment.assessed_at, created_at=now,
        expires_at=assessment.assessed_at + EXPIRY[result.level],
        approved_by="policy:auto" if auto else None, approved_at=now if auto else None,
        generator=comp.generator, validation=comp.validation,
        facts={**facts.to_dict(), "score": result.score, "hazard": result.hazard, "exposure": result.exposure,
               "confidence": result.confidence},
        supersedes_id=prev.id if prev else None, **comp.texts,
    )
    session.add(alert)
    session.flush()
    log.warning("ALERT %s %s %s for %s (%s) via %s", alert.id, kind, result.level.upper(), site.code, alert.status,
                comp.generator)
    if auto:
        dispatch_alert(session, alert, settings)
    return alert


def _subscriber_matches(sub: Subscriber, alert: Alert, site: Site, exposures: list[dict]) -> bool:
    if not sub.active:
        return False
    # all-clear messages go to everyone who could have received the original alert
    if alert.kind != "all_clear" and RANK.get(alert.level, 0) < RANK.get(sub.min_level, 1):
        return False
    if sub.aoi_ids and site.aoi_id not in sub.aoi_ids:
        return False
    if sub.site_ids and site.id not in sub.site_ids and site.code not in sub.site_ids:
        return False
    if sub.geom:
        area = from_geojson(sub.geom)
        pts = [Point(site.lon, site.lat)] + [Point(e["lon"], e["lat"]) for e in exposures]
        if area is None or not any(area.intersects(p) for p in pts):
            return False
    return True


def select_recipients(session: Session, alert: Alert, site: Site) -> list[Subscriber]:
    exposures = site_exposures(session, site.id)
    subs = session.scalars(select(Subscriber).where(Subscriber.active.is_(True))).all()
    return [s for s in subs if _subscriber_matches(s, alert, site, exposures)]


def _message_for(alert: Alert, lang: str, cap_url: str) -> Message:
    if lang == "en":
        subject, text, sms = alert.title_en, alert.body_en, alert.sms_en
    elif lang == "ne":
        subject, text, sms = alert.title_ne, alert.body_ne, alert.sms_ne
    else:
        subject = f"{alert.title_ne} / {alert.title_en}"
        text = f"{alert.body_ne}\n\n———\n\n{alert.body_en}"
        sms = alert.sms_ne
    payload = {
        "id": alert.uid, "level": alert.level, "kind": alert.kind, "status": alert.status,
        "issued_at": alert.issued_at.isoformat(), "expires_at": alert.expires_at.isoformat() if alert.expires_at else None,
        "site": {k: alert.facts.get(k) for k in ("site_code", "site_name_en", "site_name_ne", "site_kind", "lat", "lon")},
        "title_en": alert.title_en, "title_ne": alert.title_ne, "body_en": alert.body_en, "body_ne": alert.body_ne,
        "sms_en": alert.sms_en, "sms_ne": alert.sms_ne, "cap_url": cap_url,
        "exposed": alert.facts.get("exposed", []), "reasons": alert.facts.get("reasons", []),
    }
    return Message(subject, text, sms, payload)


def dispatch_alert(session: Session, alert: Alert, settings: Settings | None = None) -> dict:
    """Send an approved alert to all matching subscribers. Idempotent per (alert, subscriber, channel)."""
    settings = settings or get_settings()
    if alert.status not in ("approved", "dispatched"):
        raise ValueError(f"alert {alert.id} is {alert.status}, not approved")
    site = session.get(Site, alert.site_id)
    channels = get_channels(settings)
    cap_url = f"{settings.public_base_url.rstrip('/')}/api/alerts/{alert.uid}/cap.xml"
    stats = {"sent": 0, "failed": 0, "skipped": 0}
    for sub in select_recipients(session, alert, site):
        for ch_name in sub.channels or []:
            ch = channels.get(ch_name)
            to = {"sms": sub.phone, "email": sub.email, "webhook": sub.webhook_url}.get(ch_name)
            if ch is None or not to or not getattr(ch, "available", False):
                stats["skipped"] += 1
                continue
            d = session.scalars(select(Delivery).where(Delivery.alert_id == alert.id, Delivery.subscriber_id == sub.id,
                                                       Delivery.channel == ch_name)).first()
            if d is None:
                d = Delivery(alert_id=alert.id, subscriber_id=sub.id, channel=ch_name, status="pending", attempts=0)
                session.add(d)
            if d.status == "sent":
                continue
            d.attempts += 1
            try:
                if ch_name == "sms" and sub.language == "both":
                    ref = ch.send(to, _message_for(alert, "ne", cap_url))
                    ch.send(to, _message_for(alert, "en", cap_url))
                else:
                    ref = ch.send(to, _message_for(alert, sub.language, cap_url))
                d.status, d.provider_ref, d.sent_at, d.last_error = "sent", str(ref)[:200], utcnow(), None
                stats["sent"] += 1
            except ChannelError as e:
                d.status, d.last_error = "failed", str(e)[:1000]
                stats["failed"] += 1
                log.error("delivery failed alert=%s sub=%s ch=%s: %s", alert.id, sub.id, ch_name, e)
    alert.status = "dispatched"
    alert.dispatched_at = alert.dispatched_at or utcnow()
    session.flush()
    log.info("dispatched alert %s: %s", alert.id, stats)
    return stats


def approve_alert(session: Session, alert: Alert, user: str, settings: Settings | None = None) -> dict:
    if alert.status != "pending_review":
        raise ValueError(f"alert {alert.id} is {alert.status}")
    alert.status, alert.approved_by, alert.approved_at = "approved", user, utcnow()
    settings = settings or get_settings()
    if settings.dispatch_enabled:
        return dispatch_alert(session, alert, settings)
    return {"sent": 0, "failed": 0, "skipped": 0, "note": "dispatch disabled"}


def cancel_alert(session: Session, alert: Alert, user: str) -> None:
    if alert.status == "cancelled":
        return
    alert.status = "cancelled"
    alert.approved_by = alert.approved_by or user
    session.flush()
