from datetime import UTC, datetime, timedelta

import httpx
import respx
from shapely.geometry import Point

from himsat.alerts.channels import sign_payload
from himsat.alerts.service import approve_alert, decide, dispatch_alert, process_assessment
from himsat.config import Settings
from himsat.db.models import AOI, Alert, Asset, Delivery, RiskAssessment, Site, SiteExposure, Subscriber
from himsat.geo.geometry import set_geom
from himsat.risk.model import RiskModel

T = datetime(2026, 8, 24, 6, tzinfo=UTC)


def _result(level_feats):
    return RiskModel().assess("slope", level_feats, 0.9, 0.8)


def _prev(level, kind="alert", status="dispatched", issued=T - timedelta(days=1), score=0.5):
    return Alert(level=level, kind=kind, status=status, issued_at=issued, facts={"score": score})


def test_decide_policy():
    high = _result({"velocity_ratio": 5.0, "velocity_trend": 4})
    assert high.level == "high"
    static = RiskModel().assess("glacial_lake", {"lake_area_km2": 0.8, "glacier_contact_m": 10,
                                                 "steep_walls_deg": 50}, 0.9, 0.8)
    low = _result({})
    assert decide(None, high, T) == "alert"
    assert decide(None, static, T) is None  # static-only: no alert
    assert decide(_prev("medium"), high, T) == "alert"  # escalation
    assert decide(_prev("high", score=high.score), high, T) is None  # same level, no worse
    assert decide(_prev("high"), low, T) == "all_clear"
    assert decide(_prev("high", status="pending_review"), low, T) is None  # never sent → no all-clear


def _setup(session):
    session.add(AOI(id="test", name="Test"))
    site = Site(code="TST-S0001", aoi_id="test", kind="slope", name="Test slope", name_ne="परीक्षण भिर",
                lon=85.53, lat=28.29, elevation_m=5100)
    set_geom(site, Point(85.53, 28.29).buffer(0.01))
    a = Asset(external_id="x:1", kind="border_crossing", name="Rasuwagadhi", name_ne="रसुवागढी", lon=85.379, lat=28.279)
    session.add_all([site, a])
    session.flush()
    session.add(SiteExposure(site_id=site.id, asset_id=a.id, path_distance_km=22.8, offset_m=30,
                             height_above_channel_m=0, travel_time_min=15.2))
    near = Subscriber(name="Gosaikunda RM", language="ne", webhook_url="https://hooks.example.org/a",
                      channels=["webhook"], min_level="medium")
    set_geom(near, Point(85.38, 28.28).buffer(0.05))  # covers the exposed border post
    far = Subscriber(name="Far away", language="en", webhook_url="https://hooks.example.org/b",
                     channels=["webhook"], min_level="medium")
    set_geom(far, Point(84.0, 27.5).buffer(0.05))
    high_only = Subscriber(name="NDRRMA", language="both", webhook_url="https://hooks.example.org/c",
                           channels=["webhook", "sms"], min_level="high")
    session.add_all([near, far, high_only])
    session.flush()
    return site


@respx.mock
def test_high_alert_auto_dispatches_to_matching_subscribers(session):
    site = _setup(session)
    route = respx.post(url__startswith="https://hooks.example.org/").mock(return_value=httpx.Response(200))
    result = _result({"velocity_ratio": 5.0, "velocity_trend": 4})
    ra = RiskAssessment(site_id=site.id, assessed_at=T, level=result.level, score=result.score, hazard=result.hazard,
                        exposure=result.exposure, confidence=result.confidence)
    session.add(ra)
    session.flush()
    s = Settings(webhook_secret="s3cret", public_base_url="https://himsat.example.org")
    alert = process_assessment(session, site, ra, result, llm=None, settings=s, now=T)
    assert alert.status == "dispatched" and alert.generator == "template"
    urls = sorted(str(c.request.url) for c in route.calls)
    assert urls == ["https://hooks.example.org/a", "https://hooks.example.org/c"]
    req = route.calls[0].request
    assert req.headers["X-HimSat-Signature"] == sign_payload(req.content, "s3cret")
    # sms channel had no provider configured → skipped, not failed
    assert session.query(Delivery).filter_by(status="failed").count() == 0
    # idempotent redispatch
    dispatch_alert(session, alert, s)
    assert len(route.calls) == 2


@respx.mock
def test_medium_alert_waits_for_review(session):
    site = _setup(session)
    route = respx.post(url__startswith="https://hooks.example.org/").mock(return_value=httpx.Response(500))
    result = _result({"velocity_ratio": 3.2})
    assert result.level == "medium"
    ra = RiskAssessment(site_id=site.id, assessed_at=T, level=result.level, score=result.score, hazard=result.hazard,
                        exposure=result.exposure, confidence=result.confidence)
    session.add(ra)
    session.flush()
    s = Settings()
    alert = process_assessment(session, site, ra, result, llm=None, settings=s, now=T)
    assert alert.status == "pending_review" and not route.called
    stats = approve_alert(session, alert, "duty-officer", s)
    assert alert.status == "dispatched" and stats["failed"] == 1  # only the in-area medium subscriber
    d = session.query(Delivery).one()
    assert d.status == "failed" and "500" in d.last_error


def test_roster_import_is_idempotent(session, tmp_path):
    import pytest

    from himsat.alerts.roster import import_roster

    roster = tmp_path / "subs.yaml"
    roster.write_text("""subscribers:
  - name: Test RM
    org_type: municipality
    language: ne
    phone: "+9779800000001"
    channels: [sms]
    bbox: [85.2, 28.0, 85.6, 28.4]
  - name: Ops desk
    email: ops@example.org
    channels: [email]
""", encoding="utf-8")
    assert import_roster(session, roster) == {"created": 2, "updated": 0, "deactivated": 0}
    assert import_roster(session, roster) == {"created": 0, "updated": 2, "deactivated": 0}
    rm = session.query(Subscriber).filter_by(name="Test RM").one()
    assert rm.geom and rm.channels == ["sms"]
    roster.write_text("subscribers:\n  - name: Broken\n    channels: [sms]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sms"):
        import_roster(session, roster)


def test_example_roster_is_valid(session):
    from pathlib import Path

    from himsat.alerts.roster import import_roster

    stats = import_roster(session, Path(__file__).resolve().parents[1] / "config" / "subscribers.example.yaml")
    assert stats["created"] == 8
