from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from shapely.geometry import Point

from himsat.db.models import AOI, ChangeEvent
from himsat.geo.geometry import set_geom
from himsat.pipeline.s1 import _after_wet_snow, _confirming_event

T = datetime(2026, 8, 31, tzinfo=UTC)
CTX = SimpleNamespace(cfg=SimpleNamespace(id="test"))


def _ev(session, kind, days_ago, orbit, lon=85.50, area=300_000, **attrs):
    e = ChangeEvent(aoi_id="test", kind=kind, sensor="S1", detected_at=T - timedelta(days=days_ago), lon=lon,
                    lat=28.30, area_m2=area, confidence=0.9, attrs={"orbit": orbit, **attrs})
    set_geom(e, Point(lon, 28.30).buffer(0.004))
    session.add(e)
    session.flush()
    return e


def test_confirmation_needs_independent_geometry_and_same_kind(session):
    session.add(AOI(id="test", name="t"))
    new = _ev(session, "mass_movement", 0, orbit=121)
    acq = SimpleNamespace(datetime=T, relative_orbit=121)
    assert _confirming_event(session, CTX, new, acq) is None  # nothing else yet
    _ev(session, "mass_movement", 3, orbit=121)  # same viewing geometry: not independent
    _ev(session, "velocity_hotspot", 3, orbit=85)  # different kind: never confirms a scar
    _ev(session, "mass_movement", 3, orbit=85, snow_like=True)  # wet snow: excluded
    _ev(session, "mass_movement", 3, orbit=19, lon=86.0)  # elsewhere
    _ev(session, "mass_movement", 30, orbit=19)  # too old
    assert _confirming_event(session, CTX, new, acq) is None
    partner = _ev(session, "mass_movement", 3, orbit=85)
    assert _confirming_event(session, CTX, new, acq).id == partner.id
    hs = _ev(session, "velocity_hotspot", 0, orbit=121)
    assert _confirming_event(session, CTX, hs, acq, kind="velocity_hotspot").kind == "velocity_hotspot"


def test_change_after_wet_snow_is_a_reversal(session):
    session.add(AOI(id="test", name="t"))
    acq = SimpleNamespace(datetime=T, relative_orbit=85)
    bright = _ev(session, "mass_movement", 0, orbit=85)
    assert not _after_wet_snow(session, CTX, bright, acq)
    _ev(session, "mass_movement", 12, orbit=85, snow_like=True)
    assert _after_wet_snow(session, CTX, bright, acq)
