from datetime import UTC, datetime

import pytest

from himsat.db.models import AOI, Alert, Site
from himsat.pipeline.hindcast import reassess
from himsat.pipeline.monitor import CycleOptions

T = datetime(2026, 9, 1, tzinfo=UTC)


def test_cycle_kinds():
    assert CycleOptions().kind == "monitor" and not CycleOptions().replay
    bf = CycleOptions(backfill=True)
    assert bf.kind == "backfill" and bf.replay
    assert CycleOptions(hindcast=True).kind == "hindcast"


def _alert(session, aoi, code, status="pending_review"):
    site = Site(aoi_id=aoi, code=code, kind="glacier", name=code, name_ne=code, lon=85.5, lat=28.3)
    session.add(site)
    session.flush()
    a = Alert(site_id=site.id, level="medium", kind="alert", status=status, issued_at=T, title_en="t", body_en="b",
              sms_en="s", title_ne="t", body_ne="b", sms_ne="s", generator="template")
    session.add(a)
    session.flush()
    return a


def test_live_reassess_is_scoped_to_one_aoi(session, db_url):
    session.add_all([AOI(id="rasuwa-lhende", name="r"), AOI(id="rolwaling-tamakoshi", name="w")])
    _alert(session, "rasuwa-lhende", "LHD-G0001")
    keep = _alert(session, "rolwaling-tamakoshi", "RLW-G0001")
    session.commit()
    reassess("rasuwa-lhende", db_url, guard_dispatched=True)
    session.expire_all()
    assert [a.id for a in session.query(Alert).all()] == [keep.id]


def test_live_reassess_refuses_after_dispatch(session, db_url):
    session.add(AOI(id="rasuwa-lhende", name="r"))
    _alert(session, "rasuwa-lhende", "LHD-G0001", status="dispatched")
    session.commit()
    with pytest.raises(RuntimeError, match="dispatched"):
        reassess("rasuwa-lhende", db_url, guard_dispatched=True)
