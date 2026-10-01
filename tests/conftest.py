from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest

os.environ.setdefault("HIMSAT_ENV", "test")
os.environ.setdefault("HIMSAT_LLM_BACKEND", "none")


@pytest.fixture
def db_url(tmp_path):
    from himsat.db.session import init_db

    url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    init_db(url)  # runs the real migrations
    return url


@pytest.fixture
def session(db_url):
    from himsat.db.session import get_session_factory

    s = get_session_factory(db_url)()
    yield s
    s.rollback()
    s.close()


@pytest.fixture
def t0():
    return datetime(2026, 8, 1, tzinfo=UTC)


class FakeSite:
    def __init__(self, **kw):
        self.code = kw.get("code", "LHD-S0001")
        self.name = kw.get("name", "Unstable slope near Lirung Glacier (LHD-S0001)")
        self.name_ne = kw.get("name_ne", "लिरुङ हिमनदी नजिकको अस्थिर भिर (LHD-S0001)")
        self.kind = kw.get("kind", "slope")
        self.lat = kw.get("lat", 28.2881)
        self.lon = kw.get("lon", 85.5282)
        self.elevation_m = kw.get("elevation_m", 5130.0)


@pytest.fixture
def fake_site():
    return FakeSite()
