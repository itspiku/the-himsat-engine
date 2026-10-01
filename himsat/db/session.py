from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from himsat.config import get_settings

_engines: dict[str, Engine] = {}
_factories: dict[str, sessionmaker[Session]] = {}


def get_engine(url: str | None = None) -> Engine:
    url = url or get_settings().database_url
    if url not in _engines:
        kwargs: dict = {"future": True, "pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
            db_path = url.split("sqlite:///", 1)[-1]
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):

            @event.listens_for(engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - trivial
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA foreign_keys=ON")
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.close()

        _engines[url] = engine
    return _engines[url]


def get_session_factory(url: str | None = None) -> sessionmaker[Session]:
    url = url or get_settings().database_url
    if url not in _factories:
        _factories[url] = sessionmaker(bind=get_engine(url), expire_on_commit=False, future=True)
    return _factories[url]


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    session = get_session_factory(url)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(url: str | None = None, *, use_migrations: bool = True) -> None:
    """Bring the schema up to date.

    Alembic migrations are the source of truth. ``use_migrations=False`` falls back to
    ``create_all`` for throwaway databases (tests, hindcast scratch DBs).
    """
    url = url or get_settings().database_url
    if use_migrations:
        from himsat.db.migrate import upgrade

        upgrade(url)
    else:
        from himsat.db.models import Base

        Base.metadata.create_all(get_engine(url))
