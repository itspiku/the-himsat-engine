from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterator

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from himsat.config import Settings
from himsat.db.session import get_session_factory


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_db(settings: Settings = Depends(get_settings_dep)) -> Iterator[Session]:
    session = get_session_factory(settings.database_url)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def require_admin(x_api_key: str | None = Header(default=None), settings: Settings = Depends(get_settings_dep)) -> str:
    """Admin endpoints: ``X-API-Key`` must hash (sha256) to one of ``HIMSAT_ADMIN_API_KEYS``."""
    if not settings.admin_api_keys:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "admin API disabled: no keys configured")
    if not x_api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing X-API-Key")
    digest = hashlib.sha256(x_api_key.encode()).hexdigest()
    for k in settings.admin_api_keys:
        if hmac.compare_digest(digest, k.lower()):
            return "admin:" + digest[:8]
    raise HTTPException(status.HTTP_403_FORBIDDEN, "invalid API key")
