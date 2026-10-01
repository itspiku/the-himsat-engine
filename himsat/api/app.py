"""FastAPI application: public API + public map (static web build) + admin API + metrics."""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from sqlalchemy import func, select

from himsat import __version__
from himsat.api.routes import admin, public
from himsat.config import Settings, get_settings
from himsat.db.session import get_session_factory, init_db
from himsat.util import setup_logging

log = logging.getLogger(__name__)


def _metrics(app: FastAPI) -> None:
    reg = CollectorRegistry()
    req_count = Counter("himsat_http_requests_total", "HTTP requests", ["method", "route", "status"], registry=reg)
    req_time = Histogram("himsat_http_request_seconds", "HTTP latency", ["route"], registry=reg)
    sites_level = Gauge("himsat_sites", "Active sites by risk level", ["level"], registry=reg)
    alerts_status = Gauge("himsat_alerts", "Alerts by status", ["status"], registry=reg)
    last_run = Gauge("himsat_last_run_finished_timestamp", "Unix time of the last finished run", ["aoi"], registry=reg)

    @app.middleware("http")
    async def _observe(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        route = getattr(request.scope.get("route"), "path", "unmatched")
        req_count.labels(request.method, route, str(response.status_code)).inc()
        req_time.labels(route).observe(time.perf_counter() - t0)
        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        from himsat.db.models import Alert, PipelineRun, Site

        with get_session_factory(app.state.settings.database_url)() as s:
            for lvl, n in s.execute(select(Site.latest_level, func.count()).where(Site.status == "active")
                                    .group_by(Site.latest_level)):
                sites_level.labels(lvl).set(n)
            for st, n in s.execute(select(Alert.status, func.count()).group_by(Alert.status)):
                alerts_status.labels(st).set(n)
            for aoi, t in s.execute(select(PipelineRun.aoi_id, func.max(PipelineRun.finished_at))
                                    .group_by(PipelineRun.aoi_id)):
                if t:
                    last_run.labels(aoi).set(t.timestamp())
        return Response(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.log_json)
    init_db(settings.database_url)
    app = FastAPI(title="HimSat Engine API", version=__version__,
                  description="Satellite early warning for Himalayan glacier and glacial-lake hazards. "
                              "Public endpoints are read-only; /api/admin requires an X-API-Key.")
    app.state.settings = settings
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST", "PUT",
                       "DELETE"], allow_headers=["*"])

    @app.middleware("http")
    async def _security_headers(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/assets/"):  # content-hashed build output
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-cache")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        return response

    app.include_router(public.router)
    app.include_router(admin.router)
    _metrics(app)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse({"detail": "internal error"}, status_code=500)

    dist = settings.web_dist_dir
    if dist.exists() and (dist / "index.html").exists():
        if (dist / "assets").exists():
            app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                return JSONResponse({"detail": "Not Found"}, status_code=404)
            f = (dist / path).resolve()
            if path and f.is_file() and dist.resolve() in f.parents:
                return FileResponse(f, headers={"Cache-Control": "public, max-age=3600"})
            # the HTML shell must always be revalidated so users get new builds immediately
            return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})
    else:
        @app.get("/", include_in_schema=False)
        def root() -> dict:
            return {"name": "HimSat Engine API", "version": __version__, "docs": "/docs",
                    "note": "web map not built: run `npm run build` in web/"}
    return app
