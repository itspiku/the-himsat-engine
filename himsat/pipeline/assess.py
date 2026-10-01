"""Risk assessment of sites as of a point in time (+ alert generation)."""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from himsat.alerts.llm import LLMClient
from himsat.alerts.service import process_assessment
from himsat.config import Settings
from himsat.db.models import Alert, Observation, RiskAssessment, Site, SiteExposure
from himsat.pipeline.context import AOIContext
from himsat.risk.exposure import ExposedAsset, exposure_score
from himsat.risk.features import Obs, site_features
from himsat.risk.model import RiskModel
from himsat.sites import inventory as inv

log = logging.getLogger(__name__)


class Assessor:
    def __init__(self, ctx: AOIContext, settings: Settings, llm: LLMClient | None, dispatch: bool = True):
        self.ctx = ctx
        self.settings = settings
        self.llm = llm
        self.dispatch = dispatch
        self.model = RiskModel()
        self._assets: list[dict] | None = None

    def assets(self, session: Session) -> list[dict]:
        if self._assets is None:
            self._assets = inv.assets_in_box(session, self.ctx.cfg.routing_bbox)
        return self._assets

    def ensure_exposure(self, session: Session, site: Site) -> None:
        if (site.attrs or {}).get("exposure_computed"):
            return
        n = inv.compute_site_exposure(session, site, self.ctx.routing_dem, self.ctx.routing_grid, self.assets(session))
        attrs = dict(site.attrs or {})
        attrs["exposure_computed"] = True
        attrs["exposed_assets"] = n
        site.attrs = attrs

    def assess(self, session: Session, site: Site, as_of: datetime, now: datetime | None = None
               ) -> tuple[RiskAssessment, Alert | None]:
        self.ensure_exposure(session, site)
        rows = session.scalars(select(Observation).where(Observation.site_id == site.id,
                                                         Observation.observed_at <= as_of)).all()
        obs = [Obs(o.kind, o.sensor, o.observed_at, o.values or {}, o.quality) for o in rows]
        feats, conf = site_features(site.kind, obs, as_of, self.model.cfg, site.attrs or {})
        exps = session.scalars(select(SiteExposure).where(SiteExposure.site_id == site.id)).all()
        ex_cfg = self.model.cfg["exposure"]
        exposed = [ExposedAsset(e.asset_id, e.asset.kind, e.asset.name, e.asset.name_ne, e.path_distance_km,
                                e.offset_m, e.height_above_channel_m, e.travel_time_min) for e in exps]
        e_score, e_detail = exposure_score(exposed, ex_cfg["weights"], ex_cfg["scale"], ex_cfg["time_decay_min"])
        result = self.model.assess(site.kind, feats, e_score, conf)
        ra = RiskAssessment(site_id=site.id, assessed_at=as_of, level=result.level, score=result.score,
                            hazard=result.hazard, exposure=result.exposure, confidence=result.confidence,
                            indicators={**result.indicators, "_features": {k: v for k, v in feats.items() if v is not None},
                                        "_exposure": e_detail, "_needs_confirmation": result.needs_confirmation,
                                        "_dynamic": result.dynamic},
                            reasons=result.reasons, model_version=result.model_version)
        session.add(ra)
        session.flush()
        site.latest_level, site.latest_score, site.latest_assessed_at = result.level, result.score, as_of
        alert = process_assessment(session, site, ra, result, llm=self.llm, settings=self.settings, now=now,
                                   dispatch=self.dispatch)
        return ra, alert
