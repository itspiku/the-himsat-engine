"""Relational model.

Geometries are stored as GeoJSON text (EPSG:4326) together with bounding-box columns, so the
same schema runs on SQLite (development, single-box deployments) and PostgreSQL (production)
without a spatial extension. At the scale of a national warning system (tens of thousands of
features) bbox pre-filtering in SQL followed by exact tests in Shapely is fast enough.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


def utcnow() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetimes on every backend (SQLite drops tzinfo otherwise)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        value = value.astimezone(UTC)
        return value.replace(tzinfo=None) if dialect.name == "sqlite" else value

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON, datetime: UTCDateTime}


class GeomMixin:
    """GeoJSON geometry + bbox columns. Use ``himsat.geo.geometry.set_geom`` to assign."""

    geom: Mapped[str | None] = mapped_column(Text, nullable=True)
    min_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_lat: Mapped[float | None] = mapped_column(Float, nullable=True)


class AOI(GeomMixin, Base):
    __tablename__ = "aois"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    name_ne: Mapped[str] = mapped_column(String(200), default="")
    config: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Site(GeomMixin, Base):
    """A monitored feature: glacial lake, glacier / ice-rock slope, barrier lake, landslide."""

    __tablename__ = "sites"
    __table_args__ = (
        Index("ix_sites_bbox", "min_lon", "min_lat", "max_lon", "max_lat"),
        Index("ix_sites_aoi_kind", "aoi_id", "kind"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(32), unique=True)
    aoi_id: Mapped[str] = mapped_column(ForeignKey("aois.id"))
    kind: Mapped[str] = mapped_column(String(32))  # glacial_lake | glacier | slope | barrier_lake | landslide
    name: Mapped[str] = mapped_column(String(200))
    name_ne: Mapped[str] = mapped_column(String(200), default="")
    lon: Mapped[float] = mapped_column(Float)
    lat: Mapped[float] = mapped_column(Float)
    elevation_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(64), default="discovered")
    status: Mapped[str] = mapped_column(String(16), default="active")
    attrs: Mapped[dict[str, Any]] = mapped_column(default=dict)
    flow_path: Mapped[str | None] = mapped_column(Text, nullable=True)  # GeoJSON LineString downstream
    first_seen: Mapped[datetime | None] = mapped_column(nullable=True)
    last_seen: Mapped[datetime | None] = mapped_column(nullable=True)
    latest_level: Mapped[str] = mapped_column(String(16), default="unknown")
    latest_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    latest_assessed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)

    observations: Mapped[list[Observation]] = relationship(back_populates="site", cascade="all, delete-orphan")
    exposures: Mapped[list[SiteExposure]] = relationship(back_populates="site", cascade="all, delete-orphan")


class Scene(Base):
    """An acquisition (possibly a mosaic of several STAC items) and its processing state."""

    __tablename__ = "scenes"
    __table_args__ = (UniqueConstraint("aoi_id", "sensor", "key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    aoi_id: Mapped[str] = mapped_column(ForeignKey("aois.id"))
    sensor: Mapped[str] = mapped_column(String(8))  # S1 | S2
    key: Mapped[str] = mapped_column(String(128))
    acquired_at: Mapped[datetime] = mapped_column()
    platform: Mapped[str] = mapped_column(String(32), default="")
    relative_orbit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cloud_cover: Mapped[float | None] = mapped_column(Float, nullable=True)
    item_ids: Mapped[list[Any]] = mapped_column(default=list)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|processed|skipped|failed
    stats: Mapped[dict[str, Any]] = mapped_column(default=dict)
    processed_at: Mapped[datetime | None] = mapped_column(nullable=True)


class Observation(GeomMixin, Base):
    """One measurement of a site from one acquisition (or acquisition pair)."""

    __tablename__ = "observations"
    __table_args__ = (
        UniqueConstraint("site_id", "kind", "scene_key"),
        Index("ix_obs_site_kind_time", "site_id", "kind", "observed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(32))  # lake_area | velocity | sar_change | surface_change | turbidity
    sensor: Mapped[str] = mapped_column(String(8))
    scene_key: Mapped[str] = mapped_column(String(160))
    observed_at: Mapped[datetime] = mapped_column()
    reference_at: Mapped[datetime | None] = mapped_column(nullable=True)  # first image of a pair
    values: Mapped[dict[str, Any]] = mapped_column(default=dict)
    quality: Mapped[float] = mapped_column(Float, default=1.0)
    method: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    site: Mapped[Site] = relationship(back_populates="observations")


class ChangeEvent(GeomMixin, Base):
    """Area-wide change detections not (yet) tied to a monitored site."""

    __tablename__ = "change_events"
    __table_args__ = (Index("ix_events_aoi_time", "aoi_id", "detected_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    aoi_id: Mapped[str] = mapped_column(ForeignKey("aois.id"))
    kind: Mapped[str] = mapped_column(String(32))  # mass_movement | vegetation_loss | cropland_change | river_blockage
    sensor: Mapped[str] = mapped_column(String(8))
    detected_at: Mapped[datetime] = mapped_column()
    pre_at: Mapped[datetime | None] = mapped_column(nullable=True)
    lon: Mapped[float] = mapped_column(Float)
    lat: Mapped[float] = mapped_column(Float)
    area_m2: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    attrs: Mapped[dict[str, Any]] = mapped_column(default=dict)
    site_id: Mapped[int | None] = mapped_column(ForeignKey("sites.id", ondelete="SET NULL"), nullable=True)
    scene_key: Mapped[str] = mapped_column(String(160), default="")


class RiskAssessment(Base):
    __tablename__ = "risk_assessments"
    __table_args__ = (Index("ix_risk_site_time", "site_id", "assessed_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"))
    assessed_at: Mapped[datetime] = mapped_column()  # "as of" time: latest evidence used
    level: Mapped[str] = mapped_column(String(16))  # low | medium | high
    score: Mapped[float] = mapped_column(Float)
    hazard: Mapped[float] = mapped_column(Float)
    exposure: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    indicators: Mapped[dict[str, Any]] = mapped_column(default=dict)
    reasons: Mapped[list[Any]] = mapped_column(default=list)
    model_version: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Asset(Base):
    """Exposed element at risk downstream (settlement, hydropower, bridge, border post, school...)."""

    __tablename__ = "assets"
    __table_args__ = (Index("ix_assets_lonlat", "lon", "lat"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(64), unique=True)  # e.g. osm:node/123
    kind: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(200), default="")
    name_ne: Mapped[str] = mapped_column(String(200), default="")
    lon: Mapped[float] = mapped_column(Float)
    lat: Mapped[float] = mapped_column(Float)
    elevation_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    attrs: Mapped[dict[str, Any]] = mapped_column(default=dict)
    source: Mapped[str] = mapped_column(String(32), default="osm")


class SiteExposure(Base):
    __tablename__ = "site_exposures"
    __table_args__ = (UniqueConstraint("site_id", "asset_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"))
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"))
    path_distance_km: Mapped[float] = mapped_column(Float)
    offset_m: Mapped[float] = mapped_column(Float)
    height_above_channel_m: Mapped[float] = mapped_column(Float)
    travel_time_min: Mapped[float] = mapped_column(Float)

    site: Mapped[Site] = relationship(back_populates="exposures")
    asset: Mapped[Asset] = relationship()


class Subscriber(GeomMixin, Base):
    __tablename__ = "subscribers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200))
    org_type: Mapped[str] = mapped_column(String(32), default="other")
    language: Mapped[str] = mapped_column(String(8), default="ne")  # ne | en | both
    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    email: Mapped[str | None] = mapped_column(String(200), nullable=True)
    webhook_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    channels: Mapped[list[Any]] = mapped_column(default=list)  # sms | email | webhook
    min_level: Mapped[str] = mapped_column(String(16), default="medium")
    aoi_ids: Mapped[list[Any]] = mapped_column(default=list)  # empty = all
    site_ids: Mapped[list[Any]] = mapped_column(default=list)  # empty = all (subject to area)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


def new_uid() -> str:
    return str(uuid.uuid4())


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (Index("ix_alerts_site_time", "site_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uid: Mapped[str] = mapped_column(String(36), unique=True, default=new_uid)
    site_id: Mapped[int] = mapped_column(ForeignKey("sites.id", ondelete="CASCADE"))
    assessment_id: Mapped[int | None] = mapped_column(ForeignKey("risk_assessments.id"), nullable=True)
    level: Mapped[str] = mapped_column(String(16))
    kind: Mapped[str] = mapped_column(String(16), default="alert")  # alert | update | all_clear
    # pending_review | approved | dispatched | suppressed | cancelled
    status: Mapped[str] = mapped_column(String(20), default="pending_review")
    issued_at: Mapped[datetime] = mapped_column()  # the evidence time the alert refers to
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(nullable=True)
    dispatched_at: Mapped[datetime | None] = mapped_column(nullable=True)
    title_en: Mapped[str] = mapped_column(Text, default="")
    body_en: Mapped[str] = mapped_column(Text, default="")
    sms_en: Mapped[str] = mapped_column(Text, default="")
    title_ne: Mapped[str] = mapped_column(Text, default="")
    body_ne: Mapped[str] = mapped_column(Text, default="")
    sms_ne: Mapped[str] = mapped_column(Text, default="")
    generator: Mapped[str] = mapped_column(String(100), default="template")
    facts: Mapped[dict[str, Any]] = mapped_column(default=dict)
    validation: Mapped[dict[str, Any]] = mapped_column(default=dict)
    supersedes_id: Mapped[int | None] = mapped_column(ForeignKey("alerts.id"), nullable=True)

    site: Mapped[Site] = relationship()
    deliveries: Mapped[list[Delivery]] = relationship(back_populates="alert", cascade="all, delete-orphan")


class Delivery(Base):
    __tablename__ = "deliveries"
    __table_args__ = (UniqueConstraint("alert_id", "subscriber_id", "channel"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[int] = mapped_column(ForeignKey("alerts.id", ondelete="CASCADE"))
    subscriber_id: Mapped[int] = mapped_column(ForeignKey("subscribers.id", ondelete="CASCADE"))
    channel: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending | sent | failed
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider_ref: Mapped[str | None] = mapped_column(String(200), nullable=True)
    sent_at: Mapped[datetime | None] = mapped_column(nullable=True)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)

    alert: Mapped[Alert] = relationship(back_populates="deliveries")
    subscriber: Mapped[Subscriber] = relationship()


class CellVelocity(Base):
    """Downslope velocity of one watch cell (≈1 km² of steep/high ice or rock) from one image pair.

    Every steep, high slope in an AOI is tracked this way, not only mapped glaciers. The anomaly
    test that promotes a cell to a monitored 'slope' site runs on these rows.
    """

    __tablename__ = "cell_velocities"
    __table_args__ = (UniqueConstraint("aoi_id", "cell", "pair_key"),
                      Index("ix_cellvel_aoi_cell_time", "aoi_id", "cell", "pair_end"))

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    aoi_id: Mapped[str] = mapped_column(String(64))
    cell: Mapped[str] = mapped_column(String(16))  # "row_col" on the AOI cell lattice
    pair_key: Mapped[str] = mapped_column(String(160))
    pair_start: Mapped[datetime] = mapped_column()
    pair_end: Mapped[datetime] = mapped_column()
    orbit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    v: Mapped[float] = mapped_column(Float)  # median downslope velocity, m/day
    se: Mapped[float] = mapped_column(Float)  # standard error, m/day
    n: Mapped[int] = mapped_column(Integer)


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    aoi_id: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(16), default="monitor")
    started_at: Mapped[datetime] = mapped_column(default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running | ok | failed
    stats: Mapped[dict[str, Any]] = mapped_column(default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
