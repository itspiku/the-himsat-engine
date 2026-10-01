from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class AlertOut(BaseModel):
    uid: str
    id: int
    site_code: str
    site_name: str
    site_name_ne: str
    site_kind: str
    lon: float
    lat: float
    level: str
    kind: str
    status: str
    issued_at: datetime
    created_at: datetime
    expires_at: datetime | None
    dispatched_at: datetime | None
    title_en: str
    title_ne: str
    body_en: str
    body_ne: str
    sms_en: str
    sms_ne: str
    generator: str
    reasons: list[dict[str, Any]] = Field(default_factory=list)
    exposed: list[dict[str, Any]] = Field(default_factory=list)


class SubscriberIn(BaseModel):
    name: str
    org_type: Literal["municipality", "drr_authority", "hydropower", "police", "army", "community", "media",
                      "tourism", "other"] = "other"
    language: Literal["ne", "en", "both"] = "ne"
    phone: str | None = None
    email: str | None = None
    webhook_url: str | None = None
    channels: list[Literal["sms", "email", "webhook"]] = Field(default_factory=lambda: ["sms"])
    min_level: Literal["medium", "high"] = "medium"
    aoi_ids: list[str] = Field(default_factory=list)
    site_ids: list[str] = Field(default_factory=list)
    area: dict[str, Any] | None = Field(default=None, description="GeoJSON polygon limiting the area of interest")
    active: bool = True


class SubscriberOut(SubscriberIn):
    id: int
    created_at: datetime


class ApproveIn(BaseModel):
    user: str = Field(min_length=2, max_length=100, description="duty officer approving the alert")


class RunIn(BaseModel):
    aoi_id: str
    start: datetime | None = None
    end: datetime | None = None
    dispatch: bool = True
