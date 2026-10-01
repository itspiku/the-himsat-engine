"""Runtime configuration.

Two layers:

* ``Settings``: deployment settings from environment variables (prefix ``HIMSAT_``) or a ``.env`` file.
  These cover secrets, endpoints and switches.
* YAML files in ``config/`` hold the science and policy configuration: areas of interest
  (``aois.yaml``) and the risk model (``risk.yaml``). They are versioned with the code, so a
  threshold change is reviewed like a code change.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HIMSAT_", env_file=".env", extra="ignore")

    env: Literal["dev", "prod", "test"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False

    # storage
    database_url: str = f"sqlite:///{(REPO_ROOT / 'data' / 'himsat.db').as_posix()}"
    data_dir: Path = REPO_ROOT / "data"
    config_dir: Path = REPO_ROOT / "config"
    cache_enabled: bool = True

    # imagery
    stac_provider: Literal["planetary-computer", "earth-search"] = "planetary-computer"
    stac_max_items: int = 2000
    s2_max_scene_cloud: float = 80.0  # skip whole acquisitions cloudier than this (%)
    http_retries: int = 4
    io_workers: int = 8  # concurrent remote raster reads

    # AI models
    device: str = "auto"  # auto | cpu | cuda
    segmenter: Literal["auto", "spectral", "prithvi"] = "auto"
    prithvi_checkpoint: Path | None = None  # fine-tuned head+encoder weights (.pt)
    prithvi_backbone: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL"
    sam_model: str | None = None  # e.g. "facebook/sam-vit-base"; None disables SAM refinement

    # LLM (self-hosted, OpenAI-compatible: Ollama / vLLM / llama.cpp server / LocalAI)
    llm_backend: Literal["openai", "none"] = "openai"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:7b-instruct"
    llm_api_key: str = "not-needed"
    llm_timeout_s: float = 90.0
    llm_temperature: float = 0.2
    # languages the LLM may write; others come from the native-reviewed templates
    llm_languages: list[str] = Field(default_factory=lambda: ["en"])

    # alerting
    dispatch_enabled: bool = True
    auto_dispatch_levels: list[str] = Field(default_factory=lambda: ["high"])
    public_base_url: str = "http://localhost:8000"
    emergency_hotline: str = "100"
    alert_sender_name: str = "HimSat Engine"

    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str = "alerts@himsat.local"
    smtp_starttls: bool = True

    sms_provider: Literal["none", "sparrow", "twilio"] = "none"
    sparrow_token: str | None = None
    sparrow_from: str | None = None
    sparrow_url: str = "https://api.sparrowsms.com/v2/sms/"
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_from: str | None = None

    webhook_secret: str | None = None

    # API
    admin_api_keys: list[str] = Field(default_factory=list)  # sha256 hex digests of admin keys
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    web_dist_dir: Path = REPO_ROOT / "web" / "dist"

    # scheduler
    schedule_interval_minutes: int = 360
    lookback_days: int = 120

    @field_validator("auto_dispatch_levels", "admin_api_keys", "cors_origins", "llm_languages", mode="before")
    @classmethod
    def _split_csv(cls, v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("["):
                import json

                return json.loads(s)
            return [x.strip() for x in s.split(",") if x.strip()]
        return v

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def products_dir(self) -> Path:
        return self.data_dir / "products"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ---------------------------------------------------------------------------------------------
# Areas of interest
# ---------------------------------------------------------------------------------------------


class AOIConfig(BaseModel):
    id: str
    name: str
    name_ne: str = ""
    bbox: tuple[float, float, float, float]  # lon/lat: west, south, east, north
    exposure_bbox: tuple[float, float, float, float] | None = None  # larger box for downstream routing
    resolution_m: float = 10.0
    tile_size_px: int = 2048
    tile_overlap_px: int = 128
    min_lake_area_m2: float = 5000.0
    min_glacial_lake_elevation_m: float = 3000.0
    watch_min_elevation_m: float = 3300.0  # slopes above this are tracked for instability
    watch_min_slope_deg: float = 20.0
    s1_max_pair_days: int = 37  # pairs up to ~36 days (12/24/36-day baselines)
    code_prefix: str = "HS"
    timezone: str = "Asia/Kathmandu"

    @property
    def routing_bbox(self) -> tuple[float, float, float, float]:
        return self.exposure_bbox or self.bbox


def load_yaml(name: str, settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    path = settings.config_dir / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_aois(settings: Settings | None = None) -> dict[str, AOIConfig]:
    raw = load_yaml("aois.yaml", settings)
    return {a["id"]: AOIConfig(**a) for a in raw.get("aois", [])}


def get_aoi(aoi_id: str, settings: Settings | None = None) -> AOIConfig:
    aois = load_aois(settings)
    if aoi_id not in aois:
        raise KeyError(f"Unknown AOI '{aoi_id}'. Known: {', '.join(sorted(aois)) or '(none)'}")
    return aois[aoi_id]
