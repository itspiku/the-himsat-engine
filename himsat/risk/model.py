"""Explainable hazard × exposure risk model (configuration in ``config/risk.yaml``)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

from himsat.alerts.phrases import reason_text
from himsat.config import load_yaml


@lru_cache
def load_risk_config() -> dict[str, Any]:
    return load_yaml("risk.yaml")


# indicators that measure change over time (vs. static susceptibility such as size or steepness)
DYNAMIC = {"lake_growth_recent_pct", "lake_growth_annual_pct", "turbidity_rise", "barrier_lake", "velocity_ratio",
           "velocity_trend", "velocity_steep_m_day", "new_fractures_m", "sar_change_km2", "landslide_area_km2",
           "insar_ratio", "insar_mm_month"}
ALERT_MIN_DYNAMIC = 0.15


def ramp(x: float, x0: float, x1: float) -> float:
    """Linear 0→1 between x0 and x1 (works for decreasing ramps where x0 > x1)."""
    if x0 == x1:
        return 1.0 if x >= x1 else 0.0
    t = (x - x0) / (x1 - x0)
    return max(0.0, min(1.0, t))


@dataclass
class Indicator:
    code: str
    value: float
    score: float
    weight: float

    @property
    def contribution(self) -> float:
        return self.score * self.weight


@dataclass
class RiskResult:
    level: str
    score: float
    hazard: float
    exposure: float
    confidence: float
    indicators: dict[str, dict] = field(default_factory=dict)
    reasons: list[dict] = field(default_factory=list)
    model_version: str = ""
    needs_confirmation: bool = False
    dynamic: float = 0.0  # hazard from observed *change* only (growth, acceleration, fractures, fresh scars)

    @property
    def alertable(self) -> bool:
        """Alerts need observed change: static susceptibility alone is a map status, not a warning."""
        return self.dynamic >= ALERT_MIN_DYNAMIC

    def to_dict(self) -> dict:
        return asdict(self)


class RiskModel:
    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or load_risk_config()
        self.version = str(self.cfg.get("model_version", "dev"))

    def level_for(self, score: float) -> str:
        lv = self.cfg["levels"]
        if score >= lv["high"]:
            return "high"
        if score >= lv["medium"]:
            return "medium"
        return "low"

    def assess(self, kind: str, features: dict[str, float | None], exposure: float,
               confidence: float) -> RiskResult:
        inds: list[Indicator] = []
        for code, spec in self.cfg["indicators"].items():
            if kind not in spec.get("kinds", []):
                continue
            x = features.get(code)
            if x is None:
                continue
            x0, x1 = spec["x"]
            inds.append(Indicator(code, float(x), ramp(float(x), x0, x1), float(spec["w"])))
        # gates: e.g. a lake's contact with ice or steep walls matters in proportion to its size
        by_code = {i.code: i for i in inds}
        for ind in inds:
            gate = self.cfg["indicators"][ind.code].get("gate")
            if gate:
                g = by_code.get(gate["by"])
                g_score = g.score if g is not None else 0.0
                ind.score *= max(float(gate.get("floor", 0.0)), g_score)
        p_not = 1.0
        for ind in inds:
            p_not *= 1.0 - ind.contribution
        hazard = 1.0 - p_not
        p_dyn = 1.0
        for ind in inds:
            if ind.code in DYNAMIC:
                p_dyn *= 1.0 - ind.contribution
        floor = float(self.cfg.get("exposure_floor", 0.4))
        score = hazard * (floor + (1.0 - floor) * exposure)
        level = self.level_for(score)
        dynamic = 1.0 - p_dyn
        if level == "high" and dynamic < ALERT_MIN_DYNAMIC:
            level = "medium"  # static susceptibility alone never exceeds "potentially dangerous"
        needs_conf = False
        if level == "high" and confidence < float(self.cfg.get("min_confidence_for_high", 0.3)):
            level, needs_conf = "medium", True
        reasons = [
            {"code": i.code, "value": round(i.value, 4), "contribution": round(i.contribution, 3),
             "text_en": reason_text(i.code, i.value, "en"), "text_ne": reason_text(i.code, i.value, "ne")}
            for i in sorted(inds, key=lambda i: -i.contribution) if i.contribution >= 0.05
        ]
        return RiskResult(
            level=level, score=round(score, 4), hazard=round(hazard, 4), exposure=round(exposure, 4),
            confidence=round(confidence, 3),
            indicators={i.code: {"value": round(i.value, 4), "score": round(i.score, 3), "weight": i.weight}
                        for i in inds},
            reasons=reasons, model_version=self.version, needs_confirmation=needs_conf,
            dynamic=round(dynamic, 4),
        )
