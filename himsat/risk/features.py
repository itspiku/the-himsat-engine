"""Turn a site's observation history into risk-model indicators, as of a given time.

Everything is computed only from observations at or before ``as_of``. Live monitoring and
hindcast replay therefore produce identical results for the same evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median
from typing import Any

from himsat.detect.cells import MIN_V, RATIO_FLOOR, weighted_mean


@dataclass
class Obs:
    kind: str
    sensor: str
    observed_at: datetime
    values: dict[str, Any]
    quality: float = 1.0


def _in_window(o: Obs, as_of: datetime, lo_days: float, hi_days: float) -> bool:
    age = (as_of - o.observed_at).total_seconds() / 86400.0
    return lo_days <= age <= hi_days


def _med(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(median(xs)) if xs else None


def lake_features(obs: list[Obs], as_of: datetime, cfg: dict, attrs: dict) -> tuple[dict, float]:
    w = cfg["windows"]
    qmin = cfg["quality"]["min_lake_quality"]
    good = [o for o in obs if o.kind == "lake_area" and o.observed_at <= as_of and o.quality >= qmin
            and o.values.get("area_m2") is not None]
    feats: dict[str, float | None] = {}
    feats["steep_walls_deg"] = attrs.get("max_surrounding_slope_deg")
    feats["glacier_contact_m"] = attrs.get("glacier_distance_m")
    if attrs.get("barrier"):
        feats["barrier_lake"] = 1.0
    if not good:
        a = attrs.get("area_m2")
        feats["lake_area_km2"] = a / 1e6 if a else None
        return feats, 0.1

    by_sensor: dict[str, dict[str, float | None]] = {}
    for sensor in ("S2", "S1"):
        so = [o for o in good if o.sensor == sensor]
        if not so:
            continue
        recent = [o for o in so if _in_window(o, as_of, 0, w["recent_days"])]
        cmp_ = [o for o in so if _in_window(o, as_of, *w["compare_days"])]
        ann = [o for o in so if _in_window(o, as_of, *w["annual_days"])]
        by_sensor[sensor] = {
            "now": _med([o.values["area_m2"] for o in recent]),
            "cmp": _med([o.values["area_m2"] for o in cmp_]),
            "ann": _med([o.values["area_m2"] for o in ann]),
            "latest": max(so, key=lambda o: o.observed_at).values["area_m2"],
            "latest_at": max(so, key=lambda o: o.observed_at).observed_at,
        }

    def growth(key: str) -> float | None:
        for sensor in ("S2", "S1"):  # prefer optical, fall back to radar (same-sensor comparisons only)
            s = by_sensor.get(sensor)
            if s and s["now"] and s[key]:
                return 100.0 * (s["now"] - s[key]) / s[key]
        return None

    feats["lake_growth_recent_pct"] = growth("cmp")
    feats["lake_growth_annual_pct"] = growth("ann")
    latest_s2 = by_sensor.get("S2")
    latest_any = max(by_sensor.values(), key=lambda s: s["latest_at"])
    area = (latest_s2 or latest_any)["now"] or (latest_s2 or latest_any)["latest"]
    feats["lake_area_km2"] = area / 1e6 if area else None

    s2 = [o for o in good if o.sensor == "S2"]
    gd = _med([o.values.get("glacier_distance_m") for o in s2 if _in_window(o, as_of, 0, 400)])
    if gd is not None:
        feats["glacier_contact_m"] = gd
    t_now = _med([o.values.get("turbidity") for o in s2 if _in_window(o, as_of, 0, w["recent_days"])])
    t_cmp = _med([o.values.get("turbidity") for o in s2 if _in_window(o, as_of, *w["compare_days"])])
    if t_now is not None and t_cmp:
        feats["turbidity_rise"] = (t_now - t_cmp) / abs(t_cmp)

    _event_features(obs, as_of, w, feats)
    recent_all = [o for o in good if _in_window(o, as_of, 0, w["recent_days"])]
    has_cmp = feats["lake_growth_recent_pct"] is not None
    q = sum(o.quality for o in recent_all) / len(recent_all) if recent_all else 0.0
    conf = 0.2 + (0.35 if recent_all else 0.0) + (0.25 if has_cmp else 0.0) + 0.2 * q
    return feats, min(conf, 1.0)


def _event_features(obs: list[Obs], as_of: datetime, w: dict, feats: dict) -> None:
    ev = w["event_days"]
    sar = [o for o in obs if o.kind == "sar_change" and _in_window(o, as_of, 0, ev)]
    if sar:
        feats["sar_change_km2"] = sum(o.values.get("area_m2", 0.0) for o in sar) / 1e6
    # optical "new crack" length is noisy (illumination, viewing geometry, fresh snow): it counts only
    # when it is persistently anomalous against the site's own history
    surf = [o for o in obs if o.kind == "surface_change" and o.observed_at <= as_of
            and o.values.get("observed_fraction", 1.0) >= 0.5]
    recent = [o.values.get("new_fracture_length_m", 0.0) for o in surf if _in_window(o, as_of, 0, ev)]
    hist = [o.values.get("new_fracture_length_m", 0.0) for o in surf if _in_window(o, as_of, ev, 365)]
    if len(hist) >= 4 and len(recent) >= 2:
        med = float(median(hist))
        mad = float(median([abs(h - med) for h in hist]))
        scale = 1.4826 * mad + 50.0  # metres; floor avoids divide-by-zero on very stable sites
        anomalous = [v for v in recent if (v - med) / scale > 3.0]
        if len(anomalous) >= 2:
            feats["new_fractures_m"] = max(anomalous) - med


def ice_features(obs: list[Obs], as_of: datetime, cfg: dict, attrs: dict) -> tuple[dict, float]:
    w = cfg["windows"]
    feats: dict[str, float | None] = {"source_slope_deg": attrs.get("mean_slope_deg")}

    def v_of(o: Obs) -> float | None:
        # the downslope component is unbiased under noise; speed magnitude is only a fallback
        return o.values.get("v_down_median", o.values.get("median_m_per_day"))

    def se_of(o: Obs) -> float:
        return float(o.values.get("v_down_se") or o.values.get("noise_m_per_day") or 0.05)

    vel = sorted(
        [o for o in obs if o.kind == "velocity" and o.observed_at <= as_of and v_of(o) is not None
         and o.values.get("coverage", 0) >= 0.2 and o.values.get("n_points", 0) >= 4],
        key=lambda o: o.observed_at)
    _event_features(obs, as_of, w, feats)
    if not vel:
        return feats, 0.15
    recent = [o for o in vel if _in_window(o, as_of, 0, w["recent_days"])]
    lo, hi = w["velocity_baseline_days"]
    base = [o for o in vel if _in_window(o, as_of, lo, hi)]
    wr = weighted_mean([v_of(o) for o in recent], [se_of(o) for o in recent]) if recent else None
    wb = (weighted_mean([v_of(o) for o in base], [se_of(o) for o in base])
          if len(base) >= w["velocity_min_baseline_pairs"] else None)
    if wr is not None:
        v_now, se_now = wr
        feats["velocity_m_day"] = v_now
        if (attrs.get("mean_slope_deg") or 0) >= 25 and v_now > 3 * se_now:
            feats["velocity_steep_m_day"] = v_now
        if wb is not None:
            v_base, se_base = wb
            z = (v_now - v_base) / math.hypot(se_now, se_base)
            feats["velocity_z"] = z
            # a speed-up only counts when it is statistically significant and physically material
            significant = z >= w.get("velocity_min_z", 3.0) and v_now >= MIN_V
            feats["velocity_ratio"] = v_now / max(v_base, RATIO_FLOOR) if significant else 1.0
            # sustained acceleration: consecutive latest pairs individually above the baseline
            run = 0
            for o in reversed(vel):
                if v_of(o) - v_base > 2 * se_of(o):
                    run += 1
                else:
                    break
            feats["velocity_trend"] = float(run) if significant else 0.0
    cov = sum(o.values.get("coverage", 0) for o in recent) / len(recent) if recent else 0.0
    conf = 0.2 + (0.3 if recent else 0.0) + (0.3 if wb is not None else 0.0) + 0.2 * cov
    return feats, min(conf, 1.0)


def landslide_features(obs: list[Obs], as_of: datetime, cfg: dict, attrs: dict) -> tuple[dict, float]:
    """Fresh, cross-orbit-confirmed mass movement: the hazard decays once no new change is seen."""
    recent = [o for o in obs if o.kind == "sar_change" and _in_window(o, as_of, 0, cfg["windows"]["event_days"])]
    feats: dict[str, float | None] = {}
    if recent:
        feats["landslide_area_km2"] = max(o.values.get("area_m2", 0.0) for o in recent) / 1e6
    conf = 0.3 + 0.2 * min(len(recent), 2) + 0.1 * (max((o.quality for o in recent), default=0) >= 0.8)
    return feats, min(conf, 0.8)


def site_features(kind: str, obs: list[Obs], as_of: datetime, cfg: dict, attrs: dict) -> tuple[dict, float]:
    if kind in ("glacial_lake", "barrier_lake"):
        return lake_features(obs, as_of, cfg, attrs)
    if kind in ("glacier", "slope"):
        return ice_features(obs, as_of, cfg, attrs)
    if kind == "landslide":
        return landslide_features(obs, as_of, cfg, attrs)
    return {}, 0.0


def recent_window_start(as_of: datetime, days: int) -> datetime:
    return as_of - timedelta(days=days)
