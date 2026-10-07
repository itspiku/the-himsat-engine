"""Hindcast: replay the satellite archive through the unchanged live pipeline and report what
would have been seen, and when, before a real event.

Hindcasts use a scratch database under ``data/hindcasts/<name>/``. Alerts are generated but
never dispatched, and they are timestamped at acquisition time + data latency. The report
separates evidence available *before* the event from post-event detections.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from himsat.config import get_aoi, get_settings
from himsat.db.models import Alert, ChangeEvent, Observation, RiskAssessment, Scene, Site
from himsat.db.session import session_scope
from himsat.geo.geometry import from_geojson
from himsat.pipeline.monitor import CycleOptions, run_cycle

log = logging.getLogger(__name__)
NPT = timedelta(hours=5, minutes=45)


def hindcast_dir(name: str) -> Path:
    return get_settings().data_dir / "hindcasts" / name


def run_hindcast(aoi_id: str, start: datetime, end: datetime, event_time: datetime | None, *, name: str | None = None,
                 sensors: tuple[str, ...] = ("S1", "S2"), fresh: bool = False, progress=None,
                 event_lonlat: tuple[float, float] | None = None, report_only: bool = False) -> Path:
    name = name or f"{aoi_id}-{end:%Y%m%d}"
    d = hindcast_dir(name)
    if fresh and d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    db_url = f"sqlite:///{(d / 'himsat.db').as_posix()}"
    settings = get_settings().model_copy(update={"dispatch_enabled": False})
    # products (composites, velocity fields) are kept per hindcast, independent of live monitoring;
    # the raster download cache is shared
    opts = CycleOptions(start=start, end=end, dispatch=False, sensors=sensors, hindcast=True, progress=progress,
                        products_dir=d / "products")
    if not report_only:
        run_cycle(aoi_id, opts, db_url=db_url, settings=settings)
    report = build_report(aoi_id, db_url, start, end, event_time, event_lonlat)
    return save_report(d, report)


def _rounded(x, nd: int = 5):
    """Floats to 5 decimals (~1 m in degrees, 0.01 mm/day in velocity): a much smaller report."""
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _rounded(v, nd) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_rounded(v, nd) for v in x]
    return x


def save_report(d: Path, report: dict) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(_rounded(report), ensure_ascii=False, separators=(",", ":"), default=str),
                                   encoding="utf-8")
    (d / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return d / "report.md"


def reassess(aoi_id: str, db_url: str, *, hindcast: bool = True, data_latency: timedelta = timedelta(hours=6),
             progress=None) -> dict:
    """Re-run the risk model and alert policy over stored observations, without reprocessing imagery.

    Use this after changing ``config/risk.yaml`` or the feature logic. Assessments and alerts are
    rebuilt in time order, exactly as they would have been issued. Only safe for hindcast/scratch
    databases (``hindcast=True``) or before any alert has been dispatched.
    """
    from sqlalchemy import delete

    from himsat.alerts.llm import LLMClient
    from himsat.db.models import Delivery, RiskAssessment
    from himsat.pipeline.assess import Assessor
    from himsat.pipeline.context import AOIContext

    settings = get_settings()
    if hindcast:
        settings = settings.model_copy(update={"dispatch_enabled": False})
    cfg = get_aoi(aoi_id)
    ctx = AOIContext(cfg, settings)
    assessor = Assessor(ctx, settings, LLMClient.from_settings(settings), dispatch=not hindcast)
    with session_scope(db_url) as s:
        if not hindcast and s.scalar(select(Alert.id).where(Alert.status == "dispatched").limit(1)):
            raise RuntimeError("refusing to rebuild alerts in a database with dispatched alerts")
        s.execute(delete(Delivery))
        s.execute(delete(Alert))
        s.execute(delete(RiskAssessment))
        for site in s.scalars(select(Site).where(Site.aoi_id == aoi_id)):
            site.latest_level, site.latest_score, site.latest_assessed_at = "unknown", None, None
        times = sorted({t for (t,) in s.execute(select(Scene.acquired_at).where(
            Scene.aoi_id == aoi_id, Scene.status == "processed"))})
    n_alerts = 0
    for t in times:
        with session_scope(db_url) as s:
            ids = sorted({sid for (sid,) in s.execute(select(Observation.site_id).where(Observation.observed_at == t))})
            for sid in ids:
                site = s.get(Site, sid)
                if site is None or site.aoi_id != aoi_id:
                    continue
                _, alert = assessor.assess(s, site, as_of=t, now=t + data_latency if hindcast else None)
                n_alerts += alert is not None
        if progress:
            progress(f"{t:%Y-%m-%d %H:%M} assessed")
    return {"times": len(times), "alerts": n_alerts}


def _simplified(geojson: str | None, tol_deg: float = 1e-4) -> dict | None:
    """Flow paths are traced at DEM resolution; ~10 m simplification is invisible on the map."""
    g = from_geojson(geojson)
    if g is None:
        return None
    from shapely.geometry import mapping

    return mapping(g.simplify(tol_deg, preserve_topology=False))


def _km(lon1, lat1, lon2, lat2) -> float:
    return math.hypot((lon1 - lon2) * 111.32 * math.cos(math.radians(lat1)), (lat1 - lat2) * 110.57)


def build_report(aoi_id: str, db_url: str, start: datetime, end: datetime, event_time: datetime | None,
                 event_lonlat: tuple[float, float] | None = None, focus_km: float = 5.0) -> dict:
    cfg = get_aoi(aoi_id)
    with session_scope(db_url) as s:
        sites = s.scalars(select(Site).where(Site.aoi_id == aoi_id)).all()
        scenes = s.scalars(select(Scene).where(Scene.aoi_id == aoi_id, Scene.acquired_at >= start,
                                               Scene.acquired_at <= end).order_by(Scene.acquired_at)).all()
        # the database may hold earlier seasons (baselines); the report covers the replay window only
        alerts = s.scalars(select(Alert).where(Alert.issued_at >= start, Alert.issued_at <= end)
                           .order_by(Alert.created_at)).all()
        events = s.scalars(select(ChangeEvent).where(ChangeEvent.aoi_id == aoi_id, ChangeEvent.detected_at >= start,
                                                     ChangeEvent.detected_at <= end)
                           .order_by(ChangeEvent.detected_at)).all()
        site_by_id = {x.id: x for x in sites}

        def lead(t: datetime) -> float | None:
            return round((event_time - t).total_seconds() / 3600.0, 1) if event_time else None

        def rel(site: Site) -> float | None:
            if not event_lonlat:
                return None
            g = from_geojson(site.geom)
            if g is None:
                return _km(site.lon, site.lat, *event_lonlat)
            from shapely.geometry import Point

            return round(g.distance(Point(*event_lonlat)) * 111.0, 2)

        site_rows = []
        for st in sites:
            ras = s.scalars(select(RiskAssessment).where(RiskAssessment.site_id == st.id,
                                                         RiskAssessment.assessed_at >= start,
                                                         RiskAssessment.assessed_at <= end)
                            .order_by(RiskAssessment.assessed_at)).all()
            if not ras:
                continue
            obs = s.scalars(select(Observation).where(Observation.site_id == st.id, Observation.observed_at >= start,
                                                      Observation.observed_at <= end)
                            .order_by(Observation.observed_at)).all()
            pre = [r for r in ras if not event_time or r.assessed_at < event_time]
            max_pre = max(pre, key=lambda r: r.score) if pre else None
            site_rows.append({
                "id": st.id, "code": st.code, "name": st.name, "name_ne": st.name_ne, "kind": st.kind,
                "lon": st.lon, "lat": st.lat, "elevation_m": st.elevation_m, "source": st.source,
                "first_seen": st.first_seen, "distance_to_event_km": rel(st),
                "geometry": json.loads(st.geom) if st.geom else None,
                "flow_path": _simplified(st.flow_path),
                "max_pre_event": ({"level": max_pre.level, "score": max_pre.score, "at": max_pre.assessed_at,
                                   "reasons": max_pre.reasons} if max_pre else None),
                "timeline": [{"at": r.assessed_at, "level": r.level, "score": r.score, "hazard": r.hazard,
                              "exposure": r.exposure, "confidence": r.confidence,
                              "reasons": [x["code"] for x in r.reasons]} for r in ras],
                "observations": [{"at": o.observed_at, "kind": o.kind, "sensor": o.sensor, "quality": o.quality,
                                  "values": {k: v for k, v in (o.values or {}).items()
                                             if isinstance(v, (int, float)) and v is not None}} for o in obs],
                "exposure_top": _exposure_top(s, st.id),
            })
        alert_rows = [{
            "id": a.id, "site": site_by_id[a.site_id].code if a.site_id in site_by_id else a.site_id,
            "level": a.level, "kind": a.kind, "evidence_at": a.issued_at, "available_at": a.created_at,
            "lead_time_h": lead(a.created_at), "pre_event": (a.created_at < event_time) if event_time else None,
            "generator": a.generator, "title_en": a.title_en, "title_ne": a.title_ne, "body_en": a.body_en,
            "body_ne": a.body_ne, "sms_en": a.sms_en, "sms_ne": a.sms_ne,
        } for a in alerts]
        ev_rows = [{"kind": e.kind, "sensor": e.sensor, "at": e.detected_at, "pre_at": e.pre_at, "lon": e.lon,
                    "lat": e.lat, "area_km2": round(e.area_m2 / 1e6, 4), "confidence": e.confidence,
                    "distance_to_event_km": round(_km(e.lon, e.lat, *event_lonlat), 2) if event_lonlat else None,
                    "attrs": {k: v for k, v in (e.attrs or {}).items() if isinstance(v, (int, float, str))}}
                   for e in events if e.area_m2 >= 50_000]
        scene_stats = {"S1": sum(1 for x in scenes if x.sensor == "S1" and x.status == "processed"),
                       "S2": sum(1 for x in scenes if x.sensor == "S2" and x.status == "processed"),
                       "skipped": sum(1 for x in scenes if x.status == "skipped"),
                       "failed": sum(1 for x in scenes if x.status == "failed")}
    focus = sorted([r for r in site_rows if r["distance_to_event_km"] is not None
                    and r["distance_to_event_km"] <= focus_km], key=lambda r: r["distance_to_event_km"])
    return {
        "aoi": cfg.model_dump(mode="json"), "period": [start, end], "event_time": event_time,
        "event_lonlat": event_lonlat, "generated_at": datetime.now(UTC), "scenes": scene_stats,
        "sites": site_rows, "focus_sites": [r["code"] for r in focus], "alerts": alert_rows, "change_events": ev_rows,
        "summary": summarize(site_rows, alert_rows, ev_rows, event_time, {r["code"] for r in focus}),
    }


def summarize(sites: list[dict], alerts: list[dict], events: list[dict], event_time, focus: set[str],
              source_km: float = 2.0, warning_days: float = 30.0) -> dict:
    """Headline evaluation.

    A *warning* is a MEDIUM/HIGH alert for a site within ``source_km`` of the source issued in the
    ``warning_days`` before the event. Every other pre-event alert counts as a potential false alarm.
    """
    if event_time is None:
        return {"alerts": len(alerts)}
    near = {s["code"] for s in sites if s["distance_to_event_km"] is not None and s["distance_to_event_km"] <= source_km}
    pre = [a for a in alerts if a["pre_event"] and a["kind"] != "all_clear"]
    at_source = [a for a in pre if a["site"] in near and a["lead_time_h"] <= warning_days * 24]
    elsewhere = [a for a in pre if a not in at_source]
    first = min(at_source, key=lambda a: a["available_at"]) if at_source else None
    src_levels = [s["max_pre_event"]["level"] for s in sites if s["code"] in near and s["max_pre_event"]]
    rank = {"low": 0, "medium": 1, "high": 2}
    post_mm = sorted([e for e in events if e["kind"] == "mass_movement" and e["confidence"] >= 0.7
                      and (e["distance_to_event_km"] or 99) <= 5 and _as_dt(e["at"]) > event_time],
                     key=lambda e: e["at"])
    return {
        "warned_before_event": bool(at_source),
        "first_warning_lead_time_h": first["lead_time_h"] if first else None,
        "first_warning_level": first["level"] if first else None,
        "max_pre_event_level_near_source": max(src_levels, key=lambda lv: rank.get(lv, -1)) if src_levels else None,
        "pre_event_alerts_elsewhere": len(elsewhere),
        "pre_event_alerts_elsewhere_by_level": {lv: sum(a["level"] == lv for a in elsewhere) for lv in ("medium", "high")},
        "post_event_detection_h": round((_as_dt(post_mm[0]["at"]) - event_time).total_seconds() / 3600, 1)
        if post_mm else None,
    }


def _as_dt(t):
    return t if isinstance(t, datetime) else datetime.fromisoformat(str(t))


def _exposure_top(session, site_id: int, n: int = 8) -> list[dict]:
    from himsat.alerts.service import site_exposures

    return [{k: e[k] for k in ("kind", "name", "name_ne", "travel_time_min", "path_distance_km")}
            for e in site_exposures(session, site_id) if e["name"]][:n]


def _fmt_t(t) -> str:
    if t is None:
        return "–"
    if isinstance(t, str):
        t = datetime.fromisoformat(t)
    return (t.astimezone(UTC) + NPT).strftime("%Y-%m-%d %H:%M NPT")


def render_markdown(r: dict) -> str:
    L = []
    aoi = r["aoi"]
    L.append(f"# HimSat hindcast — {aoi['name']}\n")
    L.append(f"Replay window: {_fmt_t(r['period'][0])} → {_fmt_t(r['period'][1])}  ")
    if r["event_time"]:
        L.append(f"Real event: **{_fmt_t(r['event_time'])}**" + (f" at {r['event_lonlat'][1]:.4f}N {r['event_lonlat'][0]:.4f}E"
                                                                 if r["event_lonlat"] else "") + "  ")
    sc = r["scenes"]
    L.append(f"Acquisitions processed: {sc['S1']} Sentinel-1, {sc['S2']} Sentinel-2 "
             f"({sc['skipped']} skipped as cloudy/no coverage, {sc['failed']} failed)\n")
    pre_alerts = [a for a in r["alerts"] if a["pre_event"]]
    post_alerts = [a for a in r["alerts"] if a["pre_event"] is False]
    sm = r.get("summary") or {}
    if r["event_time"] and sm:
        L.append("## Summary\n")
        if sm["warned_before_event"]:
            L.append(f"- **Warned before the event:** yes, {sm['first_warning_level'].upper()} alert "
                     f"{sm['first_warning_lead_time_h']:.1f} h ahead for a site within 2 km of the source (30-day window).")
        else:
            L.append(f"- **Warned before the event:** no (no MEDIUM/HIGH alert within 2 km of the source in the 30 days before). Highest pre-event status within 2 km: "
                     f"{(sm['max_pre_event_level_near_source'] or 'not assessed').upper()}.")
        el = sm["pre_event_alerts_elsewhere_by_level"]
        L.append(f"- **Pre-event alerts elsewhere** (potential false alarms; MEDIUM alerts are held for review): "
                 f"{sm['pre_event_alerts_elsewhere']} ({el['high']} HIGH, {el['medium']} MEDIUM).")
        if sm["post_event_detection_h"] is not None:
            L.append(f"- **Event detected after it happened:** confident radar mass-movement detection near the "
                     f"source {sm['post_event_detection_h']:.0f} h after the event.")
        L.append("")
    L.append("## Alerts the system would have issued\n")
    if not r["alerts"]:
        L.append("None.\n")
    else:
        L.append("| available | lead time | site | level | kind | generator |\n|---|---|---|---|---|---|")
        for a in r["alerts"]:
            h = a["lead_time_h"]
            lt = "–" if h is None else (f"**{h:.1f} h before**" if h >= 0 else f"{-h:.1f} h after")
            L.append(f"| {_fmt_t(a['available_at'])} | {lt} | {a['site']} | {a['level'].upper()} | {a['kind']} | {a['generator']} |")
        L.append("")
    L.append(f"Pre-event alerts: **{len(pre_alerts)}**; post-event alerts: {len(post_alerts)}.\n")
    if r["focus_sites"]:
        L.append("## Sites near the event source\n")
        by = {s["code"]: s for s in r["sites"]}
        for code in r["focus_sites"]:
            s = by[code]
            mp = s["max_pre_event"]
            L.append(f"### {s['code']} — {s['name']} ({s['kind']}, {s['distance_to_event_km']} km from source)\n")
            if mp:
                L.append(f"Highest pre-event assessment: **{mp['level'].upper()}** (score {mp['score']:.2f}) at {_fmt_t(mp['at'])}")
                for x in mp["reasons"]:
                    L.append(f"- {x['text_en']}")
            vel = [o for o in s["observations"] if o["kind"] == "velocity"]
            if vel:
                L.append("\n| pair end | downslope v (m/day) | ±SE | coverage |\n|---|---|---|---|")
                for o in vel[-14:]:
                    v = o["values"]
                    L.append(f"| {_fmt_t(o['at'])} | {v.get('v_down_median', float('nan')):+.3f} | "
                             f"{v.get('v_down_se') or float('nan'):.3f} | {v.get('coverage', 0):.2f} |")
            L.append("")
    big = sorted([e for e in r["change_events"] if e["kind"] == "mass_movement"], key=lambda e: -e["area_km2"])[:10]
    if big:
        L.append("## Largest mass-movement detections (Sentinel-1 backscatter change)\n")
        L.append("| detected | pre-image | location | area km² | distance to source km | confidence |\n|---|---|---|---|---|---|")
        for e in big:
            L.append(f"| {_fmt_t(e['at'])} | {_fmt_t(e['pre_at'])} | {e['lat']:.4f}N {e['lon']:.4f}E | {e['area_km2']:.3f} | "
                     f"{e['distance_to_event_km'] if e['distance_to_event_km'] is not None else '–'} | {e['confidence']:.2f} |")
        L.append("")
    first = next((a for a in r["alerts"] if a["level"] in ("medium", "high")), None)
    if first:
        L.append("## Example alert text\n")
        L.append(f"**{first['title_ne']}**\n\n```\n{first['body_ne']}\n```\n")
        L.append(f"**{first['title_en']}**\n\n```\n{first['body_en']}\n```\n")
    return "\n".join(L) + "\n"
