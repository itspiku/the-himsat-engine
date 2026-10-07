from datetime import UTC, datetime, timedelta

from himsat.pipeline.hindcast import render_markdown, summarize

EVENT = datetime(2026, 8, 26, 2, 0, tzinfo=UTC)


def _site(code, km, level="medium"):
    return {"code": code, "distance_to_event_km": km,
            "max_pre_event": {"level": level, "score": 0.5, "at": EVENT, "reasons": []} if level else None}


def _alert(site, hours_before, level="medium", kind="alert"):
    at = EVENT - timedelta(hours=hours_before)
    return {"id": 1, "site": site, "level": level, "kind": kind, "evidence_at": at, "available_at": at,
            "lead_time_h": hours_before, "pre_event": hours_before > 0, "title_en": "", "title_ne": "",
            "body_en": "", "body_ne": "", "generator": "template"}


def test_summary_separates_warning_from_false_alarms():
    sites = [_site("G1", 0.8), _site("G2", 12.0, "high"), _site("L1", 1.5, None)]
    alerts = [_alert("G1", 44.6), _alert("G1", 900),  # second one: too early to count as this warning
              _alert("G2", 100, "high"), _alert("G2", -10, "high"), _alert("G1", 5, kind="all_clear")]
    events = [{"kind": "mass_movement", "at": EVENT + timedelta(hours=26), "confidence": 0.9, "distance_to_event_km": 1.0},
              {"kind": "mass_movement", "at": EVENT + timedelta(hours=2), "confidence": 0.5, "distance_to_event_km": 1.0}]
    s = summarize(sites, alerts, events, EVENT, {"G1"})
    assert s["warned_before_event"] and s["first_warning_lead_time_h"] == 44.6 and s["first_warning_level"] == "medium"
    assert s["max_pre_event_level_near_source"] == "medium"
    assert s["pre_event_alerts_elsewhere"] == 2
    assert s["pre_event_alerts_elsewhere_by_level"] == {"medium": 1, "high": 1}
    assert s["post_event_detection_h"] == 26.0


def test_summary_reports_a_miss():
    s = summarize([_site("G1", 0.8, "low")], [_alert("G9", 30)], [], EVENT, {"G1"})
    assert not s["warned_before_event"] and s["first_warning_lead_time_h"] is None
    assert s["max_pre_event_level_near_source"] == "low" and s["post_event_detection_h"] is None


def test_summary_without_event_counts_alerts():
    assert summarize([], [_alert("G1", 5)], [], None, set()) == {"alerts": 1}


def test_markdown_states_the_outcome():
    sites = [{**_site("G1", 0.8), "name": "Glacier", "kind": "glacier", "observations": []}]
    alerts = [_alert("G1", 44.6)]
    r = {"aoi": {"name": "Test"}, "period": [EVENT - timedelta(days=60), EVENT + timedelta(days=5)],
         "event_time": EVENT, "event_lonlat": [85.5, 28.3], "scenes": {"S1": 10, "S2": 3, "skipped": 1, "failed": 0},
         "sites": sites, "focus_sites": ["G1"], "alerts": alerts, "change_events": [],
         "summary": summarize(sites, alerts, [], EVENT, {"G1"})}
    md = render_markdown(r)
    assert "Warned before the event:** yes, MEDIUM alert 44.6 h ahead" in md
    assert "Pre-event alerts: **1**" in md
