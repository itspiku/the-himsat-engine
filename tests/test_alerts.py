import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime

import pytest

from himsat.alerts import phrases as P
from himsat.alerts.compose import compose, validate
from himsat.alerts.facts import allowed_numbers, build_facts, numbers_in
from himsat.alerts.llm import LLMError, parse_json_object
from himsat.alerts.templates import SMS_MAX, render

REASONS = [{"code": "velocity_ratio", "text_en": P.reason_text("velocity_ratio", 6.2, "en"),
            "text_ne": P.reason_text("velocity_ratio", 6.2, "ne")},
           {"code": "velocity_trend", "text_en": P.reason_text("velocity_trend", 3, "en"),
            "text_ne": P.reason_text("velocity_trend", 3, "ne")}]
EXPOSURES = [
    {"kind": "border_crossing", "name": "Rasuwagadhi border crossing", "name_ne": "रसुवागढी नाका",
     "travel_time_min": 15.1, "path_distance_km": 22.8},
    {"kind": "settlement", "name": "Shyaphru Bensi", "name_ne": "स्याफ्रु बेसी", "travel_time_min": 25.3,
     "path_distance_km": 37.9},
    {"kind": "hydropower", "name": "Upper Trishuli 3A", "name_ne": "", "travel_time_min": 42.3,
     "path_distance_km": 63.4},
]


@pytest.fixture
def facts(fake_site):
    return build_facts(alert_kind="alert", level="high", previous_level=None, site=fake_site, reasons=REASONS,
                       exposures=EXPOSURES, evidence_time=datetime(2026, 8, 24, 0, 18, tzinfo=UTC),
                       needs_confirmation=False, hotline="100", base_url="https://himsat.example.org")


def test_nepali_digits():
    assert P.ne_digits("2026-08-24 100") == "२०२६-०८-२४ १००"
    assert P.en_digits("१५ मिनेट") == "15 मिनेट"
    assert P.fmt_num(6.2, "ne") == "६.२" and P.fmt_num(0.123, "en") == "0.12" and P.fmt_num(37.9) == "38"


def test_template_is_valid_and_complete(facts):
    t = render(facts)
    assert validate(t, facts) == []
    assert "रसुवागढी नाका" in t["body_ne"] and "Rasuwagadhi" in t["body_en"]
    assert "१५ मिनेट" in t["body_ne"] and "15 minutes" in t["body_en"]
    assert "उच्च जोखिम" in t["title_ne"] and t["title_en"].startswith("HIGH RISK")
    assert len(t["sms_en"]) <= SMS_MAX["en"] and len(t["sms_ne"]) <= SMS_MAX["ne"]
    assert P.ACTION["high"]["ne"] in t["body_ne"]


def test_validator_rejects_hallucinated_numbers_and_missing_nepali(facts):
    t = render(facts)
    bad = dict(t, body_en=t["body_en"] + "\nAbout 2,000 people live in the area.")
    assert any("numbers not in FACTS" in i for i in validate(bad, facts))
    bad2 = dict(t, body_ne="High risk at the slope. Move to high ground. Emergency 100.")
    assert any("not mainly Nepali" in i for i in validate(bad2, facts))
    bad3 = dict(t, body_en=t["body_en"].replace("Rasuwagadhi", "the border"))
    assert any("first place at risk" in i for i in validate(bad3, facts))


def test_allowed_numbers_cover_devanagari(facts):
    assert numbers_in("बाढी १५ मिनेटमा") <= allowed_numbers(facts)
    assert "2000" not in allowed_numbers(facts)


class FakeLLM:
    name = "llm:fake"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def chat_json(self, system, user, max_tokens=1800, schema=None):
        self.calls += 1
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_compose_accepts_valid_llm_draft(facts):
    draft = render(facts)
    draft["body_en"] = draft["body_en"].replace("What the satellites show", "Satellite observations")
    c = compose(facts, FakeLLM([draft]), languages=("en", "ne"))
    assert c.generator == "llm:fake" and c.validation["status"] == "accepted"


def test_nepali_stays_on_template_by_default(facts):
    draft = render(facts)
    draft["body_ne"] = draft["body_ne"].replace("के गर्ने", "के गर्नुपर्छ")
    c = compose(facts, FakeLLM([draft]))
    assert c.texts["body_ne"] == render(facts)["body_ne"] and c.generator.endswith("+template")


def test_compose_retries_then_falls_back(facts):
    t = render(facts)
    bad = dict(t, body_en=t["body_en"] + " 9999 deaths expected.")
    llm = FakeLLM([bad, bad])
    c = compose(facts, llm)
    assert llm.calls == 2 and c.generator == "template" and c.texts == render(facts)
    assert c.validation["status"] == "fallback"


def test_compose_survives_llm_outage(facts):
    c = compose(facts, FakeLLM([LLMError("connection refused")]))
    assert c.generator == "template"


def test_parse_json_object_variants():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('<think>hmm</think> Here: {"a": 2} done') == {"a": 2}
    with pytest.raises(LLMError):
        parse_json_object("no json here")


def test_cap_export_is_valid_xml(facts):
    from himsat.alerts.cap import CAP_NS, alert_to_cap

    class A:
        uid = "0b6c1a7e-0000-4000-8000-000000000001"
        level, kind, status = "high", "alert", "dispatched"
        issued_at = created_at = dispatched_at = datetime(2026, 8, 24, 6, tzinfo=UTC)
        approved_at = None
        expires_at = datetime(2026, 8, 27, 6, tzinfo=UTC)
        facts = {"score": 0.71}

    t = render(facts)
    a = A()
    for k, v in t.items():
        setattr(a, k, v)
    xml = alert_to_cap(a, type("S", (), {"name": "Slope", "code": "LHD-S0001", "lat": 28.29, "lon": 85.53})(),
                       "HimSat", "https://himsat.example.org")
    root = ET.fromstring(xml.encode())
    ns = {"c": CAP_NS}
    assert root.find("c:msgType", ns).text == "Alert"
    infos = root.findall("c:info", ns)
    assert [i.find("c:language", ns).text for i in infos] == ["ne-NP", "en-US"]
    assert infos[0].find("c:severity", ns).text == "Severe"
    json.dumps(t, ensure_ascii=False)
