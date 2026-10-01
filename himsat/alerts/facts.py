"""The structured fact sheet behind every alert.

Both the deterministic templates and the LLM work *only* from these facts. The validator also
derives from them the set of numbers and place names an alert may mention.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta, timezone

from himsat.alerts import phrases as P

NPT = timezone(timedelta(hours=5, minutes=45), "NPT")


@dataclass
class ExposedPlace:
    name_en: str
    name_ne: str
    kind: str
    travel_time_min: float
    distance_km: float


@dataclass
class AlertFacts:
    alert_kind: str  # alert | update | all_clear
    level: str
    previous_level: str | None
    site_code: str
    site_name_en: str
    site_name_ne: str
    site_kind: str
    lat: float
    lon: float
    elevation_m: float | None
    evidence_time_utc: str
    evidence_time_npt: str
    reasons: list[dict] = field(default_factory=list)  # {code, text_en, text_ne}
    exposed: list[ExposedPlace] = field(default_factory=list)
    needs_confirmation: bool = False
    hotline: str = "100"
    url: str = ""

    @property
    def hazard_en(self) -> str:
        return P.HAZARD.get(self.site_kind, {}).get("en", "flood")

    @property
    def hazard_ne(self) -> str:
        return P.HAZARD.get(self.site_kind, {}).get("ne", "बाढी")

    @property
    def earliest(self) -> ExposedPlace | None:
        named = [e for e in self.exposed if e.name_en or e.name_ne]
        return min(named, key=lambda e: e.travel_time_min) if named else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hazard_en"], d["hazard_ne"] = self.hazard_en, self.hazard_ne
        d["level_en"] = P.LEVEL[self.level]["en"]
        d["level_ne"] = P.LEVEL[self.level]["ne"]
        d["action_en"] = P.ACTION[self.level]["en"]
        d["action_ne"] = P.ACTION[self.level]["ne"]
        return d


def place_name(e: ExposedPlace, lang: str) -> str:
    if lang == "ne":
        return e.name_ne or e.name_en
    return e.name_en or e.name_ne


def build_facts(*, alert_kind: str, level: str, previous_level: str | None, site, reasons: list[dict],
                exposures: list[dict], evidence_time: datetime, needs_confirmation: bool, hotline: str,
                base_url: str, max_places: int = 6) -> AlertFacts:
    """``site``: object with code/name/name_ne/kind/lat/lon/elevation_m. ``exposures``: dicts from DB."""
    named = [e for e in exposures if e.get("name") or e.get("name_ne")]
    # prioritise: the soonest-reached places, but always keep hydropower / border / towns in the list
    key_kinds = {"town", "hydropower", "border_crossing"}
    named.sort(key=lambda e: (e["kind"] not in key_kinds, e["travel_time_min"]))
    chosen = sorted(named[:max_places], key=lambda e: e["travel_time_min"])
    places = [ExposedPlace(e.get("name") or "", e.get("name_ne") or "", e["kind"],
                           round(float(e["travel_time_min"])), round(float(e["path_distance_km"]), 1))
              for e in chosen]
    return AlertFacts(
        alert_kind=alert_kind, level=level, previous_level=previous_level, site_code=site.code,
        site_name_en=site.name, site_name_ne=site.name_ne or site.name, site_kind=site.kind,
        lat=round(site.lat, 4), lon=round(site.lon, 4),
        elevation_m=round(site.elevation_m) if site.elevation_m else None,
        evidence_time_utc=evidence_time.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        evidence_time_npt=evidence_time.astimezone(NPT).strftime("%Y-%m-%d %H:%M"),
        reasons=[{"code": r["code"], "text_en": r["text_en"], "text_ne": r["text_ne"]} for r in reasons[:4]],
        exposed=places, needs_confirmation=needs_confirmation, hotline=hotline,
        url=f"{base_url.rstrip('/')}/#site={site.code}",
    )


_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def numbers_in(text: str) -> set[str]:
    """Normalised numbers appearing in text (Devanagari digits → ASCII, thousands separators removed)."""
    t = P.en_digits(text)
    out = set()
    for m in _NUM.findall(t):
        s = m.replace(",", "")
        try:
            f = float(s)
        except ValueError:
            continue
        out.add(_norm(f))
    return out


def _norm(f: float) -> str:
    return f"{f:.4f}".rstrip("0").rstrip(".")


def allowed_numbers(f: AlertFacts) -> set[str]:
    """Every number an alert may mention: those in the facts and in their rendered phrases."""
    blobs = [f.site_code, f.site_name_en, f.site_name_ne, f.evidence_time_utc, f.evidence_time_npt, f.hotline,
             str(f.lat), str(f.lon), str(f.elevation_m or ""), f.url]
    for r in f.reasons:
        blobs += [r["text_en"], r["text_ne"]]
    for e in f.exposed:
        blobs += [e.name_en, e.name_ne, str(e.travel_time_min), str(e.distance_km)]
    nums = set()
    for b in blobs:
        nums |= numbers_in(b)
    # dates may be written as "26 August" or "2026-08-26"; times as "08:37"
    for part in re.split(r"[-: ]", f.evidence_time_npt + " " + f.evidence_time_utc):
        if part.isdigit():
            nums.add(_norm(float(part)))
    # SMS/format tokens like "GLOF", "S1"; and 1..2 for list numbering
    nums |= {"1", "2"}
    return nums
