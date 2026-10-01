"""Deterministic bilingual alert texts: the guaranteed fallback when the LLM draft is unusable."""

from __future__ import annotations

from himsat.alerts import phrases as P
from himsat.alerts.facts import AlertFacts, place_name

SMS_MAX = {"en": 306, "ne": 268}  # 2 GSM-7 segments / 4 UCS-2 segments


def _places(f: AlertFacts, lang: str, n: int = 4) -> str:
    sep = ", " if lang == "en" else ", "
    items = []
    for e in f.exposed[:n]:
        nm = place_name(e, lang)
        if not nm:
            continue
        t = P.fmt_num(e.travel_time_min, lang, 0)
        items.append(f"{nm} (~{t} min)" if lang == "en" else f"{nm} (~{t} मिनेट)")
    return sep.join(items)


def _arrival(f: AlertFacts, lang: str) -> str:
    e = f.earliest
    if not e:
        return ""
    return P.ARRIVAL[lang].format(name=place_name(e, lang), min=P.fmt_num(e.travel_time_min, lang, 0))


def render(f: AlertFacts) -> dict[str, str]:
    out: dict[str, str] = {}
    for lang in ("en", "ne"):
        level = P.LEVEL[f.level][lang]
        site = f.site_name_en if lang == "en" else f.site_name_ne
        kind = P.SITE_KIND.get(f.site_kind, {}).get(lang, f.site_kind)
        hazard = f.hazard_en if lang == "en" else f.hazard_ne
        if f.alert_kind == "all_clear":
            title = (f"{level}: risk reduced at {site}" if lang == "en" else f"{level}: {site} को जोखिम घटेको छ")
            body = "\n".join([
                P.ALL_CLEAR[lang].format(site=site, level=level),
                P.FOOTER[lang].format(hotline=P.ne_digits(f.hotline) if lang == "ne" else f.hotline),
                f.url,
            ])
            sms = f"HimSat: {P.ALL_CLEAR[lang].format(site=site, level=level)}"
            out.update({f"title_{lang}": title, f"body_{lang}": body, f"sms_{lang}": sms[: SMS_MAX[lang]]})
            continue

        if lang == "en":
            prefix = "UPDATE – " if f.alert_kind == "update" else ""
            title = f"{prefix}{level}: {hazard} threat from {site}"
            head = f"{level} — {site} ({kind}"
            head += f", {P.fmt_num(f.elevation_m, 'en', 0)} m" if f.elevation_m else ""
            head += f", {f.lat}N {f.lon}E)"
            changed = f"What the satellites show (as of {f.evidence_time_npt} NPT):"
            todo = "What to do: "
            detail = "Details and map: "
        else:
            prefix = "अद्यावधिक – " if f.alert_kind == "update" else ""
            title = f"{prefix}{level}: {site} बाट {hazard}को खतरा"
            head = f"{level} — {site} ({kind}"
            head += f", {P.fmt_num(f.elevation_m, 'ne', 0)} मिटर" if f.elevation_m else ""
            head += ")"
            changed = f"स्याटेलाइटले देखाएको परिवर्तन ({P.ne_digits(f.evidence_time_npt)} नेपाली समय):"
            todo = "के गर्ने: "
            detail = "विस्तृत विवरण र नक्सा: "
        lines = [head, changed]
        lines += [f"- {r['text_' + lang]}" for r in f.reasons]
        arr = _arrival(f, lang)
        if arr:
            lines.append(arr)
        places = _places(f, lang)
        if places:
            lines.append(P.AT_RISK[lang].format(list=places))
        lines.append(todo + P.ACTION[f.level][lang])
        if f.needs_confirmation:
            lines.append(P.UNCONFIRMED[lang])
        lines.append(P.FOOTER[lang].format(hotline=P.ne_digits(f.hotline) if lang == "ne" else f.hotline))
        lines.append(detail + f.url)
        body = "\n".join(lines)

        out[f"title_{lang}"] = title
        out[f"body_{lang}"] = body
        out[f"sms_{lang}"] = sms_text(f, lang)
    return out


def sms_text(f: AlertFacts, lang: str) -> str:
    level = P.LEVEL[f.level][lang]
    site = f.site_name_en if lang == "en" else f.site_name_ne
    reason = f.reasons[0]["text_" + lang] if f.reasons else ""
    e = f.earliest
    if lang == "en":
        parts = [f"HimSat {level}: {f.hazard_en} threat from {site}."]
        if reason:
            parts.append(reason + ".")
        if e:
            parts.append(f"Could reach {place_name(e, 'en')} in ~{P.fmt_num(e.travel_time_min, 'en', 0)} min.")
        parts.append("Move away from the river to high ground." if f.level == "high"
                     else "Stay alert, avoid the riverbank.")
        parts.append(f"Emergency {f.hotline}")
    else:
        parts = [f"HimSat {level}: {site} बाट {f.hazard_ne}को खतरा।"]
        if reason:
            parts.append(reason + "।")
        if e:
            parts.append(f"~{P.fmt_num(e.travel_time_min, 'ne', 0)} मिनेटमा {place_name(e, 'ne')} पुग्न सक्छ।")
        parts.append("नदी किनारबाट अग्लो ठाउँमा जानुहोस्।" if f.level == "high"
                     else "सतर्क रहनुहोस्, नदी किनार नजानुहोस्।")
        parts.append(f"आपतकालीन {P.ne_digits(f.hotline)}")
    text = " ".join(parts)
    limit = SMS_MAX[lang]
    if len(text) > limit and reason:  # drop the reason first, then hard-truncate
        parts = [p for p in parts if not p.startswith(reason)]
        text = " ".join(parts)
    return text[:limit]
