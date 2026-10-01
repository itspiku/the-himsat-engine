"""Alert composition: LLM draft → strict validation → deterministic fallback.

The LLM makes alerts read naturally, especially in Nepali, and it can blend several signals
into one clear story. It is never trusted with facts. Every number in its output must come from
the fact sheet. The site, the earliest-hit place, the risk level and the emergency number must
all appear, and the Nepali text must actually be in Nepali. If a draft fails and a repair retry
also fails, the deterministic template goes out instead. Every alert records which generator
produced it and the validator's findings.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from himsat.alerts import phrases as P
from himsat.alerts.facts import AlertFacts, allowed_numbers, numbers_in, place_name
from himsat.alerts.llm import LLMClient, LLMError
from himsat.alerts.templates import SMS_MAX, render

log = logging.getLogger(__name__)

FIELDS = ("title_en", "body_en", "sms_en", "title_ne", "body_ne", "sms_ne")
SCHEMA = {"type": "object", "properties": {k: {"type": "string"} for k in FIELDS}, "required": list(FIELDS),
          "additionalProperties": False}

SYSTEM_PROMPT = """You write official early-warning messages for Nepal's glacier and glacial-lake hazard monitoring system (HimSat).
Readers: villagers, trekkers, hydropower workers, municipal and district disaster officials. Many read only Nepali.

STRICT RULES
1. Use ONLY the information in FACTS. Never invent or change numbers, places, dates, causes, casualties or instructions.
2. Every number you write must appear in FACTS (you may write it with Devanagari digits in Nepali).
3. Mention: the hazard and the site; what the satellites observed (the reasons); which places are at risk and how soon it could reach the first of them; what people must do (use the given action text faithfully); the emergency number.
4. Nepali (title_ne, body_ne, sms_ne): plain, clear, respectful Nepali in Devanagari script with Devanagari digits. Short sentences. No English words except place names or "GLOF".
5. English (title_en, body_en, sms_en): plain, calm, direct.
6. Do not add reassurance or speculation. If FACTS says needs_confirmation is true, say the signal is not yet confirmed on the ground.
7. sms_en at most 300 characters; sms_ne at most 260 characters. Titles at most 90 characters.
8. Include the URL from FACTS at the end of body_en and body_ne.

Return ONLY a JSON object with exactly these string keys: title_en, body_en, sms_en, title_ne, body_ne, sms_ne."""


@dataclass
class Composition:
    texts: dict[str, str]
    generator: str
    validation: dict = field(default_factory=dict)


def _devanagari_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "ऀ" <= c <= "ॿ") / len(letters)


def validate(texts: dict, f: AlertFacts) -> list[str]:
    """Return a list of problems (empty = acceptable)."""
    issues: list[str] = []
    for k in FIELDS:
        if not isinstance(texts.get(k), str) or not texts[k].strip():
            issues.append(f"missing or empty field {k}")
    if issues:
        return issues
    allowed = allowed_numbers(f)
    for k in FIELDS:
        extra = numbers_in(texts[k]) - allowed
        if extra:
            issues.append(f"{k} contains numbers not in FACTS: {sorted(extra)[:6]}")
    for k in ("title_ne", "body_ne", "sms_ne"):
        if _devanagari_ratio(texts[k]) < 0.6:
            issues.append(f"{k} is not mainly Nepali (Devanagari)")
    for k in ("title_en", "body_en", "sms_en"):
        if _devanagari_ratio(texts[k]) > 0.1:
            issues.append(f"{k} should be English")
    lvl_en = P.LEVEL[f.level]["en"].lower().split()[0]  # "high" / "medium" / "low"
    if lvl_en not in (texts["title_en"] + texts["sms_en"]).lower():
        issues.append("risk level missing from English title/SMS")
    if P.LEVEL[f.level]["ne"].split()[0] not in texts["title_ne"] + texts["sms_ne"]:
        issues.append("risk level missing from Nepali title/SMS")
    site_tokens_en = {f.site_name_en.lower(), f.site_code.lower()}
    if not any(t and t in (texts["title_en"] + texts["body_en"]).lower() for t in site_tokens_en):
        issues.append("site name missing from English text")
    if not any(t and t in texts["title_ne"] + texts["body_ne"] for t in (f.site_name_ne, f.site_name_en, f.site_code)):
        issues.append("site name missing from Nepali text")
    e = f.earliest
    if e and f.alert_kind != "all_clear":
        if not any(n and n.lower() in texts["body_en"].lower() for n in (e.name_en, e.name_ne)):
            issues.append(f"first place at risk ({place_name(e, 'en')}) missing from body_en")
        if not any(n and n in texts["body_ne"] for n in (e.name_ne, e.name_en)):
            issues.append(f"first place at risk ({place_name(e, 'ne')}) missing from body_ne")
        mins = P.fmt_num(e.travel_time_min, "en", 0)
        for lang in ("en", "ne"):
            if mins not in numbers_in(texts[f"body_{lang}"]):
                issues.append(f"arrival time ({mins} min) missing from body_{lang}")
    for lang in ("en", "ne"):
        hot = f.hotline if lang == "en" else P.ne_digits(f.hotline)
        body = texts[f"body_{lang}"]
        if f.hotline not in P.en_digits(body) and hot not in body:
            issues.append(f"emergency number missing from body_{lang}")
    urls = set(re.findall(r"https?://\S+", " ".join(texts[k] for k in FIELDS)))
    bad_urls = [u for u in urls if not u.rstrip(".,)").startswith(f.url.split("#")[0])]
    if bad_urls:
        issues.append(f"unknown URLs: {bad_urls[:2]}")
    if f.level == "high" and re.search(r"\b(no (risk|danger)|safe to|nothing to worry)\b", texts["body_en"].lower()):
        issues.append("reassuring language contradicts a HIGH alert")
    return issues


def _fix_lengths(texts: dict, fallback: dict) -> list[str]:
    fixed = []
    for lang in ("en", "ne"):
        k = f"sms_{lang}"
        if len(texts[k]) > SMS_MAX[lang]:
            texts[k] = fallback[k]
            fixed.append(f"{k} too long; replaced by template")
    for k in ("title_en", "title_ne"):
        if len(texts[k]) > 120:
            texts[k] = fallback[k]
            fixed.append(f"{k} too long; replaced by template")
    return fixed


def required_items(f: AlertFacts) -> str:
    """Explicit checklist: small models follow a concrete list far better than general rules."""
    items = [f"the risk level word '{P.LEVEL[f.level]['en'].split()[0]}' (English) and "
             f"'{P.LEVEL[f.level]['ne'].split()[0]}' (Nepali)",
             f"the site name '{f.site_name_en}' / '{f.site_name_ne}'",
             f"the emergency number {f.hotline} in body_en and body_ne",
             f"the link {f.url} at the end of body_en and body_ne"]
    e = f.earliest
    if e and f.alert_kind != "all_clear":
        m = P.fmt_num(e.travel_time_min, "en", 0)
        items.append(f"in body_en: '{place_name(e, 'en')}' and '{m} minutes'; in body_ne: "
                     f"'{place_name(e, 'ne')}' and '{P.ne_digits(m)} मिनेट'")
    return "REQUIRED in your answer:\n- " + "\n- ".join(items)


def compose(f: AlertFacts, llm: LLMClient | None, retries: int = 1,
            languages: tuple[str, ...] = ("en",)) -> Composition:
    """Draft with the LLM, validate, fall back to templates.

    ``languages`` lists the languages whose text the LLM may write. Others always come from the
    reviewed templates. Nepali defaults to templates: in testing, 7-8B open models produced
    grammatically wrong Nepali that no automatic check can catch.
    """
    template = render(f)
    if llm is None or not languages:
        return Composition(template, "template", {"llm": "disabled"})
    facts_json = json.dumps(f.to_dict(), ensure_ascii=False, indent=1)
    base_user = f"FACTS:\n{facts_json}\n\n{required_items(f)}"
    user = base_user + "\n\nWrite the alert now."
    attempts: list[dict] = []
    for i in range(retries + 1):
        try:
            draft = llm.chat_json(SYSTEM_PROMPT, user, schema=SCHEMA)
        except LLMError as e:
            log.warning("LLM unavailable (%s); using template", e)
            attempts.append({"attempt": i + 1, "error": str(e)})
            break
        texts = {k: str(draft.get(k, "")).strip() for k in FIELDS}
        for k in FIELDS:  # languages not delegated to the LLM keep the reviewed template text
            if k.rsplit("_", 1)[1] not in languages:
                texts[k] = template[k]
        fixed = _fix_lengths(texts, template) if all(texts.values()) else []
        issues = validate(texts, f)
        attempts.append({"attempt": i + 1, "issues": issues, "fixed": fixed})
        if not issues:
            gen = f"{llm.name}[{','.join(languages)}]+template" if set(languages) != {"en", "ne"} else llm.name
            return Composition(texts, gen, {"attempts": attempts, "status": "accepted", "llm_languages": list(languages)})
        log.info("LLM draft rejected (attempt %d): %s", i + 1, "; ".join(issues))
        user = (base_user + "\n\nYour previous answer was rejected for these reasons:\n- " + "\n- ".join(issues)
                + "\nRewrite the alert and fix every problem. Return only the JSON object.")
    return Composition(template, "template", {"attempts": attempts, "status": "fallback"})
