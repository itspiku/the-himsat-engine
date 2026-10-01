"""Bilingual (Nepali / English) phrase bank.

This is the single source of the wording used in risk reasons, deterministic alert templates
and the LLM validator. Nepali uses Devanagari digits. Wording is kept short and actionable for
SMS and loudspeaker use. Changes should be reviewed by a native speaker from the disaster risk
reduction (DRR) community.
"""

from __future__ import annotations

_NE_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")
_EN_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def ne_digits(s: str | float | int) -> str:
    return str(s).translate(_NE_DIGITS)


def en_digits(s: str) -> str:
    return s.translate(_EN_DIGITS)


def fmt_num(v: float, lang: str = "en", digits: int | None = None) -> str:
    """Human rounding: 2 significant decimals for small values, integers for large."""
    if digits is None:
        av = abs(v)
        digits = 0 if av >= 20 else (1 if av >= 2 else 2)
    s = f"{v:.{digits}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return ne_digits(s) if lang == "ne" else s


LEVEL = {
    "high": {"en": "HIGH RISK", "ne": "उच्च जोखिम"},
    "medium": {"en": "MEDIUM RISK", "ne": "मध्यम जोखिम"},
    "low": {"en": "LOW RISK", "ne": "न्यून जोखिम"},
    "unknown": {"en": "UNKNOWN", "ne": "अज्ञात"},
}

SITE_KIND = {
    "glacial_lake": {"en": "glacial lake", "ne": "हिमताल"},
    "glacier": {"en": "glacier", "ne": "हिमनदी"},
    "slope": {"en": "ice/rock slope", "ne": "हिउँ-चट्टानको भिर"},
    "barrier_lake": {"en": "landslide-dammed lake", "ne": "पहिरोले थुनिएको ताल"},
    "landslide": {"en": "landslide", "ne": "पहिरो"},
}

HAZARD = {
    "glacial_lake": {"en": "glacial lake outburst flood (GLOF)", "ne": "हिमताल विस्फोट बाढी (GLOF)"},
    "glacier": {"en": "ice avalanche and debris flood", "ne": "हिमपहिरो र लेदोसहितको बाढी"},
    "slope": {"en": "rock-ice avalanche and debris flood", "ne": "चट्टान-हिउँ पहिरो र लेदोसहितको बाढी"},
    "barrier_lake": {"en": "flood from a breaching landslide dam", "ne": "पहिरोले थुनेको ताल फुटेर आउने बाढी"},
    "landslide": {"en": "landslide and debris flow", "ne": "पहिरो र लेदो बाढी"},
}

ASSET_KIND = {
    "town": {"en": "town", "ne": "बजार"},
    "settlement": {"en": "settlement", "ne": "बस्ती"},
    "hydropower": {"en": "hydropower plant", "ne": "जलविद्युत आयोजना"},
    "dam": {"en": "dam/weir", "ne": "बाँध"},
    "border_crossing": {"en": "border crossing", "ne": "नाका"},
    "bridge": {"en": "bridge", "ne": "पुल"},
    "school": {"en": "school", "ne": "विद्यालय"},
    "health": {"en": "health facility", "ne": "स्वास्थ्य संस्था"},
    "tourism": {"en": "lodge/hotel", "ne": "होटल/लज"},
    "police": {"en": "police post", "ne": "प्रहरी चौकी"},
    "government": {"en": "government office", "ne": "सरकारी कार्यालय"},
}

# {v} is replaced with the formatted indicator value
REASON = {
    "lake_area_km2": {"en": "Lake area is {v} km²", "ne": "तालको क्षेत्रफल {v} वर्ग कि.मि. छ"},
    "lake_growth_recent_pct": {"en": "Lake grew {v}% in about a month",
                               "ne": "ताल करिब एक महिनामा {v}% ले बढेको छ"},
    "lake_growth_annual_pct": {"en": "Lake is {v}% larger than a year ago",
                               "ne": "ताल एक वर्षअघिभन्दा {v}% ठूलो भएको छ"},
    "glacier_contact_m": {"en": "Lake is {v} m from glacier ice", "ne": "ताल हिमनदीको हिउँबाट {v} मिटर मात्र टाढा छ"},
    "steep_walls_deg": {"en": "Steep slopes (up to {v}°) above the lake could send avalanches into it",
                        "ne": "तालमाथि {v}° सम्मका ठाडा भिर छन्, जहाँबाट हिउँ वा चट्टान तालमा खस्न सक्छ"},
    "turbidity_rise": {"en": "Lake water has turned more turbid (+{v}%)", "ne": "तालको पानी धमिलो भएको छ (+{v}%)"},
    "barrier_lake": {"en": "A landslide has blocked the river and formed a lake",
                     "ne": "पहिरोले नदी थुनेर ताल बनेको छ"},
    "velocity_ratio": {"en": "Ice/rock is moving {v}× faster than normal",
                       "ne": "हिउँ/चट्टान सामान्यभन्दा {v} गुणा छिटो सरिरहेको छ"},
    "velocity_steep_m_day": {"en": "The surface is moving {v} m per day on steep ground",
                             "ne": "ठाडो भिरमा सतह दैनिक {v} मिटर सरिरहेको छ"},
    "velocity_trend": {"en": "Speed increased in {v} consecutive measurements",
                       "ne": "लगातार {v} पटकको मापनमा गति बढेको छ"},
    "new_fractures_m": {"en": "New cracks totalling about {v} m have appeared",
                        "ne": "करिब {v} मिटर लामा नयाँ चिराहरू देखा परेका छन्"},
    "sar_change_km2": {"en": "Radar shows fresh surface disturbance over {v} km²",
                       "ne": "राडार तस्बिरमा {v} वर्ग कि.मि. क्षेत्रमा सतहमा नयाँ हलचल देखिएको छ"},
    "source_slope_deg": {"en": "The source area is very steep ({v}°)", "ne": "स्रोत क्षेत्र निकै ठाडो छ ({v}°)"},
    "landslide_area_km2": {"en": "Active landslide area of {v} km²", "ne": "सक्रिय पहिरो क्षेत्र {v} वर्ग कि.मि."},
}

ACTION = {
    "high": {
        "en": ("Move away from the river banks to high ground now. Do not cross or stay near the river. "
               "Hydropower and road crews: stop riverside work and move staff to safety. "
               "Local authorities: activate sirens and evacuation."),
        "ne": ("नदी किनारबाट तुरुन्तै अग्लो सुरक्षित स्थानमा जानुहोस्। नदी नतर्नुहोस् र किनारमा नबस्नुहोस्। "
               "जलविद्युत आयोजना र सडकमा खटिएका कामदारहरू नदी किनारको काम रोकेर सुरक्षित स्थानमा जानुहोस्। "
               "स्थानीय तहले साइरन बजाई उद्धार तथा स्थानान्तरण सुरु गर्नुहोस्।"),
    },
    "medium": {
        "en": ("Stay alert. Avoid spending time near the river, know your nearest high ground, and follow "
               "official instructions. Authorities: verify on the ground and prepare evacuation plans."),
        "ne": ("सतर्क रहनुहोस्। नदी किनारमा अनावश्यक समय नबिताउनुहोस्, नजिकको अग्लो सुरक्षित स्थान पहिल्यै थाहा "
               "पाउनुहोस् र आधिकारिक निर्देशन पालना गर्नुहोस्। सम्बन्धित निकायले स्थलगत पुष्टि गरी उद्धार योजना तयार राख्नुहोस्।"),
    },
    "low": {"en": "No action needed. Monitoring continues.", "ne": "तत्काल कुनै कदम आवश्यक छैन। अनुगमन जारी छ।"},
}

ALL_CLEAR = {
    "en": "Risk at {site} has decreased to {level}. Continue to follow local authority guidance.",
    "ne": "{site} को जोखिम घटेर {level} मा झरेको छ। स्थानीय प्रशासनको निर्देशन पालना गरिरहनुहोस्।",
}

ARRIVAL = {
    "en": "A flood could reach {name} in as little as {min} minutes.",
    "ne": "बाढी {min} मिनेटभित्रै {name} पुग्न सक्छ।",
}

AT_RISK = {"en": "At risk downstream: {list}.", "ne": "तल्लो तटीय क्षेत्रमा जोखिममा: {list}।"}

UNCONFIRMED = {
    "en": "Satellite signal not yet confirmed on the ground.",
    "ne": "यो स्याटेलाइट संकेत स्थलगत रूपमा पुष्टि भइसकेको छैन।",
}

FOOTER = {
    "en": "Automated satellite early warning (HimSat Engine). Follow local authorities. Emergency: {hotline}.",
    "ne": "यो स्याटेलाइट तस्बिरमा आधारित स्वचालित पूर्वचेतावनी हो (HimSat)। स्थानीय प्रशासनको निर्देशन पालना गर्नुहोस्। आपतकालीन सम्पर्क: {hotline}।",
}


def reason_text(code: str, value: float, lang: str) -> str:
    tpl = REASON.get(code, {}).get(lang)
    if not tpl:
        return code
    if code == "velocity_trend":
        v = fmt_num(int(round(value)), lang, 0)
    elif code in ("lake_growth_recent_pct", "lake_growth_annual_pct"):
        v = fmt_num(value, lang, 0)
    elif code == "turbidity_rise":
        v = fmt_num(value * 100.0, lang, 0)
    elif code in ("glacier_contact_m", "new_fractures_m", "steep_walls_deg", "source_slope_deg"):
        v = fmt_num(round(value / 10.0) * 10.0 if code != "steep_walls_deg" and code != "source_slope_deg"
                    else value, lang, 0)
    else:
        v = fmt_num(value, lang)
    return tpl.format(v=v)
