export type Lang = "ne" | "en";

const STRINGS = {
  title: { en: "HimSat Engine", ne: "हिमस्याट इन्जिन" },
  subtitle: { en: "Glacier & glacial-lake early warning from space", ne: "अन्तरिक्षबाट हिमनदी र हिमताल पूर्वचेतावनी" },
  live: { en: "Live", ne: "प्रत्यक्ष" },
  hindcast: { en: "Hindcast", ne: "पुनरावलोकन" },
  admin: { en: "Admin", ne: "प्रशासन" },
  alerts: { en: "Alerts", ne: "सूचनाहरू" },
  activeAlerts: { en: "Active alerts", ne: "सक्रिय सूचनाहरू" },
  noAlerts: { en: "No active alerts.", ne: "अहिले कुनै सक्रिय सूचना छैन।" },
  sites: { en: "Monitored sites", ne: "अनुगमित स्थलहरू" },
  site: { en: "Site", ne: "स्थल" },
  about: { en: "About", ne: "बारेमा" },
  risk: { en: "Risk", ne: "जोखिम" },
  high: { en: "High", ne: "उच्च" },
  medium: { en: "Medium", ne: "मध्यम" },
  low: { en: "Low", ne: "न्यून" },
  unknown: { en: "Not assessed", ne: "मूल्याङ्कन बाँकी" },
  why: { en: "What the satellites show", ne: "स्याटेलाइटले के देखायो" },
  downstream: { en: "Downstream at risk", ne: "तल्लो तटीय क्षेत्रमा जोखिम" },
  arrival: { en: "earliest arrival", ne: "सबैभन्दा छिटो पुग्ने समय" },
  minutes: { en: "min", ne: "मिनेट" },
  km: { en: "km", ne: "कि.मि." },
  elevation: { en: "Elevation", ne: "उचाइ" },
  area: { en: "Area", ne: "क्षेत्रफल" },
  assessed: { en: "Assessed", ne: "मूल्याङ्कन" },
  velocity: { en: "Downslope velocity (m/day)", ne: "भिरतर्फको गति (मिटर/दिन)" },
  lakeArea: { en: "Lake area (km²)", ne: "तालको क्षेत्रफल (वर्ग कि.मि.)" },
  insarVelocity: { en: "InSAR line-of-sight motion (mm/month)", ne: "इनसार दृष्टिरेखा गति (मिलिमिटर/महिना)" },
  riskHistory: { en: "Risk score", ne: "जोखिम अंक" },
  close: { en: "Close", ne: "बन्द" },
  basemap: { en: "Basemap", ne: "आधार नक्सा" },
  map: { en: "Map", ne: "नक्सा" },
  satellite: { en: "Satellite", ne: "स्याटेलाइट" },
  layers: { en: "Layers", ne: "तहहरू" },
  changes: { en: "Recent surface changes", ne: "हालैका सतह परिवर्तन" },
  potentiallyDangerous: { en: "Potentially dangerous (no recent change observed)", ne: "सम्भावित खतरनाक (हालै परिवर्तन देखिएको छैन)" },
  issued: { en: "Issued", ne: "जारी" },
  leadTime: { en: "lead time", ne: "अग्रिम समय" },
  beforeEvent: { en: "before the event", ne: "घटनाभन्दा पहिले" },
  afterEvent: { en: "after the event", ne: "घटनापछि" },
  event: { en: "Event", ne: "घटना" },
  play: { en: "Play", ne: "चलाउनुहोस्" },
  pause: { en: "Pause", ne: "रोक्नुहोस्" },
  selectHindcast: { en: "Select a hindcast", ne: "पुनरावलोकन छान्नुहोस्" },
  noHindcasts: { en: "No hindcast reports yet. Run `himsat hindcast …`.", ne: "पुनरावलोकन प्रतिवेदन छैन।" },
  emergency: { en: "Emergency: 100 (Nepal Police)", ne: "आपतकालीन सम्पर्क: १०० (नेपाल प्रहरी)" },
  disclaimer: {
    en: "Automated satellite-based early warning. Always follow instructions from local authorities.",
    ne: "यो स्याटेलाइटमा आधारित स्वचालित पूर्वचेतावनी हो। सधैं स्थानीय प्रशासनको निर्देशन पालना गर्नुहोस्।",
  },
  aboutText: {
    en: "HimSat watches Himalayan glaciers, glacial lakes and steep ice/rock slopes with free Copernicus Sentinel-1 radar (sees through clouds) and Sentinel-2 optical images. It maps lakes, measures how fast ice and rock are moving, detects fresh landslide and avalanche scars, traces the downstream flood path, and issues bilingual alerts when observed change makes a site dangerous.",
    ne: "हिमस्याटले निःशुल्क कोपर्निकस सेन्टिनेल-१ राडार (बादल छेडेर हेर्न सक्ने) र सेन्टिनेल-२ तस्बिर प्रयोग गरी हिमालका हिमनदी, हिमताल र ठाडा हिउँ-चट्टानका भिरहरूको अनुगमन गर्छ। यसले ताल नक्साङ्कन, हिउँ र चट्टानको गति मापन, नयाँ पहिरो पहिचान, बाढीको बाटो र तल्लो तटीय जोखिम विश्लेषण गरी खतरा देखिँदा नेपाली र अङ्ग्रेजीमा सूचना जारी गर्छ।",
  },
  apiKey: { en: "Admin API key", ne: "प्रशासन API कुञ्जी" },
  reviewQueue: { en: "Alerts waiting for review", ne: "समीक्षाको पर्खाइमा रहेका सूचनाहरू" },
  approve: { en: "Approve & send", ne: "स्वीकृत गरी पठाउनुहोस्" },
  cancel: { en: "Cancel", ne: "रद्द गर्नुहोस्" },
  officer: { en: "Duty officer name", ne: "कर्तव्यरत अधिकारीको नाम" },
  subscribers: { en: "Subscribers", ne: "सदस्यहरू" },
  addSubscriber: { en: "Add subscriber", ne: "सदस्य थप्नुहोस्" },
  save: { en: "Save", ne: "सुरक्षित गर्नुहोस्" },
  runs: { en: "Pipeline runs", ne: "प्रणाली सञ्चालन" },
  runNow: { en: "Run now", ne: "अहिले चलाउनुहोस्" },
  error: { en: "Something went wrong", ne: "केही समस्या भयो" },
  legend: { en: "Legend", ne: "सङ्केत" },
  searchSites: { en: "Search sites by name or code", ne: "नाम वा कोडले स्थल खोज्नुहोस्" },
  flowPath: { en: "Flood path (selected site)", ne: "बाढीको बाटो (छानिएको स्थल)" },
  exposedPlace: { en: "Place at risk, with arrival time", ne: "जोखिममा रहेको स्थान र पुग्ने समय" },
  eventSource: { en: "Event source", ne: "घटनाको स्रोत" },
  radarChange: { en: "Radar-detected surface change", ne: "राडारले देखाएको सतह परिवर्तन" },
  noMatch: { en: "No matching sites.", ne: "मिल्ने स्थल भेटिएन।" },
  kind_glacial_lake: { en: "Glacial lake", ne: "हिमताल" },
  kind_glacier: { en: "Glacier", ne: "हिमनदी" },
  kind_slope: { en: "Unstable ice/rock slope", ne: "अस्थिर हिउँ-चट्टान भिर" },
  kind_barrier_lake: { en: "Landslide-dammed lake", ne: "पहिरोले थुनिएको ताल" },
  kind_landslide: { en: "Landslide / mass movement", ne: "पहिरो" },
} as const;

export type Key = keyof typeof STRINGS;

let lang: Lang = (() => {
  try {
    const v = localStorage.getItem("himsat.lang");
    if (v === "en" || v === "ne") return v;
  } catch {
    /* storage unavailable */
  }
  return navigator.language?.startsWith("ne") ? "ne" : "en";
})();

export function getLang(): Lang {
  return lang;
}

export function setLang(l: Lang): void {
  lang = l;
  document.documentElement.lang = l;
  try {
    localStorage.setItem("himsat.lang", l);
  } catch {
    /* ignore */
  }
}

export function t(k: Key): string {
  return STRINGS[k][lang];
}

const NE_DIGITS = "०१२३४५६७८९";
export function num(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  const s = v.toFixed(digits);
  return lang === "ne" ? s.replace(/[0-9]/g, (d) => NE_DIGITS[Number(d)]) : s;
}

export function fmtTime(iso: string | null | undefined): string {
  if (!iso) return "–";
  const d = new Date(iso);
  const s = d.toLocaleString(lang === "ne" ? "ne-NP" : "en-GB", {
    timeZone: "Asia/Kathmandu", year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
  // browsers without Nepali locale data fall back to Latin digits
  return lang === "ne" ? `${s.replace(/[0-9]/g, (x) => NE_DIGITS[Number(x)])} नेपाली समय` : `${s} NPT`;
}

export function kindLabel(kind: string): string {
  const k = `kind_${kind}` as Key;
  return k in STRINGS ? t(k) : kind;
}

export function levelLabel(level: string): string {
  return (["high", "medium", "low"] as const).includes(level as "high") ? t(level as "high") : t("unknown");
}
