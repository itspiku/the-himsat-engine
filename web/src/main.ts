import type { Map as MLMap } from "maplibre-gl";
import { api, type Alert, type FC, type HindcastReport, type SiteDetail, type SiteProps } from "./api";
import { lineChart, type Point } from "./chart";
import { fmtTime, getLang, kindLabel, levelLabel, num, setLang, t, type Lang } from "./i18n";
import { createMap, fitTo, LEVEL_COLOR, onReady, setBasemap, setData, setVisible } from "./map";
import "./style.css";

type Mode = "live" | "hindcast" | "admin";

interface State {
  mode: Mode;
  sites: FC<SiteProps> | null;
  alerts: Alert[];
  selected: string | null;
  hindcast: HindcastReport | null;
  hcList: { name: string; aoi_name: string; event_time: string | null }[];
  hcName: string | null;
  hcIndex: number; // index into hindcast timeline dates
  hcDates: number[];
  playing: number | null;
}

const S: State = { mode: "live", sites: null, alerts: [], selected: null, hindcast: null, hcList: [], hcName: null,
  hcIndex: 0, hcDates: [], playing: null };
let map: MLMap;

const $ = <T extends HTMLElement = HTMLElement>(sel: string, root: ParentNode = document) => root.querySelector(sel) as T;
const esc = (s: string | null | undefined) =>
  (s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
const nm = (en: string, ne: string) => (getLang() === "ne" ? ne || en : en || ne);

function levelBadge(level: string): string {
  const icon = level === "high" ? "▲" : level === "medium" ? "◆" : level === "low" ? "●" : "○";
  return `<span class="badge lv-${esc(level)}" style="--lv:${LEVEL_COLOR[level] ?? LEVEL_COLOR.unknown}">${icon} ${esc(levelLabel(level))}</span>`;
}

// ------------------------------------------------------------------------------------------
// shell
// ------------------------------------------------------------------------------------------
function shell(): void {
  $("#app").innerHTML = `
  <header class="top">
    <div class="brand">
      <svg viewBox="0 0 64 64" aria-hidden="true"><path d="M6 50 L24 20 L32 33 L41 16 L58 50 Z" fill="currentColor"/><circle cx="50" cy="12" r="5" fill="#f5a524"/></svg>
      <div><h1 data-i18n="title"></h1><p data-i18n="subtitle"></p></div>
    </div>
    <nav class="modes" role="tablist">
      <button data-mode="live" data-i18n="live"></button>
      <button data-mode="hindcast" data-i18n="hindcast"></button>
      <button data-mode="admin" data-i18n="admin"></button>
    </nav>
    <div class="tools">
      <button class="ghost" id="basemap"></button>
      <button class="ghost" id="lang" lang="ne"></button>
    </div>
  </header>
  <main class="body">
    <div id="map" class="map"></div>
    <aside class="panel" id="panel"></aside>
  </main>
  <footer class="foot"><span data-i18n="disclaimer"></span> · <strong data-i18n="emergency"></strong></footer>`;
  document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((b) =>
    b.addEventListener("click", () => setMode(b.dataset.mode as Mode)));
  $("#lang").addEventListener("click", () => {
    setLang(getLang() === "ne" ? "en" : "ne");
    translate();
    render();
  });
  let base: "osm" | "s2" = "osm";
  $("#basemap").addEventListener("click", () => {
    base = base === "osm" ? "s2" : "osm";
    setBasemap(map, base);
    $("#basemap").textContent = base === "osm" ? t("satellite") : t("map");
  });
  translate();
}

function translate(): void {
  document.querySelectorAll<HTMLElement>("[data-i18n]").forEach((e) => (e.textContent = t(e.dataset.i18n as "title")));
  $("#lang").textContent = getLang() === "ne" ? "English" : "नेपाली";
  $("#basemap").textContent = t("satellite");
  document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((b) =>
    b.setAttribute("aria-selected", String(b.dataset.mode === S.mode)));
}

function setMode(m: Mode): void {
  if (S.playing) stopPlay();
  S.mode = m;
  S.selected = null;
  location.hash = m === "live" ? "" : m;
  translate();
  clearSelection();
  void load();
}

// ------------------------------------------------------------------------------------------
// data loading
// ------------------------------------------------------------------------------------------
async function load(): Promise<void> {
  const panel = $("#panel");
  panel.innerHTML = `<div class="loading">…</div>`;
  try {
    if (S.mode === "live") {
      const [sites, alerts] = await Promise.all([api.sites(), api.alerts(true)]);
      S.sites = sites;
      S.alerts = alerts;
      showSites(sites);
      setData(map, "event-point", { type: "FeatureCollection", features: [] });
      api.events().then((ev) => setData(map, "events", ev as unknown as GeoJSON.FeatureCollection)).catch(() => undefined);
      if (sites.features.length) fitTo(map, sites.features.map((f) => f.geometry), 10);
    } else if (S.mode === "hindcast") {
      await loadHindcastList();
      return;
    } else {
      renderAdmin();
      return;
    }
  } catch (e) {
    panel.innerHTML = `<p class="error">${t("error")}: ${esc(String(e))}</p>`;
    return;
  }
  render();
}

function showSites(fc: FC<SiteProps>): void {
  // markers only for sites that are not tiny lakes, to keep the map readable
  const feats = fc.features.map((f) => {
    const marker = f.properties.kind === "slope" || f.properties.kind === "landslide" || f.properties.kind === "barrier_lake";
    return { ...f, properties: { ...f.properties, _marker: marker } };
  });
  const centroids = feats.map((f) => ({ type: "Feature" as const, geometry: { type: "Point" as const, coordinates: [f.properties.lon, f.properties.lat] }, properties: f.properties }));
  setData(map, "sites", { type: "FeatureCollection", features: [...feats, ...centroids] as GeoJSON.Feature[] });
}

function render(): void {
  if (S.mode === "live") return S.selected ? void renderSite(S.selected) : renderLive();
  if (S.mode === "hindcast") return renderHindcast();
  return renderAdmin();
}

// ------------------------------------------------------------------------------------------
// live view
// ------------------------------------------------------------------------------------------
function renderLive(): void {
  const panel = $("#panel");
  const feats = S.sites?.features ?? [];
  const rank: Record<string, number> = { high: 0, medium: 1, low: 2, unknown: 3 };
  const top = [...feats].sort((a, b) => (rank[a.properties.level] ?? 4) - (rank[b.properties.level] ?? 4)
    || (b.properties.score ?? 0) - (a.properties.score ?? 0)).slice(0, 40);
  const counts = { high: 0, medium: 0, low: 0, unknown: 0 } as Record<string, number>;
  feats.forEach((f) => (counts[f.properties.level] = (counts[f.properties.level] ?? 0) + 1));
  panel.innerHTML = `
    <section class="tiles">
      ${(["high", "medium", "low"] as const).map((l) => `<div class="tile">${levelBadge(l)}<b>${num(counts[l])}</b></div>`).join("")}
    </section>
    <section>
      <h2>${t("activeAlerts")}</h2>
      ${S.alerts.length ? S.alerts.map(alertCard).join("") : `<p class="muted">${t("noAlerts")}</p>`}
    </section>
    <section>
      <h2>${t("sites")} <span class="muted">(${num(feats.length)})</span></h2>
      <ul class="sitelist">${top.map((f) => siteRow(f.properties)).join("")}</ul>
    </section>
    <details class="about"><summary>${t("about")}</summary><p>${t("aboutText")}</p></details>`;
  panel.querySelectorAll<HTMLElement>("[data-site]").forEach((e) =>
    e.addEventListener("click", () => selectSite(e.dataset.site!)));
}

function siteRow(p: SiteProps): string {
  return `<li><button class="row" data-site="${esc(p.code)}">${levelBadge(p.level)}
    <span class="grow"><b>${esc(nm(p.name, p.name_ne))}</b><small>${esc(kindLabel(p.kind))} · ${esc(p.code)}</small></span>
    <span class="muted">${p.score !== null ? num(p.score * 100) : "–"}</span></button></li>`;
}

function alertCard(a: Alert): string {
  const ne = getLang() === "ne";
  return `<article class="alert lv-${esc(a.level)}" style="--lv:${LEVEL_COLOR[a.level]}">
    <header>${levelBadge(a.level)}<time>${fmtTime(a.issued_at)}</time></header>
    <h3>${esc(ne ? a.title_ne : a.title_en)}</h3>
    <details><summary>${esc(ne ? a.sms_ne : a.sms_en)}</summary><pre>${esc(ne ? a.body_ne : a.body_en)}</pre></details>
    <button class="link" data-site="${esc(a.site_code)}">${esc(nm(a.site_name, a.site_name_ne))} →</button>
  </article>`;
}

async function selectSite(code: string): Promise<void> {
  S.selected = code;
  if (S.mode === "hindcast") return renderHindcastSite(code);
  await renderSite(code);
}

function clearSelection(): void {
  setData(map, "selected-path", { type: "FeatureCollection", features: [] });
  setData(map, "exposed", { type: "FeatureCollection", features: [] });
}

function showPathAndExposure(path: GeoJSON.LineString | null, exposed: { name: string; name_ne: string; lon?: number; lat?: number; travel_time_min: number }[]): void {
  setData(map, "selected-path", path ? { type: "Feature", geometry: path, properties: {} } : { type: "FeatureCollection", features: [] });
  setData(map, "exposed", {
    type: "FeatureCollection",
    features: exposed.filter((e) => e.lon !== undefined && (e.name || e.name_ne)).slice(0, 25).map((e) => ({
      type: "Feature", geometry: { type: "Point", coordinates: [e.lon!, e.lat!] },
      properties: { label: `${nm(e.name, e.name_ne)} · ${num(e.travel_time_min)} ${t("minutes")}` },
    })),
  });
}

function reasonsHtml(reasons: { text_en: string; text_ne: string }[]): string {
  if (!reasons.length) return `<p class="muted">—</p>`;
  return `<ul class="reasons">${reasons.map((r) => `<li>${esc(getLang() === "ne" ? r.text_ne : r.text_en)}</li>`).join("")}</ul>`;
}

function exposedHtml(ex: { kind: string; name: string; name_ne: string; travel_time_min: number; path_distance_km: number }[]): string {
  const named = ex.filter((e) => e.name || e.name_ne);
  if (!named.length) return `<p class="muted">—</p>`;
  return `<table class="tbl"><thead><tr><th></th><th>${t("arrival")}</th><th>${t("km")}</th></tr></thead><tbody>
    ${named.slice(0, 12).map((e) => `<tr><td><b>${esc(nm(e.name, e.name_ne))}</b><small>${esc(e.kind)}</small></td>
      <td>${num(e.travel_time_min)} ${t("minutes")}</td><td>${num(e.path_distance_km, 1)}</td></tr>`).join("")}
  </tbody></table>`;
}

async function renderSite(code: string): Promise<void> {
  const panel = $("#panel");
  panel.innerHTML = `<div class="loading">…</div>`;
  let d: SiteDetail;
  try {
    d = await api.site(code);
  } catch (e) {
    panel.innerHTML = `<p class="error">${esc(String(e))}</p>`;
    return;
  }
  showPathAndExposure(d.flow_path, d.exposed);
  fitTo(map, [d.geometry, d.flow_path ? { type: "LineString", coordinates: d.flow_path.coordinates.slice(0, 400) } : null], 12);
  const r = d.risk;
  panel.innerHTML = `
    <button class="ghost back" id="back">← ${t("close")}</button>
    <h2>${esc(nm(d.name, d.name_ne))}</h2>
    <p class="meta">${levelBadge(d.level)} ${esc(kindLabel(d.kind))} · ${esc(d.code)}
      ${d.elevation_m ? ` · ${t("elevation")} ${num(d.elevation_m)} m` : ""}</p>
    ${r ? `<p class="muted">${t("assessed")}: ${fmtTime(r.assessed_at)} · ${t("risk")} ${num(r.score * 100)}/${num(100)}</p>` : ""}
    ${r?.potentially_dangerous_only && r.level !== "low" ? `<p class="note">${t("potentiallyDangerous")}</p>` : ""}
    <h3>${t("why")}</h3>${reasonsHtml(r?.reasons ?? [])}
    <div id="charts"></div>
    <h3>${t("downstream")}</h3>${exposedHtml(d.exposed)}
    ${d.alerts.length ? `<h3>${t("alerts")}</h3><ul class="plain">${d.alerts.map((a) => `<li>${levelBadge(a.level)} ${fmtTime(a.issued_at)} — ${esc(getLang() === "ne" ? a.title_ne : a.title_en)}</li>`).join("")}</ul>` : ""}`;
  $("#back").addEventListener("click", () => {
    S.selected = null;
    clearSelection();
    renderLive();
  });
  const [obs, ras] = await Promise.all([api.observations(code).catch(() => []), api.assessments(code).catch(() => [])]);
  const charts = $("#charts");
  drawSiteCharts(charts, d.kind, obs, ras.map((a) => ({ t: Date.parse(a.at), v: a.score * 100, label: levelLabel(a.level) })), null);
}

function drawSiteCharts(root: HTMLElement, kind: string, obs: { at: string; kind: string; sensor: string; quality: number; values: Record<string, number> }[],
                        risk: Point[], eventTime: number | null): void {
  root.innerHTML = "";
  const fmtT = (x: number) => new Date(x).toLocaleDateString(getLang() === "ne" ? "ne-NP" : "en-GB", { month: "short", day: "numeric" });
  const add = () => root.appendChild(document.createElement("div"));
  if (kind === "glacier" || kind === "slope") {
    // aggregate pairs ending at the same time (several baselines / orbits) into one weighted point
    const byT = new Map<number, { v: number; w: number }>();
    for (const o of obs) {
      if (o.kind !== "velocity" || o.values.v_down_median === undefined) continue;
      const tt = Date.parse(o.at);
      const se = o.values.v_down_se || 0.05;
      const w = 1 / (se * se);
      const cur = byT.get(tt) ?? { v: 0, w: 0 };
      cur.v += o.values.v_down_median * w;
      cur.w += w;
      byT.set(tt, cur);
    }
    const pts = [...byT.entries()].map(([tt, a]) => ({ t: tt, v: a.v / a.w, se: 1 / Math.sqrt(a.w) }));
    lineChart(add(), pts, { title: t("velocity"), unit: "m/d", zeroLine: true, eventTime, eventLabel: t("event"),
      fmt: (v) => num(v, 2), fmtT });
  }
  if (kind === "glacial_lake" || kind === "barrier_lake") {
    const pts = obs.filter((o) => o.kind === "lake_area" && o.quality >= 0.8 && o.values.area_m2 !== undefined)
      .map((o) => ({ t: Date.parse(o.at), v: o.values.area_m2 / 1e6, label: o.sensor }));
    lineChart(add(), pts, { title: t("lakeArea"), eventTime, eventLabel: t("event"), fmt: (v) => num(v, 3), fmtT });
  }
  if (risk.length) lineChart(add(), risk, { title: t("riskHistory"), eventTime, eventLabel: t("event"), fmt: (v) => num(v), fmtT, step: true });
}

// ------------------------------------------------------------------------------------------
// hindcast view
// ------------------------------------------------------------------------------------------
async function loadHindcastList(): Promise<void> {
  const panel = $("#panel");
  const list = await api.hindcasts();
  if (!list.length) {
    panel.innerHTML = `<p class="muted">${t("noHindcasts")}</p>`;
    return;
  }
  S.hcList = list;
  // prefer an explicitly requested report, then the newest one that replays a real event
  const want = new URLSearchParams(location.hash.split("?")[1] ?? "").get("name")
    ?? [...list].reverse().find((h) => h.event_time)?.name ?? list[list.length - 1].name;
  await openHindcast(list.find((h) => h.name === want)?.name ?? list[0].name);
}

async function openHindcast(name: string): Promise<void> {
  const r = await api.hindcast(name);
  S.hindcast = r;
  S.hcName = name;
  const dates = new Set<number>();
  r.sites.forEach((s) => s.timeline.forEach((e) => dates.add(Date.parse(e.at))));
  if (r.event_time) dates.add(Date.parse(r.event_time));
  S.hcDates = [...dates].sort((a, b) => a - b);
  const ev = r.event_time ? Date.parse(r.event_time) : null;
  S.hcIndex = ev ? Math.max(0, S.hcDates.findIndex((d) => d >= ev) - 1) : S.hcDates.length - 1;
  if (r.event_lonlat) {
    setData(map, "event-point", { type: "Feature", geometry: { type: "Point", coordinates: r.event_lonlat }, properties: {} });
  }
  setVisible(map, "events-dot", true);
  const [w, s, e, n] = r.aoi.bbox;
  onReady(map, () => map.fitBounds([[w, s], [e, n]], { padding: 30, duration: 500 }));
  renderHindcast();
}

function hindcastSitesAt(time: number): FC<SiteProps> {
  const r = S.hindcast!;
  return {
    type: "FeatureCollection",
    features: r.sites.map((s) => {
      let cur = { level: "unknown", score: null as number | null, at: null as string | null };
      for (const e of s.timeline) if (Date.parse(e.at) <= time) cur = { level: e.level, score: e.score, at: e.at };
      return { type: "Feature" as const, geometry: s.geometry, properties: {
        code: s.code, kind: s.kind, name: s.name, name_ne: s.name_ne, lon: s.lon, lat: s.lat, elevation_m: s.elevation_m,
        level: cur.level, score: cur.score, assessed_at: cur.at } };
    }),
  };
}

function renderHindcast(): void {
  const r = S.hindcast;
  const panel = $("#panel");
  if (!r) return;
  const now = S.hcDates[S.hcIndex] ?? Date.now();
  const ev = r.event_time ? Date.parse(r.event_time) : null;
  const fc = hindcastSitesAt(now);
  showSites(fc);
  // radar mass-movement detections as they would have appeared up to the selected time
  setData(map, "events", {
    type: "FeatureCollection",
    features: r.change_events.filter((e) => e.kind === "mass_movement" && e.area_km2 >= 0.25 && e.confidence >= 0.7
      && Date.parse(e.at) <= now).map((e) => ({
      type: "Feature" as const, geometry: { type: "Point" as const, coordinates: [e.lon, e.lat] },
      properties: { at: e.at, area: e.area_km2 } })),
  });
  const counts: Record<string, number> = {};
  fc.features.forEach((f) => (counts[f.properties.level] = (counts[f.properties.level] ?? 0) + 1));
  const issued = r.alerts.filter((a) => Date.parse(a.available_at) <= now);
  const before = ev ? r.alerts.filter((a) => Date.parse(a.available_at) < ev) : [];
  const focus = r.focus_sites.map((c) => r.sites.find((s) => s.code === c)!).filter(Boolean);
  panel.innerHTML = `
    ${S.hcList.length > 1 ? `<label class="picker">${t("selectHindcast")}
      <select id="hcpick">${S.hcList.map((h) => `<option value="${esc(h.name)}" ${h.name === S.hcName ? "selected" : ""}>${esc(h.name)}${h.event_time ? " ★" : ""}</option>`).join("")}</select></label>` : ""}
    <h2>${esc(getLang() === "ne" ? r.aoi.name_ne || r.aoi.name : r.aoi.name)}</h2>
    ${ev ? `<p class="meta"><b>${t("event")}:</b> ${fmtTime(r.event_time)}</p>` : ""}
    <div class="timeline">
      <button id="play" class="primary">${S.playing ? t("pause") : t("play")}</button>
      <input id="slider" type="range" min="0" max="${S.hcDates.length - 1}" value="${S.hcIndex}" aria-label="time">
      <output>${fmtTime(new Date(now).toISOString())}${ev ? ` <span class="${now < ev ? "pre" : "post"}">${now < ev ? t("beforeEvent") : t("afterEvent")}</span>` : ""}</output>
    </div>
    <section class="tiles">${(["high", "medium", "low"] as const).map((l) => `<div class="tile">${levelBadge(l)}<b>${num(counts[l] ?? 0)}</b></div>`).join("")}</section>
    <p class="muted">${num(r.scenes.S1)} Sentinel-1 · ${num(r.scenes.S2)} Sentinel-2${ev ? ` · ${t("alerts")} ${t("beforeEvent")}: ${num(before.length)}` : ""}</p>
    <section><h2>${t("alerts")}</h2>
      ${issued.length ? issued.slice(-12).reverse().map((a) => `<article class="alert" style="--lv:${LEVEL_COLOR[a.level]}">
        <header>${levelBadge(a.level)}<time>${fmtTime(a.available_at)}</time></header>
        <h3>${esc(getLang() === "ne" ? a.title_ne : a.title_en)}</h3>
        ${a.lead_time_h !== null ? `<p class="muted">${a.lead_time_h >= 0 ? `${t("leadTime")}: ${num(a.lead_time_h, 1)} h ${t("beforeEvent")}` : `${num(-a.lead_time_h, 1)} h ${t("afterEvent")}`}</p>` : ""}
        <details><summary>…</summary><pre>${esc(getLang() === "ne" ? a.body_ne : a.body_en)}</pre></details>
        <button class="link" data-site="${esc(a.site)}">${esc(a.site)} →</button></article>`).join("") : `<p class="muted">${t("noAlerts")}</p>`}
    </section>
    ${focus.length ? `<section><h2>${t("sites")} – ${t("event")}</h2><ul class="sitelist">${focus.map((s) => {
      const p = fc.features.find((f) => f.properties.code === s.code)!.properties;
      return siteRow(p);
    }).join("")}</ul></section>` : ""}`;
  document.querySelector<HTMLSelectElement>("#hcpick")?.addEventListener("change", (e) => {
    if (S.playing) stopPlay();
    void openHindcast((e.target as HTMLSelectElement).value);
  });
  const slider = $<HTMLInputElement>("#slider");
  slider.addEventListener("input", () => {
    S.hcIndex = Number(slider.value);
    renderHindcast();
  });
  $("#play").addEventListener("click", () => (S.playing ? stopPlay() : startPlay()));
  panel.querySelectorAll<HTMLElement>("[data-site]").forEach((e) => e.addEventListener("click", () => selectSite(e.dataset.site!)));
}

function startPlay(): void {
  if (S.hcIndex >= S.hcDates.length - 1) S.hcIndex = 0;
  S.playing = window.setInterval(() => {
    if (S.hcIndex >= S.hcDates.length - 1) return stopPlay();
    S.hcIndex += 1;
    renderHindcast();
  }, 350);
  renderHindcast();
}

function stopPlay(): void {
  if (S.playing) window.clearInterval(S.playing);
  S.playing = null;
  if (S.mode === "hindcast") renderHindcast();
}

function renderHindcastSite(code: string): void {
  const r = S.hindcast!;
  const s = r.sites.find((x) => x.code === code);
  if (!s) return;
  if (S.playing) stopPlay();
  const panel = $("#panel");
  const ev = r.event_time ? Date.parse(r.event_time) : null;
  showPathAndExposure(s.flow_path, []);
  fitTo(map, [s.geometry], 12);
  panel.innerHTML = `
    <button class="ghost back" id="back">← ${t("close")}</button>
    <h2>${esc(nm(s.name, s.name_ne))}</h2>
    <p class="meta">${esc(kindLabel(s.kind))} · ${esc(s.code)}${s.distance_to_event_km !== null ? ` · ${num(s.distance_to_event_km, 1)} ${t("km")}` : ""}</p>
    ${s.max_pre_event ? `<h3>${t("why")} (${t("beforeEvent")})</h3><p>${levelBadge(s.max_pre_event.level)} ${fmtTime(s.max_pre_event.at)}</p>${reasonsHtml(s.max_pre_event.reasons)}` : ""}
    <div id="charts"></div>
    <h3>${t("downstream")}</h3>${exposedHtml(s.exposure_top)}`;
  $("#back").addEventListener("click", () => {
    S.selected = null;
    clearSelection();
    renderHindcast();
  });
  drawSiteCharts($("#charts"), s.kind, s.observations, s.timeline.map((e) => ({ t: Date.parse(e.at), v: e.score * 100, label: levelLabel(e.level) })), ev);
}

// ------------------------------------------------------------------------------------------
// admin view
// ------------------------------------------------------------------------------------------
function getKey(): string {
  try {
    return sessionStorage.getItem("himsat.key") ?? "";
  } catch {
    return "";
  }
}

function renderAdmin(): void {
  const panel = $("#panel");
  panel.innerHTML = `
    <h2>${t("admin")}</h2>
    <form id="keyform" class="form"><label>${t("apiKey")}<input id="key" type="password" autocomplete="off" value="${esc(getKey())}"></label>
      <label>${t("officer")}<input id="officer" autocomplete="name"></label><button class="primary">OK</button></form>
    <div id="adm"></div>`;
  $("#keyform").addEventListener("submit", (e) => {
    e.preventDefault();
    try {
      sessionStorage.setItem("himsat.key", $<HTMLInputElement>("#key").value.trim());
    } catch {
      /* ignore */
    }
    void loadAdmin();
  });
  if (getKey()) void loadAdmin();
}

async function loadAdmin(): Promise<void> {
  const key = getKey();
  const box = $("#adm");
  try {
    const [queue, subs, runs] = await Promise.all([api.admin.queue(key), api.admin.subscribers(key), api.admin.runs(key)]);
    box.innerHTML = `
      <section><h3>${t("reviewQueue")} (${num(queue.length)})</h3>
        ${queue.map((a) => `<article class="alert" style="--lv:${LEVEL_COLOR[a.level]}">
          <header>${levelBadge(a.level)}<time>${fmtTime(a.issued_at)}</time><small>${esc(a.generator)}</small></header>
          <h3>${esc(a.title_ne)}</h3><h4>${esc(a.title_en)}</h4>
          <details><summary>SMS</summary><pre>${esc(a.sms_ne)}\n\n${esc(a.sms_en)}</pre></details>
          <details><summary>…</summary><pre>${esc(a.body_ne)}\n\n${esc(a.body_en)}</pre></details>
          <div class="actions"><button class="primary" data-approve="${a.id}">${t("approve")}</button>
          <button class="ghost" data-cancel="${a.id}">${t("cancel")}</button></div></article>`).join("") || `<p class="muted">—</p>`}
      </section>
      <section><h3>${t("subscribers")} (${num(subs.length)})</h3>
        <table class="tbl"><tbody>${subs.map((s) => `<tr><td><b>${esc(String(s.name))}</b><small>${esc(String(s.org_type))} · ${esc(String(s.language))}</small></td><td>${esc((s.channels as string[]).join(", "))}</td><td>${esc(String(s.min_level))}</td></tr>`).join("")}</tbody></table>
        <details><summary>${t("addSubscriber")}</summary>
          <form id="subform" class="form">
            <input name="name" placeholder="name" required>
            <select name="org_type"><option>municipality</option><option>drr_authority</option><option>hydropower</option><option>police</option><option>community</option><option>other</option></select>
            <select name="language"><option value="ne">ne</option><option value="en">en</option><option value="both">both</option></select>
            <input name="phone" placeholder="+977…"><input name="email" type="email" placeholder="email"><input name="webhook_url" placeholder="https://…">
            <select name="min_level"><option>medium</option><option>high</option></select>
            <button class="primary">${t("save")}</button></form></details>
      </section>
      <section><h3>${t("runs")}</h3><table class="tbl"><tbody>${runs.slice(0, 8).map((r) => `<tr><td>${esc(String(r.aoi))}</td><td>${esc(String(r.status))}</td><td>${fmtTime(String(r.started_at))}</td></tr>`).join("")}</tbody></table></section>`;
    const officer = () => $<HTMLInputElement>("#officer").value.trim();
    box.querySelectorAll<HTMLButtonElement>("[data-approve]").forEach((b) => b.addEventListener("click", async () => {
      if (!officer()) return alert(t("officer"));
      b.disabled = true;
      await api.admin.approve(key, Number(b.dataset.approve), officer()).catch((e) => alert(String(e)));
      void loadAdmin();
    }));
    box.querySelectorAll<HTMLButtonElement>("[data-cancel]").forEach((b) => b.addEventListener("click", async () => {
      if (!officer()) return alert(t("officer"));
      await api.admin.cancel(key, Number(b.dataset.cancel), officer()).catch((e) => alert(String(e)));
      void loadAdmin();
    }));
    $("#subform").addEventListener("submit", async (e) => {
      e.preventDefault();
      const fd = new FormData(e.target as HTMLFormElement);
      const body: Record<string, unknown> = Object.fromEntries([...fd.entries()].filter(([, v]) => v !== ""));
      body.channels = [body.phone && "sms", body.email && "email", body.webhook_url && "webhook"].filter(Boolean);
      await api.admin.addSubscriber(key, body).catch((err) => alert(String(err)));
      void loadAdmin();
    });
  } catch (e) {
    box.innerHTML = `<p class="error">${esc(String(e))}</p>`;
  }
}

// ------------------------------------------------------------------------------------------
async function main(): Promise<void> {
  setLang(getLang() as Lang);
  shell();
  map = createMap($("#map"));
  map.on("click", "site-fill", (e) => e.features?.[0] && selectSite(e.features[0].properties.code));
  map.on("click", "site-dot", (e) => e.features?.[0] && selectSite(e.features[0].properties.code));
  const h = location.hash.replace("#", "");
  const site = h.startsWith("site=") ? h.slice(5) : null;
  S.mode = h.startsWith("hindcast") ? "hindcast" : h === "admin" ? "admin" : "live";
  translate();
  await load();
  if (site) void selectSite(site);
}

void main();
