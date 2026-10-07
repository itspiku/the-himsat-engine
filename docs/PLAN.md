# HimSat Engine — Implementation Plan

This plan turns the proposal into a production system. It notes where the build
departs from the original pitch and explains each change.

## 1. What changed from the pitch (and why)

| Pitch | Build | Reason |
|---|---|---|
| "AI does all detection" | Every stage has an **AI path** and a **physics/rule fallback**. Outputs record which one produced them. | A warning system must never go quiet because a model checkpoint or GPU is missing. The fallbacks also serve as a baseline when evaluating the models. |
| LLM writes alerts | The LLM **drafts** bilingual alerts from a structured fact sheet. A **validator** checks every number and place name against the facts. If the draft fails, a **deterministic Nepali/English template** is sent instead. | An LLM that makes up a number or a village in a disaster alert is dangerous. The validator makes hallucinated alerts unsendable. |
| Risk "classifier" | An explainable **evidence-combination model** (noisy-OR over hazard indicators × downstream exposure). Weights and thresholds live in `config/risk.yaml`. Every score carries machine-readable reasons. | There are too few labelled GLOF/collapse events to train a trustworthy supervised classifier. Authorities need to see *why* a site is red. |
| Monitor lakes & glaciers | Also monitors **steep ice/rock slopes** anywhere in the AOI with radar offset tracking. Instability hotspots become new sites automatically. | The 26 Aug 2026 Lhende Khola source was a rock-ice slope failure, not a mapped lake. A lake-only system would have missed it. |
| Nepal only | AOIs are **catchment-based and transboundary** (Tibet headwaters included). | Water crosses borders. The 2025 and 2026 Rasuwa floods both started on or across the border. |
| Warn "downstream" | Routes the downstream path with D8 on the Copernicus DEM, runs a height-above-channel exposure test on OSM assets, and **estimates arrival time** for each settlement or hydropower plant. | "Rasuwagadhi: ~8 min" tells people what to do. "Downstream is at risk" does not. |
| Alerts | Adds **CAP 1.2** (Common Alerting Protocol) XML/Atom feeds, alongside SMS, email and signed webhooks. | CAP is the interoperable standard used by national warning systems and aggregators. |
| Auto send | A per-level dispatch policy. By default HIGH sends automatically and MEDIUM waits for duty-officer approval. Cooldowns and escalation apply. | Balances speed against false-alarm fatigue. |

## 2. Architecture

```
            ┌──────────── Sentinel-2 L2A ─┐   ┌── Sentinel-1 RTC ──┐   ┌ Copernicus DEM ┐  ┌ OSM ┐
            │  (Planetary Computer STAC)  │   │                    │   │                │  │     │
            └──────────────┬──────────────┘   └─────────┬──────────┘   └───────┬────────┘  └──┬──┘
                           ▼                            ▼                      ▼              ▼
 ingest/    STAC search → acquisition grouping → windowed COG reads onto a UTM tile grid (+ disk cache)
                           │                            │                      │              │
 detect/    segmentation (Prithvi-EO-2.0 fine-tuned ▸ SAM refine ▸ spectral fallback)         │
            lakes → polygons; SAR water in cloud     offset tracking (glacier/slope velocity) │
            optical cracks, turbidity                SAR log-ratio change (scars, deposits)   │
                           └───────────────┬────────────┘                      │              │
 sites/     match detections to site inventory, auto-discover new lakes / hotspots / barrier lakes
                                           │                                   │              │
 risk/      per-site features → hazard (noisy-OR) × exposure (flow path + assets + travel time)
                                           │
 alerts/    policy (transition, cooldown, approval) → facts → LLM draft → validator → template fallback
            dispatch: SMS (Sparrow/Twilio) · email · signed webhook · CAP feed
                                           │
 db/        SQLAlchemy (SQLite dev / PostgreSQL prod) — sites, observations, risk, alerts, deliveries
 api/       FastAPI — public GeoJSON/CAP endpoints, admin (API key) for review & subscribers
 web/       MapLibre public map (Nepali/English), site timelines, alert feed, hindcast replay
```

## 3. Lessons from real data (changes made during the build)

| Observation on 2026 Rasuwa data | Change |
|---|---|
| Single-pair S1 velocity noise is ≈0.1–0.2 m/day. The Lhende source moved only ~0.005 → 0.05–0.07 m/day before collapse. | **Watch cells** over all steep/high terrain. Downslope projection removes the noise bias of speed magnitude. 12/24/36-day pairs on 3 orbits are stacked with inverse-variance weighting. The anomaly needs z ≥ 3, ≥ 0.03 m/day, ≥ 2× baseline and ≥ 2 orbits. |
| A snowy March radar pair produced 299 change blobs, 50 "landslides" and 106 HIGH alerts. | Disturbed-scene gate (> 4% of the tile changing); **cross-orbit confirmation** before a change counts as evidence; landslide sites no longer double-count the same radar evidence. |
| Static susceptibility (lake size, glacier contact, steep walls) flagged dozens of small ponds. | Indicators are size-gated. Static-only evidence is capped at MEDIUM status. Alerts require observed change (`dynamic ≥ 0.15`). |
| Deep clear lakes are almost black after atmospheric correction, so NDWI fails (Gosainkunda was missed). | "Dark flat water" rule; DEM flatness (Copernicus DEM flattens lake surfaces) separates lakes from rivers. |
| Qwen3-8B's English drafts omitted the arrival time or the emergency number. Given empty facts it invented a phone number, a deadline and a URL. Its Nepali was ungrammatical. | Explicit required-items checklist in the prompt, a validator with one repair round, then the template. **Nepali comes from reviewed templates by default.** |
| A point-wise velocity "hotspot" test at 0.15 m/day created ~20 spurious slope sites per spring scene (snow decorrelation) and 33 HIGH alerts in a quiet month in Rolwaling. | Point-wise hotspots are kept only for fast motion (≥ 1 m/day). Slow anomalies need the watch-cell z-test with cross-orbit confirmation. |
| Glaciers speed up every melt season; against a near-zero spring baseline, ordinary 3–5 cm/day summer motion looked like a "10×" anomaly. | Ratio floor of 0.03 m/day and a minimum recent speed of 0.05 m/day. **Recommended next step:** compare with the same season of the previous year once a year of history exists. |
| The optical "new crack" metric reported 700–1,600 m of cracks between scenes three days apart (geometry and fresh snow). | Crack evidence counts only when ≥ 2 recent scenes are > 3 robust SD above the site's own history. |
| The hindcast's only "warning" (G0049, 44.6 h ahead) came from a spring baseline. In August 2025 the same glacier moved just as fast as in August 2026. | **Same-season baseline**: velocities are compared with the same ±30 days one year earlier (2025 radar processed for the hindcast). The warning disappeared: it was seasonal, not a precursor. |
| Formal offset-tracking errors (≈ 0.03–0.07 m/day) understated the real pair-to-pair scatter (≈ 0.1 m/day). Three pairs ending on one acquisition were counted as independent. | **Measured noise floor**: no pair may claim more precision than the site's own baseline scatter (robust SD), and standard errors count acquisitions, not pairs. Pre-event alerts elsewhere fell from 15 to 2. |
| OSM lacked the Rasuwagadhi border crossing and most upper-Trishuli hydropower. | Curated asset file; `at_channel` assets use height 0 above the channel. |
| Tibetan settlements in OSM often have Chinese-script names only. | Site names use only Latin or Devanagari names. |

## 4. Milestones (all implemented)

1. **Foundation**: package layout, settings, DB models and migrations, logging, CLI.
2. **Ingest**: STAC providers (Planetary Computer default, Earth Search alternative), S2 offset handling, mosaicking, tiling, cache, DEM.
3. **Detection**: indices, spectral segmenter, lake vectorisation, SAR water, offset tracking, change detection, cracks.
4. **Sites and exposure**: inventory matching, discovery, OSM asset sync, D8 downstream tracing, travel time.
5. **Risk**: features, the configurable model, reasons.
6. **Alerts**: templates (ne/en), LLM client (OpenAI-compatible: Ollama/vLLM/llama.cpp), validator, dispatch channels, CAP.
7. **Pipeline and hindcast**: an incremental monitoring cycle, plus a replay of the Rasuwa archive up to 26 Aug 2026 that produces a report.
8. **API and web map**.
9. **AI models**: Prithvi-EO-2.0 fine-tuning pipeline (weak labels from spectral rules, OSM/inventory glacier polygons and SAM refinement), inference integration and evaluation.
10. **Production**: Docker/compose, scheduler, metrics, tests, CI, docs (operations runbook, model card, data licences).
