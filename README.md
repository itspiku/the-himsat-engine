# HimSat Engine

**Open-source satellite early warning for Himalayan glacier, glacial-lake and ice/rock-slope hazards.**

HimSat watches every glacier, glacial lake and steep ice/rock slope in a catchment. It uses free
Copernicus **Sentinel-1 radar**, which sees through monsoon cloud, and **Sentinel-2 optical**
imagery. It maps lakes, measures how fast ice and rock move, detects fresh avalanche and landslide
scars, traces the downstream flood path to every settlement and hydropower plant, and issues
bilingual **Nepali/English** warnings with arrival-time estimates by SMS, e-mail, signed webhooks
and the **CAP 1.2** standard feed.

It was built in response to the 26 August 2026 Lhende Khola rock-ice avalanche and debris flood
(Rasuwa–Trishuli corridor). Nobody was watching the upstream ice. HimSat makes watching it
automatic, cheap and transboundary.

> **Status:** v1.0. The full pipeline runs on real archive data, with tests, Docker deployment
> and docs. See [Validation & limitations](#validation--limitations) for what the system can and
> cannot detect before operational use.

---

## What it does

| Stage | How | Module |
|---|---|---|
| Ingest | Planetary Computer STAC (S2 L2A, S1 RTC, Copernicus DEM), windowed COG reads on a UTM tile grid with disk cache, parallel I/O | `himsat/ingest` |
| Map lakes, ice, debris | **Prithvi-EO-2.0** (NASA/IBM) fine-tuned on weak labels → **SAM** outline refinement → physics rules as guaranteed fallback | `himsat/ml`, `himsat/detect/segmentation.py` |
| Lake area under cloud | Sentinel-1 dark-water mapping (local Otsu) around known lakes | `detect/lakes.py` |
| Ice / rock motion | Sentinel-1 offset tracking (batched FFT cross-correlation, sub-pixel), downslope projection, stable-terrain correction, 12/24/36-day pairs on 3 orbits | `detect/velocity.py` |
| Watch every slope | ~1 km² **watch cells** over all steep, high terrain; inverse-variance stacking, z-score + cross-orbit anomaly test → new "unstable slope" sites | `detect/cells.py` |
| Scars, cracks, turbidity | S1 log-ratio change (disturbed-scene gate + cross-orbit confirmation), S2 ridge-filter fractures, red/green turbidity, NDVI loss | `detect/change.py` |
| Downstream exposure | D8 flow path with pit escape over GLO-30, height-above-channel corridor test, arrival times per hazard type, OSM + curated assets | `risk/exposure.py` |
| Risk | Explainable hazard (noisy-OR of indicators, size-gated) × exposure; alerts need *observed change* | `risk/`, `config/risk.yaml` |
| Alerts | Facts → self-hosted LLM draft (optional) → strict validator → Nepali/English templates; review queue; Sparrow SMS / Twilio / SMTP / webhooks; CAP 1.2 + Atom | `alerts/` |
| Public map | MapLibre, Nepali/English, live risk, alert feed, site evidence charts, flow paths, hindcast replay, admin review console | `web/` |
| Hindcast | The live pipeline replayed over the archive into a scratch DB → report + replay | `pipeline/hindcast.py` |

```
 Sentinel-1/2 + DEM ─► tiles ─► segmentation · velocity · watch cells · change ─► sites & observations
        OSM assets ─► flow paths & exposure ──────────────────────────────────────┐        │
                                                                                   ▼        ▼
                                               risk (hazard × exposure, reasons) ─► alert policy
                                                                                   │
                   LLM draft ─► validator ─► template fallback ─► review/auto ─► SMS · e-mail · webhook · CAP
```

See [docs/PLAN.md](docs/PLAN.md) for the design decisions and [docs/OPERATIONS.md](docs/OPERATIONS.md) for the runbook.

---

## Quick start (development)

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                            # add ,ml for Prithvi/SAM (needs PyTorch)
himsat db upgrade
himsat aoi init rasuwa-lhende                      # assets + glacier inventory
himsat run rasuwa-lhende --start 2026-09-01 --no-dispatch
himsat serve                                       # API + map on http://localhost:8000
```

Web map development: `cd web && npm install && npm run dev` (proxies `/api` to :8000).

### Hindcast the 26 August 2026 event

```bash
himsat hindcast rasuwa-lhende --start 2026-03-01 --end 2026-09-10 \
  --event-time 2026-08-26T02:52:00+00:00 --event-lon 85.5282 --event-lat 28.2881 --name rasuwa-2026
```

The report is written to `data/hindcasts/rasuwa-2026/report.md`. Open the map's **Hindcast** tab to replay it.

### Production

```bash
cp .env.example .env && $EDITOR .env
docker compose up -d                              # PostgreSQL + API/map + monitoring worker
docker compose --profile llm up -d ollama         # optional self-hosted LLM
```

---

## AI components

| Component | Model | Role | Without it |
|---|---|---|---|
| Segmentation | Prithvi-EO-2.0-100M-TL + light decoder, fine-tuned (`himsat ml build-dataset`, `himsat ml train`): val mIoU 0.70 vs held-out weak labels | water, snow/ice, **debris-covered ice** (invisible to spectral rules) | physics rules |
| Outline refinement | SAM (ViT-B) with box prompts, accepted only if IoU ≥ 0.6 with the coarse mask | sharper lake areas | coarse outlines |
| Anomaly detection | statistical (watch cells, z-scores, cross-orbit confirmation) | what is unusual for *this* slope | – |
| Risk model | explainable evidence combination | why a site is red | – |
| Alert wording | any open-weight LLM behind an OpenAI-compatible API (Ollama, vLLM, LM Studio, `himsat llm-serve`) | natural English text | validated templates |

**Lake-area benchmark** against published areas of Gosainkunda, Tsho Rolpa, Imja Tsho and Thulagi
(`himsat ml benchmark`): mean absolute error **rules 12.6 % · Prithvi 10.6 % · rules + SAM 8.6 % ·
Prithvi + SAM 10.3 %**. Prithvi is better on dark or turbid lakes but overestimates large ones.
Details are in the [model card](docs/MODEL_CARD.md). The recommended lake-area setting is rules + SAM
(`HIMSAT_SEGMENTER=spectral`, `HIMSAT_SAM_MODEL=facebook/sam-vit-base`).

**LLM safety.** The LLM only sees a fact sheet. Its output is rejected if it contains any number
not in the facts, an unknown URL, a missing site, place, arrival time, risk level or emergency
number, or text in the wrong script. A rejected draft gets one repair attempt, then the
deterministic template is used. Tested with Qwen3-8B (Q4): its first drafts omitted the arrival
time, and given empty facts it invented "within 24 hours", a false emergency number and a fake
URL. The validator caught all of these. Its Nepali was grammatically wrong in ways no automatic
check can catch, so **Nepali text comes from the reviewed templates by default**
(`HIMSAT_LLM_LANGUAGES=en`). Enable LLM Nepali only after native-speaker evaluation of your model.

---

## Validation & limitations

* **Hindcast of 26 Aug 2026:** see [the report](data/hindcasts/rasuwa-2026/report.md) and the summary below.
* Sentinel-1 offset tracking + watch-cell stacking detects ≈ 2–5 cm/day of coherent motion over
  ~1 km². Slower precursors (the published InSAR signal was ~10 mm/month) need interferometry
  (ASF HyP3). That integration is on the roadmap.
* Optical evidence (cracks, turbidity, lake outlines) is unavailable under monsoon cloud. Radar
  carries the monitoring then.
* Arrival times assume a constant flood-front speed per hazard type, so they are indicative only.
  They are not a hydrodynamic model.
* Curated infrastructure coordinates are approximate. Replace them with surveyed data.
* Alerts are advisory. HIGH alerts can be auto-dispatched; everything else goes through a duty officer.

## Repository layout

```
himsat/            Python package (ingest, detect, ml, risk, sites, alerts, pipeline, api, cli)
config/            aois.yaml (areas of interest), risk.yaml (risk model, versioned)
data/assets/       curated infrastructure      data/osm/  cached OSM extracts (ODbL)
data/hindcasts/    hindcast reports
web/               MapLibre public map (TypeScript, Vite)
deploy/            container entrypoint         docs/      plan, operations, model card, licences
tests/             pytest suite (no network needed)
```

## Licence

Code: Apache-2.0. Data sources and their licences: [docs/DATA_LICENSES.md](docs/DATA_LICENSES.md).
Contains modified Copernicus Sentinel data (2025–2026). © OpenStreetMap contributors.
