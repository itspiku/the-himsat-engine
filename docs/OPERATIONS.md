# HimSat operations runbook

This runbook is for whoever runs HimSat for a municipality, the disaster authority (NDRRMA/DDMC) or a hydropower operator.

## 1. Deploy

```bash
cp .env.example .env            # set POSTGRES_PASSWORD, HIMSAT_PUBLIC_BASE_URL, channels …
docker compose build
docker compose up -d            # db + api (map on :8000) + worker (monitoring loop)
docker compose exec api himsat admin new-key      # put the printed hash into HIMSAT_ADMIN_API_KEYS, restart api
docker compose exec api himsat aoi init rasuwa-trishuli
```

Optional self-hosted LLM for more natural alert wording:

```bash
docker compose --profile llm up -d ollama
docker compose exec ollama ollama pull qwen2.5:7b-instruct
docker compose exec api himsat admin check       # STAC, LLM, channels, segmenter
```

Without an LLM, alerts use the deterministic Nepali/English templates. They are complete and
validated, so the warning system works either way.

Without Docker: install into `/opt/himsat` (`pip install ".[postgres]"`) and use the systemd units
in `deploy/systemd/`.

Put a TLS-terminating reverse proxy (nginx, Caddy, Traefik) in front of port 8000; see
`deploy/nginx.conf` for an example with admin rate limiting. Rate-limit
`/api/admin`. The public map must stay reachable over slow mobile links: responses are
gzip-compressed and the map loads only its own JavaScript bundle (≈300 kB gzipped) plus tiles.

## 2. What runs when

| Process | What it does | Cadence |
|---|---|---|
| `worker` (`himsat watch …`) | searches new Sentinel-1/-2 acquisitions per AOI, processes them in time order, reassesses touched sites, issues alerts | every `HIMSAT_SCHEDULE_INTERVAL_MINUTES` (default 180) |
| `api` | public map, GeoJSON/CAP feeds, admin review | always |

Sentinel-1 revisits each orbit every 12 days (S1A/S1C/S1D share orbits, so the effective revisit is about 6 days), and the Rasuwa AOI is covered by three relative orbits. Sentinel-2 revisits every 2–5 days, but
it is often cloudy in the monsoon. Planetary Computer usually publishes scenes a few hours to a day after
acquisition. A 3-hour cycle therefore catches each new pass soon after publication.

## 3. Alert handling

* **HIGH** (with sufficient confidence) alerts go out automatically (`HIMSAT_AUTO_DISPATCH_LEVELS=high`).
* **MEDIUM**, low-confidence HIGH ("needs confirmation") and all-clear messages wait in the
  **review queue**: map → *Admin*, or `himsat alerts list --status pending_review`.
* The duty officer reads the Nepali and English text, checks the site on the map (outline,
  flow path, evidence chart) and approves or cancels. Approval is logged with the officer's
  name (`approved_by`).
* Every alert records the generator (`template` or `llm:<model>`) and the validator report.
* Cooldown: no repeat alert for the same site within 24 h (HIGH) or 72 h (MEDIUM) unless risk
  rises by ≥ 0.10. Escalation (MEDIUM→HIGH) is immediate.
* Failed deliveries: `himsat alerts retry`. Delivery status per subscriber:
  `GET /api/admin/alerts/{id}/deliveries`.

### Recommended subscriber set (Rasuwa example)

| Subscriber | Channels | min level | Area |
|---|---|---|---|
| Gosaikunda, Aamachhodingmo, Uttargaya, Kalika, Naukunda rural municipalities | SMS (ne) + email | medium | municipality polygon |
| DDMC Rasuwa / District Administration Office | SMS (both) + email | medium | district polygon |
| NDRRMA operations centre | webhook + email | medium | all AOIs |
| Hydropower operators (Rasuwagadhi, Upper Trishuli, Chilime/Sanjen …) | SMS + webhook | medium | plant location buffered 2 km |
| Armed Police Force posts, Rasuwagadhi customs | SMS | high | location buffer |
| Trekking lodge associations (Langtang) | SMS (both) | high | valley polygon |

Use the subscriber `area` polygon. A subscriber receives an alert if the site **or any exposed
downstream asset** lies inside their area.

## 4. Monitoring the system itself

* `GET /api/health`: DB reachable plus the last pipeline run. `status` is `degraded` (HTTP 200)
  when no monitoring run has finished within 2 × the schedule interval: point your uptime checker at
  this field.
* `GET /metrics` (Prometheus): `himsat_sites{level}`, `himsat_alerts{status}`,
  `himsat_last_run_finished_timestamp{aoi}`, HTTP latency.
  **Alert if the last finished run is older than 2 × the schedule interval.**
* Scenes that fail to process are marked `failed` in `scenes` and retried on the next cycle (`stats.error`
  holds the reason). One bad scene never stops a cycle.

## 5. Validating and tuning

* Run a hindcast over a past event before changing `config/risk.yaml`:
  `himsat hindcast <aoi> --start … --end … --event-time … --event-lon … --event-lat …`.
  Compare alerts, lead times and the number of false alarms. Bump `model_version`.
* After editing `config/risk.yaml` or the feature logic, rebuild a hindcast's assessments and
  alerts from its stored observations in minutes, without re-downloading imagery:
  `himsat reassess <aoi> --hindcast <name>`.
* Speed: the first run over a period downloads imagery (about 1–5 min per acquisition for a
  6-tile AOI). Later runs read the disk cache (`/data/cache`). Size it at roughly 1 GB per AOI-month.
* Thresholds that most affect false alarms: `windows.velocity_min_z`, the watch-cell test in
  `himsat/detect/cells.py` (z ≥ 3, ≥ 0.03 m/day, ≥ 2× baseline, ≥ 2 orbits), SAR change
  minimum area (`EVENT_MIN_AREA`, `ATTACH_MIN_AREA` in `pipeline/s1.py`), and `ALERT_MIN_DYNAMIC`.
* Curated infrastructure coordinates in `data/assets/curated.geojson` are **approximate**.
  Replace them with surveyed coordinates from the operators.

## 6. Backup and recovery

* PostgreSQL: daily `pg_dump`. Keep `alerts`, `deliveries`, `risk_assessments` for audit.
* Weekly housekeeping: `himsat admin prune` (raster cache > 120 days, velocity fields > 400 days;
  `--dry-run` shows what would go).
* `/data` volume: the raster cache (`cache/`) can be rebuilt. `products/` (S2 composites,
  velocity fields) speeds up change detection and should be backed up weekly.

## 7. InSAR (recommended for slow slope deformation)

Offset tracking sees motion of roughly ≥ 3–5 cm/day. Precursors such as the ~10 mm/month creep reported
before the 2026 Lhende Khola collapse need interferometry:

```bash
pip install "himsat[insar]"                       # or build the image with EXTRAS="postgres,insar"
export HIMSAT_INSAR_ENABLED=true HIMSAT_EARTHDATA_USERNAME=... HIMSAT_EARTHDATA_PASSWORD=...
himsat insar pairs rasuwa-lhende --start 2026-08-01    # preview SLC pairs (no login needed)
himsat insar run rasuwa-lhende --start 2026-06-01      # submit jobs, ingest finished products
```

With `HIMSAT_INSAR_ENABLED=true`, every monitoring cycle submits new 12-day pairs and ingests
finished products. HyP3 typically needs 20–60 minutes per job, so InSAR evidence arrives one cycle
later than amplitude evidence. Each interferogram is referenced to stable terrain. Per-site
line-of-sight velocities feed the same significance-gated anomaly test as other velocities
(`insar_ratio`, `insar_mm_month` in `risk.yaml`). Free HyP3 accounts have a monthly job quota:
monitor the AOIs that matter most.

## 8. Known limitations

* Sentinel-1 amplitude offset tracking resolves about 2–5 cm/day over ~1 km² after stacking.
  Slower creep (mm/month) needs the optional InSAR step (section 7). InSAR loses coherence on
  fast-changing snow and ice surfaces, so it complements offset tracking rather than replacing it.
* Optical change detection needs clear sky. During the monsoon, radar carries the monitoring.
* Lake outlines below ~0.005 km² are not tracked.
* Downstream arrival times use a constant wave speed per hazard type, so they are indicative
  only. They are not a hydrodynamic model.
