# Changelog

All notable changes to HimSat Engine are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). The risk model has its own version (`model_version` in
`config/risk.yaml`). It is stored with every assessment, so alerts can always be traced to the rules
that produced them.

## [Unreleased]

### Added
- `himsat run --backfill` loads history into the live database: results are timed as when the data
  arrived, and nothing is sent. `himsat reassess <aoi>` now also works on the live database until the
  first alert has been dispatched, and only touches that AOI.
- Same-season velocity baselines. Glacier and slope speeds are compared with the same ±30 days one
  year earlier, so normal summer speed-ups are not flagged as acceleration.
- `scripts/extend_hindcast_baseline.py`, which adds an earlier radar season to an existing hindcast.
- Final Rasuwa hindcast with a 2025 baseline season: no precursor was detectable by offset tracking.
  The earlier apparent warning was a normal summer speed-up. Radar detected the event 26 h after it
  happened. See the README.
- An evaluation summary in hindcast reports and in the replay view: whether there was a warning near
  the source, the lead time, alerts elsewhere and post-event detection time.
- Optional InSAR (ASF HyP3) line-of-sight velocities for mm-scale creep.
- YAML subscriber roster import and an example Rasuwa–Trishuli roster.
- A health endpoint that reports data freshness, a `prune` command, and daily housekeeping in
  `watch`.
- nginx and systemd deployment examples, plus a contributing guide and security policy.

### Changed
- Velocity significance tests use each site's measured repeatability as a noise floor and count
  independent acquisitions instead of pairs (risk model `2026.10-7`). In the Rasuwa hindcast this cut
  pre-event false alarms from 15 to 2.
- Fast velocity hotspots need cross-orbit confirmation and good correlation.
- Barrier-lake detection needs a confirmed mass movement and ground that was dry before.
- Radar lake areas are ignored when the known outline no longer looks like water (wind, ice cover).
- Optical crack length no longer adds to the risk score (weight 0) after it caused false alarms in the
  hindcast.
- Hindcast reports count only alerts within 2 km of the source and in the 30 days before the event as
  a warning. Every other pre-event alert counts as a potential false alarm.

### Fixed
- Planetary Computer URLs are re-signed on every read attempt, so long runs survive token expiry.
- Scenes are retried after transient remote-read failures.

## [1.0.0] - 2026-10-01

First complete release: ingest, segmentation (rules, Prithvi-EO-2.0 and SAM), lake mapping, Sentinel-1
offset tracking and watch cells, SAR and optical change detection, downstream exposure, explainable
risk, validated bilingual alerts (LLM optional), multi-channel dispatch, CAP 1.2/Atom feeds, public map,
hindcast replay, Docker deployment and CI.
