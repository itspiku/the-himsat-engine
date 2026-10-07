"""HimSat command-line interface."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from himsat import __version__
from himsat.config import get_settings, load_aois
from himsat.util import parse_date, setup_logging

app = typer.Typer(help="HimSat Engine — satellite early warning for glacier & glacial-lake hazards",
                  no_args_is_help=True, pretty_exceptions_show_locals=False)
db_app = typer.Typer(help="Database", no_args_is_help=True)
aoi_app = typer.Typer(help="Areas of interest", no_args_is_help=True)
alerts_app = typer.Typer(help="Alerts", no_args_is_help=True)
subs_app = typer.Typer(help="Alert subscribers", no_args_is_help=True)
admin_app = typer.Typer(help="Administration", no_args_is_help=True)
ml_app = typer.Typer(help="AI models: Prithvi fine-tuning", no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(aoi_app, name="aoi")
app.add_typer(alerts_app, name="alerts")
app.add_typer(subs_app, name="subscribers")
app.add_typer(admin_app, name="admin")
app.add_typer(ml_app, name="ml")
console = Console()
log = logging.getLogger("himsat.cli")


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="debug logging")) -> None:
    s = get_settings()
    setup_logging("DEBUG" if verbose else s.log_level, s.log_json)


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"himsat {__version__}")


# --- database ---------------------------------------------------------------------------------
@db_app.command("upgrade")
def db_upgrade() -> None:
    """Apply database migrations."""
    from himsat.db.session import init_db

    init_db()
    console.print("[green]database up to date[/]")


# --- AOIs ---------------------------------------------------------------------------------------
@aoi_app.command("list")
def aoi_list() -> None:
    t = Table("id", "name", "bbox", "prefix")
    for a in load_aois().values():
        t.add_row(a.id, a.name, ", ".join(f"{v:.2f}" for v in a.bbox), a.code_prefix)
    console.print(t)


@aoi_app.command("init")
def aoi_init(aoi_id: str, refresh: bool = typer.Option(False, help="re-download OSM data")) -> None:
    """Register an AOI, sync exposed assets and create glacier sites from the inventory."""
    from himsat.config import get_aoi
    from himsat.db.session import init_db, session_scope
    from himsat.pipeline.context import AOIContext
    from himsat.pipeline.monitor import ensure_glacier_sites
    from himsat.sites import inventory as inv

    init_db()
    cfg = get_aoi(aoi_id)
    ctx = AOIContext(cfg)
    with session_scope() as s:
        inv.ensure_aoi(s, cfg)
        n = inv.sync_assets(s, cfg, refresh=refresh)
        g = ensure_glacier_sites(s, ctx, 0.5)
    console.print(f"[green]{aoi_id}[/]: {n} assets, {g} new glacier sites")


# --- pipeline -----------------------------------------------------------------------------------
@app.command()
def run(aoi_id: str,
        start: str = typer.Option(None, help="ISO date; default: since last processed scene"),
        end: str = typer.Option(None, help="ISO date; default: now"),
        no_dispatch: bool = typer.Option(False, "--no-dispatch", help="create alerts but never send them"),
        sensors: str = typer.Option("S1,S2", help="comma list: S1,S2"),
        backfill: bool = typer.Option(False, "--backfill",
                                      help="load history: alerts timed as when data arrived, never sent")) -> None:
    """Run one monitoring cycle for an AOI."""
    from himsat.pipeline.monitor import CycleOptions, run_cycle

    opts = CycleOptions(start=parse_date(start) if start else None, end=parse_date(end) if end else None,
                        dispatch=not (no_dispatch or backfill), backfill=backfill,
                        sensors=tuple(s.strip().upper() for s in sensors.split(",")),
                        progress=lambda m: console.print(m))
    res = run_cycle(aoi_id, opts)
    console.print(f"[bold]done[/] {res.stats}")


@app.command()
def watch(aoi_ids: list[str], interval_minutes: int = typer.Option(None, help="default from settings")) -> None:
    """Run monitoring cycles forever (for a container / systemd service)."""
    from himsat.maintenance import prune
    from himsat.pipeline.monitor import CycleOptions, run_cycle

    interval = (interval_minutes or get_settings().schedule_interval_minutes) * 60
    last_prune = 0.0
    while True:
        t0 = time.monotonic()
        for aoi in aoi_ids:
            try:
                res = run_cycle(aoi, CycleOptions())
                log.info("cycle %s: %s", aoi, res.stats)
            except Exception:
                log.exception("cycle %s failed", aoi)
        if time.monotonic() - last_prune > 86400:  # daily housekeeping keeps the volume bounded
            try:
                prune()
            except Exception:
                log.exception("prune failed")
            last_prune = time.monotonic()
        sleep = max(60.0, interval - (time.monotonic() - t0))
        log.info("next cycle in %.0f min", sleep / 60)
        time.sleep(sleep)


@app.command()
def hindcast(aoi_id: str,
             start: str = typer.Option(..., help="ISO date: start of archive replay"),
             end: str = typer.Option(..., help="ISO date: end of replay"),
             event_time: str = typer.Option(None, help="ISO datetime of the real event, for lead-time analysis"),
             name: str = typer.Option(None, help="hindcast name (default: <aoi>-<end>)"),
             sensors: str = typer.Option("S1,S2"),
             event_lon: float = typer.Option(None, help="event source longitude (focus the report)"),
             event_lat: float = typer.Option(None, help="event source latitude"),
             report_only: bool = typer.Option(False, help="rebuild the report from an existing hindcast DB"),
             fresh: bool = typer.Option(False, help="delete a previous hindcast DB of this name first")) -> None:
    """Replay the archive through the live pipeline (scratch DB, no dispatch) and write a report."""
    from himsat.pipeline.hindcast import run_hindcast

    ev = (event_lon, event_lat) if event_lon is not None and event_lat is not None else None
    out = run_hindcast(aoi_id, parse_date(start), parse_date(end),
                       parse_date(event_time) if event_time else None, name=name,
                       sensors=tuple(s.strip().upper() for s in sensors.split(",")), fresh=fresh,
                       progress=lambda m: console.print(m), event_lonlat=ev, report_only=report_only)
    console.print(f"[green]report:[/] {out}")


@app.command()
def reassess(aoi_id: str, hindcast_name: str = typer.Option(None, "--hindcast", help="hindcast to rebuild"),
             event_time: str = typer.Option(None), event_lon: float = typer.Option(None),
             event_lat: float = typer.Option(None)) -> None:
    """Re-run the risk model + alert policy over stored observations (after tuning risk.yaml)."""
    from himsat.pipeline.hindcast import build_report, hindcast_dir, save_report
    from himsat.pipeline.hindcast import reassess as _reassess

    if not hindcast_name:
        # live database: allowed only while nothing has been sent (e.g. after a --backfill)
        try:
            res = _reassess(aoi_id, get_settings().database_url, hindcast=True, guard_dispatched=True,
                            progress=lambda m: None)
        except RuntimeError as e:
            raise typer.BadParameter(str(e)) from e
        console.print(f"reassessed {res['times']} acquisition times, {res['alerts']} alerts")
        return
    d = hindcast_dir(hindcast_name)
    db_url = f"sqlite:///{(d / 'himsat.db').as_posix()}"
    res = _reassess(aoi_id, db_url, hindcast=True, progress=lambda m: None)
    old = json.loads((d / "report.json").read_text(encoding="utf-8")) if (d / "report.json").exists() else {}
    ev_t = parse_date(event_time) if event_time else (parse_date(old["event_time"]) if old.get("event_time") else None)
    ev_ll = (event_lon, event_lat) if event_lon is not None else (tuple(old["event_lonlat"]) if old.get("event_lonlat") else None)
    period = old.get("period") or [None, None]
    report = build_report(aoi_id, db_url, parse_date(period[0]) if period[0] else datetime.now(UTC),
                          parse_date(period[1]) if period[1] else datetime.now(UTC), ev_t, ev_ll)
    save_report(d, report)
    console.print(f"reassessed {res['times']} acquisition times, {res['alerts']} alerts -> {d / 'report.md'}")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, workers: int = 1) -> None:
    """Run the API + public map."""
    import uvicorn

    uvicorn.run("himsat.api.app:create_app", factory=True, host=host, port=port, workers=workers,
                proxy_headers=True, forwarded_allow_ips="*")


@app.command("llm-serve")
def llm_serve(model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model id"),
              host: str = "127.0.0.1", port: int = 8081, device: str = "auto", dtype: str = "auto") -> None:
    """Serve an open-weight LLM with an OpenAI-compatible API (alternative to Ollama/vLLM)."""
    import uvicorn

    from himsat.alerts.local_llm import create_app

    uvicorn.run(create_app(model, device, dtype), host=host, port=port)


@app.command("alert-preview")
def alert_preview(site_kind: str = "slope", level: str = "high") -> None:
    """Compose a sample alert with the configured LLM and show the validator result."""
    from datetime import UTC, datetime

    from himsat.alerts import phrases as P
    from himsat.alerts.compose import compose
    from himsat.alerts.facts import build_facts
    from himsat.alerts.llm import LLMClient

    class _Site:
        code, kind = "LHD-S0001", site_kind
        name = "Unstable slope above Lhende Khola (LHD-S0001)"
        name_ne = "लेन्डे खोलामाथिको अस्थिर भिर (LHD-S0001)"
        lat, lon, elevation_m = 28.2881, 85.5282, 5130.0

    reasons = [{"code": c, "text_en": P.reason_text(c, v, "en"), "text_ne": P.reason_text(c, v, "ne")}
               for c, v in (("velocity_ratio", 6.2), ("velocity_trend", 3))]
    exposures = [
        {"kind": "border_crossing", "name": "Rasuwagadhi-Kerung border crossing", "name_ne": "रसुवागढी-केरुङ नाका",
         "travel_time_min": 15.1, "path_distance_km": 22.8},
        {"kind": "hydropower", "name": "Rasuwagadhi Hydropower headworks", "name_ne": "रसुवागढी जलविद्युत हेडवर्क्स",
         "travel_time_min": 16.0, "path_distance_km": 24.0},
        {"kind": "settlement", "name": "Shyaphru Bensi", "name_ne": "स्याफ्रु बेसी", "travel_time_min": 25.3,
         "path_distance_km": 37.9}]
    s = get_settings()
    f = build_facts(alert_kind="alert", level=level, previous_level=None, site=_Site(), reasons=reasons,
                    exposures=exposures, evidence_time=datetime(2026, 8, 24, 0, 18, tzinfo=UTC),
                    needs_confirmation=False, hotline=s.emergency_hotline, base_url=s.public_base_url)
    c = compose(f, LLMClient.from_settings(s), languages=tuple(s.llm_languages))
    for k in ("title_ne", "body_ne", "sms_ne", "title_en", "body_en", "sms_en"):
        console.print(f"[bold]{k}[/]")
        console.print(c.texts[k], end="\n\n", markup=False)
    console.print(f"generator: {c.generator}")
    console.print(json.dumps(c.validation, ensure_ascii=False, indent=1))


# --- alerts ---------------------------------------------------------------------------------
@alerts_app.command("list")
def alerts_list(status: str = typer.Option(None), limit: int = 20) -> None:
    from sqlalchemy import select

    from himsat.db.models import Alert, Site
    from himsat.db.session import session_scope

    with session_scope() as s:
        q = select(Alert, Site).join(Site).order_by(Alert.created_at.desc()).limit(limit)
        if status:
            q = q.where(Alert.status == status)
        t = Table("id", "created", "site", "level", "kind", "status", "generator")
        for a, site in s.execute(q):
            t.add_row(str(a.id), f"{a.created_at:%Y-%m-%d %H:%M}", site.code, a.level, a.kind, a.status, a.generator)
    console.print(t)


@alerts_app.command("show")
def alerts_show(alert_id: int) -> None:
    from himsat.db.models import Alert
    from himsat.db.session import session_scope

    with session_scope() as s:
        a = s.get(Alert, alert_id)
        if not a:
            raise typer.Exit(1)
        console.print(f"[bold]{a.title_ne}[/]\n{a.body_ne}\n\nSMS: {a.sms_ne}\n")
        console.print(f"[bold]{a.title_en}[/]\n{a.body_en}\n\nSMS: {a.sms_en}\n")
        console.print(f"generator={a.generator} validation={json.dumps(a.validation, ensure_ascii=False)}")


@alerts_app.command("approve")
def alerts_approve(alert_id: int, user: str = typer.Option(..., help="approving duty officer")) -> None:
    from himsat.alerts.service import approve_alert
    from himsat.db.models import Alert
    from himsat.db.session import session_scope

    with session_scope() as s:
        stats = approve_alert(s, s.get(Alert, alert_id), user)
    console.print(stats)


@alerts_app.command("cancel")
def alerts_cancel(alert_id: int, user: str = typer.Option(...)) -> None:
    from himsat.alerts.service import cancel_alert
    from himsat.db.models import Alert
    from himsat.db.session import session_scope

    with session_scope() as s:
        cancel_alert(s, s.get(Alert, alert_id), user)
    console.print("cancelled")


@alerts_app.command("retry")
def alerts_retry() -> None:
    """Retry failed deliveries of dispatched alerts."""
    from sqlalchemy import select

    from himsat.alerts.service import dispatch_alert
    from himsat.db.models import Alert, Delivery
    from himsat.db.session import session_scope

    with session_scope() as s:
        ids = {d.alert_id for d in s.scalars(select(Delivery).where(Delivery.status == "failed"))}
        for i in ids:
            console.print(i, dispatch_alert(s, s.get(Alert, i)))


# --- subscribers ------------------------------------------------------------------------------
@subs_app.command("add")
def subs_add(name: str, org_type: str = "other", language: str = "ne", phone: str = None, email: str = None,
             webhook: str = None, channels: str = "sms", min_level: str = "medium", aois: str = "",
             area_geojson: Path = typer.Option(None, help="polygon file limiting the subscriber's area")) -> None:
    from himsat.db.models import Subscriber
    from himsat.db.session import init_db, session_scope
    from himsat.geo.geometry import from_geojson, set_geom

    init_db()
    with session_scope() as s:
        sub = Subscriber(name=name, org_type=org_type, language=language, phone=phone, email=email,
                         webhook_url=webhook, channels=[c.strip() for c in channels.split(",") if c.strip()],
                         min_level=min_level, aoi_ids=[a for a in aois.split(",") if a])
        if area_geojson:
            gj = json.loads(area_geojson.read_text(encoding="utf-8"))
            geom = gj["features"][0]["geometry"] if gj.get("type") == "FeatureCollection" else gj.get("geometry", gj)
            set_geom(sub, from_geojson(geom))
        s.add(sub)
        s.flush()
        console.print(f"subscriber {sub.id} added")


@subs_app.command("import")
def subs_import(path: Path, deactivate_missing: bool = typer.Option(False, help="deactivate subscribers not in the file")) -> None:
    """Create/update subscribers from a YAML roster (see config/subscribers.example.yaml)."""
    from himsat.alerts.roster import import_roster
    from himsat.db.session import init_db, session_scope

    init_db()
    with session_scope() as s:
        console.print(import_roster(s, path, deactivate_missing))


@subs_app.command("list")
def subs_list() -> None:
    from sqlalchemy import select

    from himsat.db.models import Subscriber
    from himsat.db.session import session_scope

    with session_scope() as s:
        t = Table("id", "name", "type", "lang", "channels", "min", "aois", "active")
        for sub in s.scalars(select(Subscriber)):
            t.add_row(str(sub.id), sub.name, sub.org_type, sub.language, ",".join(sub.channels), sub.min_level,
                      ",".join(sub.aoi_ids) or "*", str(sub.active))
    console.print(t)


@subs_app.command("remove")
def subs_remove(sub_id: int) -> None:
    from himsat.db.models import Subscriber
    from himsat.db.session import session_scope

    with session_scope() as s:
        sub = s.get(Subscriber, sub_id)
        if sub:
            sub.active = False
    console.print("deactivated")


# --- admin --------------------------------------------------------------------------------------
@admin_app.command("new-key")
def admin_new_key() -> None:
    """Generate an admin API key. Put its hash into HIMSAT_ADMIN_API_KEYS."""
    key = "hs_" + secrets.token_urlsafe(32)
    console.print(f"API key (give to the operator, shown once): [bold]{key}[/]")
    console.print(f"HIMSAT_ADMIN_API_KEYS entry: {hashlib.sha256(key.encode()).hexdigest()}")


@admin_app.command("check")
def admin_check() -> None:
    """Check external dependencies: STAC, LLM, SMTP/SMS configuration, models."""
    from himsat.alerts.channels import get_channels
    from himsat.alerts.llm import LLMClient
    from himsat.ingest.stac import Catalog

    s = get_settings()
    ok = True
    try:
        Catalog().client  # noqa: B018
        console.print(f"[green]OK[/] STAC {s.stac_provider}")
    except Exception as e:
        ok = False
        console.print(f"[red]FAIL[/] STAC: {e}")
    llm = LLMClient.from_settings(s)
    if llm is None:
        console.print("[yellow]-[/] LLM disabled (template alerts only)")
    elif llm.healthy():
        console.print(f"[green]OK[/] LLM {s.llm_base_url} model={s.llm_model}")
    else:
        console.print(f"[yellow]WARN[/] LLM unreachable at {s.llm_base_url}: template fallback will be used")
    for name, ch in get_channels(s).items():
        console.print(f"{'[green]OK' if ch.available else '[yellow]--'}[/] channel {name}")
    from himsat.detect.segmentation import get_segmenter

    console.print(f"[green]OK[/] segmenter: {get_segmenter(s).name}")
    console.print(f"time: {datetime.now(UTC).isoformat()}")
    raise typer.Exit(0 if ok else 1)


@admin_app.command("prune")
def admin_prune(cache_days: float = 120, velocity_days: float = 400,
                dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Delete old raster cache and velocity fields (run from cron, e.g. weekly)."""
    from himsat.maintenance import prune

    console.print(prune(cache_days=cache_days, velocity_days=velocity_days, dry_run=dry_run))


# --- InSAR ---------------------------------------------------------------------------------------
insar_app = typer.Typer(help="InSAR via ASF HyP3 (needs HIMSAT_EARTHDATA_USERNAME/PASSWORD)", no_args_is_help=True)
app.add_typer(insar_app, name="insar")


@insar_app.command("pairs")
def insar_pairs(aoi_id: str, start: str = typer.Option(...), end: str = typer.Option(None)) -> None:
    """List Sentinel-1 SLC pairs HimSat would process (no login needed)."""
    from himsat.config import get_aoi
    from himsat.ingest import insar

    cfg = get_aoi(aoi_id)
    pairs = insar.find_pairs(insar.search_slc(cfg, parse_date(start), parse_date(end) if end else datetime.now(UTC)))
    t = Table("orbit", "reference", "secondary", "days")
    for a, b in pairs:
        t.add_row(str(a.relative_orbit), a.name, b.name, f"{(b.start - a.start).days}")
    console.print(t)


@insar_app.command("run")
def insar_run(aoi_id: str, start: str = typer.Option(...), end: str = typer.Option(None)) -> None:
    """Submit new InSAR jobs and ingest finished products into the live database."""
    from himsat.alerts.llm import LLMClient
    from himsat.config import get_aoi
    from himsat.db.session import init_db
    from himsat.pipeline.assess import Assessor
    from himsat.pipeline.context import AOIContext
    from himsat.pipeline.monitor import CycleOptions, CycleResult, run_insar

    s = get_settings()
    init_db()
    ctx = AOIContext(get_aoi(aoi_id), s)
    stats = run_insar(ctx, None, parse_date(start), parse_date(end) if end else datetime.now(UTC),
                      Assessor(ctx, s, LLMClient.from_settings(s)), CycleOptions(), CycleResult(), console.print)
    console.print(stats)


# --- ML ------------------------------------------------------------------------------------------
@ml_app.command("build-dataset")
def ml_build_dataset(aois: str = typer.Option("rasuwa-trishuli,rolwaling-tamakoshi,khumbu-dudhkoshi,manaslu-marsyangdi"),
                     start: str = typer.Option("2025-10-01"), end: str = typer.Option("2025-12-15"),
                     out: Path = typer.Option(Path("data/ml/weak-v1")), scenes_per_aoi: int = 2,
                     max_cloud: float = 10.0) -> None:
    """Build the weak-label training set from clear post-monsoon Sentinel-2 scenes."""
    from himsat.ml.dataset import build_dataset

    stats = build_dataset([a.strip() for a in aois.split(",")], parse_date(start), parse_date(end), out,
                          scenes_per_aoi=scenes_per_aoi, max_cloud=max_cloud, progress=lambda m: console.print(m))
    console.print(stats)


@ml_app.command("train")
def ml_train(data: Path = typer.Option(Path("data/ml/weak-v1")), out: Path = typer.Option(Path("models/prithvi-himsat-v1.pt")),
             epochs: int = 8, batch: int = 8, lr: float = 3e-4, unfreeze_blocks: int = 4, workers: int = 2) -> None:
    """Fine-tune Prithvi-EO-2.0 (needs a CUDA GPU with >= 2 GB free, or patience on CPU)."""
    from himsat.ml.train import train

    s = get_settings()
    res = train(data, out, backbone=s.prithvi_backbone, epochs=epochs, batch=batch, lr=lr,
                unfreeze_blocks=unfreeze_blocks, device=s.resolved_device(), workers=workers,
                progress=lambda m: console.print(m))
    console.print(f"best val mIoU {res['best_val_miou']:.3f} -> {out}  (set HIMSAT_PRITHVI_CHECKPOINT={out})")


@ml_app.command("benchmark")
def ml_benchmark(checkpoint: Path = typer.Option(None), sam: str = typer.Option(None, help="e.g. facebook/sam-vit-base"),
                 out: Path = typer.Option(Path("docs/benchmark_lakes.json"))) -> None:
    """Compare lake areas (rules / Prithvi / +SAM) with published values for reference lakes."""
    from himsat.ml.benchmark import default_segmenters, run

    rows = run(default_segmenters(checkpoint, sam, get_settings().resolved_device()), progress=console.print)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    console.print(f"written {out}")


if __name__ == "__main__":
    app()
