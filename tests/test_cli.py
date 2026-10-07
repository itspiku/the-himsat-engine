import hashlib
import re

from typer.testing import CliRunner

from himsat.cli import app

runner = CliRunner()


def test_version():
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0 and "himsat" in r.output


def test_aoi_list_includes_hindcast_aoi():
    r = runner.invoke(app, ["aoi", "list"])
    assert r.exit_code == 0 and "rasuwa-lhende" in r.output


def test_alert_preview_template(monkeypatch):
    monkeypatch.setenv("HIMSAT_LLM_BACKEND", "none")
    from himsat.config import get_settings

    get_settings.cache_clear()
    r = runner.invoke(app, ["alert-preview"])
    get_settings.cache_clear()
    assert r.exit_code == 0
    assert "उच्च जोखिम" in r.output and "HIGH RISK" in r.output and "generator: template" in r.output


def test_admin_new_key_prints_matching_hash():
    r = runner.invoke(app, ["admin", "new-key"])
    assert r.exit_code == 0
    key = re.search(r"(hs_[A-Za-z0-9_\-]+)", r.output).group(1)
    assert hashlib.sha256(key.encode()).hexdigest() in r.output


def test_subscribers_import_and_list(tmp_path, monkeypatch):
    db = tmp_path / "cli.db"
    monkeypatch.setenv("HIMSAT_DATABASE_URL", f"sqlite:///{db.as_posix()}")
    from himsat.config import get_settings

    get_settings.cache_clear()
    roster = tmp_path / "subs.yaml"
    roster.write_text("subscribers:\n  - name: Ops\n    email: ops@example.org\n    channels: [email]\n",
                      encoding="utf-8")
    try:
        r = runner.invoke(app, ["subscribers", "import", str(roster)])
        assert r.exit_code == 0, r.output
        r = runner.invoke(app, ["subscribers", "list"])
        assert r.exit_code == 0 and "Ops" in r.output
    finally:
        get_settings.cache_clear()
