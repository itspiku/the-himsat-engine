from datetime import UTC, datetime, timedelta

import numpy as np

from himsat.config import Settings
from himsat.ingest.insar import InsarClient, SlcScene, find_pairs, los_velocity_stats, read_product
from himsat.risk.features import Obs, ice_features
from himsat.risk.model import RiskModel, load_risk_config

T = datetime(2026, 8, 24, tzinfo=UTC)


def _scene(name, days, orbit=19, frame=497):
    return SlcScene(name, T - timedelta(days=days), orbit, "DESCENDING", frame)


def test_find_pairs_same_track_consecutive():
    scenes = [_scene("a", 36), _scene("b", 24), _scene("c", 12), _scene("x", 30, orbit=85, frame=88),
              _scene("y", 18, orbit=85, frame=88), _scene("far", 100)]
    pairs = [(a.name, b.name) for a, b in find_pairs(scenes)]
    assert pairs == [("a", "b"), ("x", "y"), ("b", "c")]  # sorted by secondary time; 'far' gap too long


def test_read_product_parses_dates(tmp_path):
    d = tmp_path / "S1DD_20260812T001843_20260824T001843_VVP012_INT80_G_ueF_1A2B"
    d.mkdir()
    (d / f"{d.name}_los_disp.tif").write_bytes(b"")
    (d / f"{d.name}_corr.tif").write_bytes(b"")
    p = read_product(d)
    assert p.ref_time == datetime(2026, 8, 12, 0, 18, 43, tzinfo=UTC) and p.dt_days == 12.0
    assert p.key.startswith("INSAR_S1DD") and p.corr is not None


def test_los_velocity_is_referenced_to_stable_ground():
    rng = np.random.default_rng(0)
    los = rng.normal(0.004, 0.002, (200, 200))  # 4 mm atmospheric/orbital offset everywhere
    region = np.zeros((200, 200), bool)
    region[50:90, 50:90] = True
    los[region] += -0.012  # 12 mm away from the satellite over 12 days
    stable = ~np.pad(region, 10)[10:-10, 10:-10]
    stable[region] = False
    st = los_velocity_stats(los, np.full_like(los, 0.8), region, stable, 12.0)
    assert abs(st["v_los_m_day"] - (-0.001)) < 1e-4 and st["v_los_se"] < 2e-4 and st["coverage"] == 1.0
    assert los_velocity_stats(los, np.full_like(los, 0.1), region, stable, 12.0) is None  # incoherent


class _Job:
    def __init__(self, g):
        self.job_parameters = {"granules": list(g)}


class _FakeHyP3:
    def __init__(self):
        self.jobs, self.calls = [_Job(("a", "b"))], []

    def find_jobs(self, name=None):
        return self.jobs

    def submit_insar_job(self, g1, g2, **kw):
        self.calls.append((g1, g2, kw))


def test_submit_skips_already_submitted_pairs():
    fake = _FakeHyP3()
    client = InsarClient(Settings(insar_looks="20x4"), hyp3=fake)
    n = client.submit([(_scene("a", 24), _scene("b", 12)), (_scene("b", 12), _scene("c", 0))], "himsat-x")
    assert n == 1 and fake.calls[0][:2] == ("b", "c") and fake.calls[0][2]["include_displacement_maps"]


def test_insar_acceleration_drives_risk():
    cfg = load_risk_config()

    def o(days, v, se=0.00005):
        return Obs("insar_los", "S1", T - timedelta(days=days), {"v_los_m_day": v, "v_los_se": se, "coverage": 0.9})

    base = [o(d, -0.0002) for d in (40, 52, 64, 76)]  # ~6 mm/month creep
    feats, _ = ice_features(base + [o(2, -0.0012), o(8, -0.0011)], T, cfg, {"mean_slope_deg": 35})
    assert feats["insar_ratio"] > 3 and 30 < feats["insar_mm_month"] < 40
    r = RiskModel().assess("slope", feats, 0.9, 0.7)
    assert r.alertable and any(x["code"] == "insar_ratio" for x in r.reasons)
    quiet, _ = ice_features(base + [o(2, -0.00022), o(8, -0.00018)], T, cfg, {"mean_slope_deg": 35})
    assert "insar_ratio" not in quiet
