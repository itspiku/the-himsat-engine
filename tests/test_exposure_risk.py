from datetime import timedelta

import numpy as np

from himsat.geo.grid import Grid, to_lonlat
from himsat.risk.exposure import (
    ExposureParams,
    exposed_assets,
    exposure_score,
    flow_path_from_lonlat,
    trace_downstream,
)
from himsat.risk.features import Obs, ice_features, lake_features
from himsat.risk.model import RiskModel, load_risk_config, ramp


def _valley(n=300):
    """V-shaped valley draining south with a pit (lake) halfway down."""
    r, c = np.mgrid[0:n, 0:n].astype(float)
    dem = 3000 - 5.0 * r + 6.0 * np.abs(c - n / 2) ** 1.3
    dem[100:110, 145:156] = dem[99, 150] - 20  # closed depression
    return dem.astype("float32")


def test_trace_escapes_pits_and_reaches_edge():
    dem = _valley()
    path = trace_downstream(dem, (5, 150), 30.0, max_km=100)
    assert path[-1][0] == dem.shape[0] - 1  # left through the bottom edge
    assert all(abs(c - 150) <= 3 for r, c in path if not 99 <= r <= 111)  # valley floor (pit floor is flat)


def test_exposure_uses_height_above_channel_and_arrival_time():
    g = Grid(32645, 340000, 3140000, 30.0, 300, 300)
    dem = _valley()
    lon0, lat0 = to_lonlat(*g.xy(5, 150), g.epsg)
    path = flow_path_from_lonlat(dem, g, float(lon0), float(lat0))
    assert path.length_km > 8

    def asset(i, r, c, kind="settlement", **kw):
        lon, lat = to_lonlat(*g.xy(r, c), g.epsg)
        return {"id": i, "kind": kind, "name": f"A{i}", "name_ne": "", "lon": float(lon), "lat": float(lat), **kw}

    assets = [asset(1, 200, 152), asset(2, 200, 230), asset(3, 250, 150, "hydropower", at_channel=True)]
    ex = exposed_assets(path, assets, dem, ExposureParams(), wave_speed_m_s=10.0)
    ids = [e.asset_id for e in ex]
    assert ids == [1, 3]  # asset 2 is high up the valley side
    e1 = ex[0]
    assert 5.5 < e1.path_distance_km < 6.5 and abs(e1.travel_time_min - e1.path_distance_km * 100 / 60) < 0.2
    score, detail = exposure_score(ex, {"settlement": 3, "hydropower": 6}, 12.0, 180.0)
    assert 0.4 < score < 1.0 and detail["counts"] == {"settlement": 1, "hydropower": 1}


def test_ramp():
    assert ramp(0, 1, 2) == 0 and ramp(3, 1, 2) == 1 and ramp(1.5, 1, 2) == 0.5
    assert ramp(50, 1000, 50) == 1.0 and ramp(1000, 1000, 50) == 0.0  # decreasing ramp


def test_static_susceptibility_is_not_alertable():
    m = RiskModel()
    r = m.assess("glacial_lake", {"lake_area_km2": 0.8, "glacier_contact_m": 20, "steep_walls_deg": 50}, 0.9, 0.8)
    assert r.level == "medium" and not r.alertable


def test_acceleration_with_exposure_is_high_and_explained():
    m = RiskModel()
    r = m.assess("slope", {"velocity_ratio": 5.0, "velocity_trend": 3, "source_slope_deg": 35}, 0.9, 0.8)
    assert r.level == "high" and r.alertable
    assert r.reasons[0]["code"] == "velocity_ratio" and "5" in r.reasons[0]["text_en"]
    assert "गुणा" in r.reasons[0]["text_ne"]


def test_low_confidence_high_is_downgraded():
    r = RiskModel().assess("slope", {"velocity_ratio": 5.0, "new_fractures_m": 2000}, 1.0, 0.1)
    assert r.level == "medium" and r.needs_confirmation


def test_lake_growth_features(t0):
    cfg = load_risk_config()
    obs = [Obs("lake_area", "S2", t0 - timedelta(days=45), {"area_m2": 100000}, 0.95),
           Obs("lake_area", "S2", t0 - timedelta(days=40), {"area_m2": 102000}, 0.95),
           Obs("lake_area", "S2", t0 - timedelta(days=3), {"area_m2": 130000, "glacier_distance_m": 40}, 0.95),
           Obs("lake_area", "S2", t0 - timedelta(days=2), {"area_m2": 50000}, 0.3)]  # cloudy: ignored
    f, conf = lake_features(obs, t0, cfg, {"max_surrounding_slope_deg": 38})
    assert abs(f["lake_growth_recent_pct"] - 28.7) < 1 and f["lake_area_km2"] == 0.13
    assert f["glacier_contact_m"] == 40 and conf > 0.7


def test_ice_features_require_significance(t0):
    cfg = load_risk_config()

    def obs(days, v, se=0.015):
        return Obs("velocity", "S1", t0 - timedelta(days=days), {"v_down_median": v, "v_down_se": se,
                                                                  "coverage": 1, "n_points": 50}, 1.0)

    base = [obs(d, 0.005) for d in range(40, 140, 12)]
    f, _ = ice_features(base + [obs(2, 0.13), obs(6, 0.12), obs(10, 0.11)], t0, cfg, {"mean_slope_deg": 30})
    assert f["velocity_ratio"] > 3.5 and f["velocity_z"] > 3 and f["velocity_trend"] >= 3
    f2, _ = ice_features(base + [obs(2, 0.03, 0.05), obs(6, 0.01, 0.05)], t0, cfg, {"mean_slope_deg": 30})
    assert f2["velocity_ratio"] == 1.0  # not significant


def test_fracture_evidence_needs_persistent_anomaly(t0):
    from himsat.risk.features import _event_features

    cfg = load_risk_config()

    def surf(days, v):
        return Obs("surface_change", "S2", t0 - timedelta(days=days), {"new_fracture_length_m": v,
                                                                     "observed_fraction": 0.9}, 0.9)

    feats: dict = {}
    _event_features([surf(3, 900.0)], t0, cfg["windows"], feats)
    assert "new_fractures_m" not in feats  # no history: one noisy value never counts
    hist = [surf(d, v) for d, v in ((60, 300), (90, 350), (120, 250), (150, 400), (200, 320))]
    _event_features(hist + [surf(3, 900.0)], t0, cfg["windows"], feats)
    assert "new_fractures_m" not in feats  # a single anomalous scene is not persistent
    _event_features(hist + [surf(3, 1500.0), surf(12, 1400.0)], t0, cfg["windows"], feats)
    assert feats["new_fractures_m"] > 1000


def test_radar_lake_area_far_below_outline_is_ignored(t0):
    cfg = load_risk_config()

    def s1(days, area):
        return Obs("lake_area", "S1", t0 - timedelta(days=days), {"area_m2": area, "reference_area_m2": 1_500_000}, 0.9)

    obs = [s1(50, 1_480_000), s1(40, 1_500_000), s1(3, 60_000), s1(5, 1_520_000)]
    f, _ = lake_features(obs, t0, cfg, {})
    assert abs(f["lake_growth_recent_pct"]) < 5  # the 0.06 km² wind artefact does not count


def test_ice_features_prefer_same_season_baseline(t0):
    cfg = load_risk_config()

    def obs(days, v, se=0.015):
        return Obs("velocity", "S1", t0 - timedelta(days=days), {"v_down_median": v, "v_down_se": se,
                                                                  "coverage": 1, "n_points": 50}, 1.0)

    spring = [obs(d, 0.01) for d in range(40, 140, 12)]
    now = [obs(2, 0.12), obs(6, 0.11), obs(10, 0.12)]
    last_year = [obs(365 + d, 0.115) for d in (-12, 0, 12)]
    f, _ = ice_features(spring + now + last_year, t0, cfg, {"mean_slope_deg": 30})
    assert f["velocity_seasonal_baseline"] == 1.0 and f["velocity_ratio"] == 1.0
    f2, _ = ice_features(spring + now, t0, cfg, {"mean_slope_deg": 30})
    assert f2["velocity_seasonal_baseline"] == 0.0 and f2["velocity_ratio"] > 3
