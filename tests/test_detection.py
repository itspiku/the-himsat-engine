from datetime import UTC, datetime

import numpy as np

from himsat.detect.change import detect_sar_changes, new_fractures
from himsat.detect.lakes import extract_lakes, sar_water_area
from himsat.detect.segmentation import INVALID, SNOW_ICE, WATER, SpectralSegmenter, TerrainContext
from himsat.geo.grid import Grid
from himsat.ingest.loader import S2Scene


def _scene(h=200, w=200):
    rng = np.random.default_rng(1)
    b = {k: np.full((h, w), v, "float32") + rng.normal(0, 0.005, (h, w)).astype("float32")
         for k, v in {"B02": 0.08, "B03": 0.10, "B04": 0.12, "B08": 0.25, "B11": 0.22}.items()}
    # bright glacial lake (turbid): green > NIR
    for k, v in {"B02": 0.12, "B03": 0.14, "B04": 0.10, "B08": 0.04, "B11": 0.02}.items():
        b[k][40:80, 40:90] = v
    # deep dark lake: almost black
    for k in b:
        b[k][120:150, 120:160] = 0.005
    # snow field
    for k, v in {"B02": 0.85, "B03": 0.84, "B04": 0.82, "B08": 0.70, "B11": 0.08}.items():
        b[k][0:30, 150:200] = v
    scl = np.full((h, w), 5, np.uint8)
    scl[180:200, 0:40] = 9  # cloud
    return S2Scene(datetime(2025, 10, 20, tzinfo=UTC), "S2_TEST", b, scl, 160.0, 48.0)


def test_spectral_segmentation_classes():
    sc = _scene()
    flat = np.zeros((200, 200), "float32")
    seg = SpectralSegmenter().segment(sc, TerrainContext(dem=flat + 4500, slope=flat + 2))
    assert (seg.classes[45:75, 45:85] == WATER).mean() > 0.95
    assert (seg.classes[125:145, 125:155] == WATER).mean() > 0.95
    assert (seg.classes[5:25, 155:195] == SNOW_ICE).mean() > 0.95
    assert (seg.classes[185:195, 5:35] == INVALID).all()
    assert (seg.classes[100:110, 0:30] == WATER).mean() < 0.01


def test_steep_terrain_is_not_lake():
    sc = _scene()
    steep = np.full((200, 200), 35.0, "float32")
    seg = SpectralSegmenter().segment(sc, TerrainContext(dem=steep * 0 + 4500, slope=steep))
    assert (seg.classes == WATER).sum() == 0


def test_extract_lakes_attributes():
    sc = _scene()
    flat = np.zeros((200, 200), "float32")
    g = Grid(32645, 350000, 3130000, 10, 200, 200)
    seg = SpectralSegmenter().segment(sc, TerrainContext(dem=flat + 4500, slope=flat + 2))
    lakes = extract_lakes(seg, g, flat + 4500, flat + 2, min_area_m2=5000, min_elevation_m=3000)
    areas = sorted(round(lk.area_m2 / 1e4, 1) for lk in lakes)
    assert areas == [12.0, 20.0]  # 30x40 and 40x50 pixels of 100 m²
    assert all(lk.quality > 0.95 and 28 < lk.lat < 29 for lk in lakes)
    assert min(lk.glacier_distance_m for lk in lakes) < 1000  # snow field nearby


def test_sar_water_area_measures_dark_lake():
    g = Grid(32645, 0, 0, 10, 120, 120)
    rng = np.random.default_rng(0)
    vv = rng.normal(-9, 1.5, (120, 120)).astype("float32")
    vv[40:80, 40:80] = rng.normal(-22, 1.5, (40, 40))
    ref = np.zeros((120, 120), bool)
    ref[45:75, 45:75] = True  # older, smaller outline
    area, q, mask = sar_water_area(vv, g, ref, np.zeros((120, 120)), buffer_m=200)
    assert abs(area - 1600 * 100) / (1600 * 100) < 0.1 and q > 0.8


def test_sar_change_finds_avalanche_and_ignores_band_wide_wet_snow():
    g = Grid(32645, 0, 0, 10, 300, 300)
    rng = np.random.default_rng(2)
    dem = np.tile(np.linspace(5400, 3600, 300)[:, None], (1, 300)).astype("float32")
    slope = np.full((300, 300), 30.0, "float32")
    pre = rng.normal(-10, 1.0, (300, 300)).astype("float32")
    post = pre + rng.normal(0, 0.5, (300, 300)).astype("float32")
    post[60:240, 140:180] += 6.0  # avalanche track spanning ~1100 m of relief
    post[0:25, :] -= 6.0  # wet snow over a whole elevation band
    # (on this small test grid the avalanche alone is 8% of the scene; real tiles are ~60x larger)
    regions = detect_sar_changes(pre, post, g, dem, slope, min_area_m2=20000, max_disturbed_fraction=0.2)
    assert len(regions) == 1
    r = regions[0]
    assert r.area_m2 > 50000 and r.attrs["relief_m"] > 800 and r.confidence > 0.7
    assert not r.attrs["disturbed"]


def test_sar_change_in_disturbed_scene_is_downweighted():
    g = Grid(32645, 0, 0, 10, 300, 300)
    rng = np.random.default_rng(4)
    dem = np.tile(np.linspace(5400, 3600, 300)[:, None], (1, 300)).astype("float32")
    pre = rng.normal(-10, 1.0, (300, 300)).astype("float32")
    post = pre.copy()
    for _ in range(12):  # patchy snow-related changes all over the scene
        r, c = rng.integers(0, 260, 2)
        post[r:r + 40, c:c + 30] += 5.0
    regions = detect_sar_changes(pre, post, g, dem, np.full((300, 300), 30.0, "float32"), min_area_m2=20000)
    assert regions and all(x.attrs["disturbed"] and x.confidence <= 0.5 for x in regions)


def test_new_fractures_detected():
    rng = np.random.default_rng(3)
    pre = (0.4 + rng.normal(0, 0.01, (200, 200))).astype("float32")
    post = pre.copy()
    post[100:102, 30:170] = 0.1  # a new 1.4 km dark crack
    region = np.ones((200, 200), bool)
    res = new_fractures(pre, post, region, 10.0, region)
    assert res["new_fracture_length_m"] > 1000
    assert new_fractures(pre, pre.copy(), region, 10.0, region)["new_fracture_length_m"] < 200


def test_wet_snow_darkening_is_flagged_but_avalanche_is_not():
    g = Grid(32645, 0, 0, 10, 300, 300)
    rng = np.random.default_rng(5)
    dem = np.tile(np.linspace(5400, 3600, 300)[:, None], (1, 300)).astype("float32")
    slope = np.full((300, 300), 30.0, "float32")
    pre = rng.normal(-10, 0.8, (300, 300)).astype("float32")
    post = pre.copy()
    post[20:60, 20:80] -= 6.0  # wet-snow patch: uniform darkening, ~240 m relief
    post[100:280, 200:240] += 6.0  # avalanche deposit: brightening down 1000+ m
    regions = detect_sar_changes(pre, post, g, dem, slope, min_area_m2=20000, max_disturbed_fraction=0.5)
    by_sign = {r.mean_change > 0: r for r in regions}
    assert by_sign[False].attrs["snow_like"] and not by_sign[True].attrs["snow_like"]
