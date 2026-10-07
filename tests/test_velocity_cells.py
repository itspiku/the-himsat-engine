from datetime import UTC, datetime, timedelta

import numpy as np
from scipy import ndimage

from himsat.detect.cells import anomaly, cell_stats, weighted_mean
from himsat.detect.velocity import correct_stable_ground, downslope, offset_tracking, summarize_region
from himsat.geo.grid import Grid, iter_tiles


def _shift(img, dr, dc):
    f = np.fft.fft2(img)
    ky = np.fft.fftfreq(img.shape[0])[:, None]
    kx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft2(f * np.exp(-2j * np.pi * (ky * dr + kx * dc))))


def _pair(dr=0.8, dc=-1.3, noise=0.3, size=480, seed=0):
    rng = np.random.default_rng(seed)
    base = ndimage.gaussian_filter(rng.normal(size=(size, size)), 1.5)
    moving = np.zeros(base.shape, bool)
    moving[120:360, 120:360] = True
    sec = base.copy()
    sec[moving] = _shift(base, dr, dc)[moving]
    sec += noise * rng.normal(size=base.shape) * base.std()
    return base, sec, moving


def test_offset_tracking_recovers_subpixel_shift():
    ref, sec, moving = _pair()
    g = Grid(32645, 0, 0, 10, ref.shape[1], ref.shape[0])
    vf = offset_tracking(ref, sec, g, 12.0)
    info = correct_stable_ground(vf, ~ndimage.binary_dilation(moving, iterations=40))
    assert info["stable_points"] > 30
    core = np.zeros(ref.shape, bool)
    core[160:320, 160:320] = True
    m = vf.lattice_mask(core) & vf.valid
    assert m.sum() > 20
    # east = +col, north = -row
    assert abs(np.nanmedian(vf.dx[m]) - (-1.3)) < 0.1
    assert abs(np.nanmedian(vf.dy[m]) - (-0.8)) < 0.1


def test_downslope_projection_is_signed():
    ref, sec, moving = _pair(dr=1.0, dc=0.0)  # features move 1 px south
    g = Grid(32645, 0, 0, 10, ref.shape[1], ref.shape[0])
    vf = offset_tracking(ref, sec, g, 10.0)
    south = np.full(ref.shape, 180.0, "float32")
    north = np.zeros(ref.shape, "float32")
    core = np.zeros(ref.shape, bool)
    core[160:320, 160:320] = True
    s_south = summarize_region(vf, core, south, noise_px=0.05)
    s_north = summarize_region(vf, core, north, noise_px=0.05)
    assert abs(s_south["v_down_median"] - 1.0) < 0.1  # 1 px * 10 m / 10 days
    assert abs(s_north["v_down_median"] + 1.0) < 0.1
    assert np.isfinite(downslope(vf, south)).any()


def test_cell_stats_and_ids():
    ref, sec, _ = _pair(dr=1.0, dc=0.0, size=400)
    g = Grid(32645, 0, 4000, 10, 400, 400)
    vf = offset_tracking(ref, sec, g, 10.0)
    tile = next(iter_tiles(g, 400, 0))
    cells = cell_stats(vf, np.full(g.shape, 180.0), np.ones(g.shape, bool), tile, g, noise_px=0.1)
    assert cells and all(c.se > 0 and c.n >= 8 for c in cells)
    assert {c.cell for c in cells} <= {f"{r}_{c}" for r in range(4) for c in range(4)}


def test_weighted_mean_rejects_outlier():
    m, se = weighted_mean([0.05, 0.06, 0.04, 2.0], [0.02, 0.02, 0.02, 0.02])
    assert abs(m - 0.05) < 0.01 and se < 0.02


def _rows(t, base_v, recent_v, se=0.02, orbits=(19, 85, 121)):
    rows = []
    for k in range(10):  # baseline: 40..130 days back
        rows.append((t - timedelta(days=40 + 10 * k), base_v, se, orbits[k % 3]))
    for k, o in enumerate(orbits):  # recent: last 12 days, every orbit
        rows.append((t - timedelta(days=2 + 4 * k), recent_v, se, o))
    return rows


def test_anomaly_detects_significant_cross_orbit_speedup():
    t = datetime(2026, 8, 24, tzinfo=UTC)
    a = anomaly(_rows(t, 0.005, 0.12), t)  # Lhende-like precursor
    assert a is not None and a.significant and a.ratio == 4.0 and a.orbits_confirming == 3


def test_anomaly_ignores_noise_and_single_orbit():
    t = datetime(2026, 8, 24, tzinfo=UTC)
    assert not anomaly(_rows(t, 0.005, 0.02), t).significant  # too small
    assert not anomaly(_rows(t, 0.005, 0.04), t).significant  # ordinary melt-season speed-up
    a = anomaly(_rows(t, 0.005, 0.08, orbits=(19, 19, 19)), t)
    assert a is not None and not a.significant  # only one viewing geometry confirms
    assert anomaly(_rows(t, 0.3, 0.4), t).ratio < 2  # normal glacier seasonal speed-up


def test_seasonal_baseline_cancels_normal_summer_speedup():
    t = datetime(2026, 7, 15, tzinfo=UTC)
    spring = [(t - timedelta(days=40 + 10 * k), 0.01, 0.02, (19, 85, 121)[k % 3]) for k in range(6)]
    now = [(t - timedelta(days=2 + 4 * k), 0.12, 0.02, o) for k, o in enumerate((19, 85, 121))]
    # without last year's data, a summer speed-up looks anomalous...
    assert anomaly(spring + now, t).significant
    # ...but the same glacier moved just as fast last summer: no anomaly
    last_summer = [(t - timedelta(days=365 + d), 0.11, 0.02, o) for d, o in ((-10, 19), (0, 85), (10, 121))]
    a = anomaly(spring + now + last_summer, t)
    assert a is not None and not a.significant and abs(a.v_base - 0.11) < 0.01
    # a slope that was quiet last summer and is fast now is still flagged
    quiet_last = [(t - timedelta(days=365 + d), 0.01, 0.02, o) for d, o in ((-10, 19), (0, 85), (10, 121))]
    assert anomaly(spring + now + quiet_last, t).significant
