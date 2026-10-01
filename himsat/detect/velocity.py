"""Surface velocity by image offset tracking (feature tracking).

Two co-registered images of the same scene are compared chip by chip. The displacement that
maximises their cross-correlation is the surface motion between the acquisition dates. This is
the method behind ITS_LIVE, autoRIFT and COSI-Corr, reduced to its core:

1. high-pass filter both images (removes illumination and backscatter trends)
2. extract chips on a regular lattice where motion is plausible (ice, steep high terrain)
3. normalised cross-correlation in the Fourier domain (batched), Hann-windowed
4. sub-pixel peak by 2-D parabolic interpolation
5. reject weak peaks (correlation, SNR), then spatial outliers (median test)
6. remove the co-registration residual using stable (non-moving) terrain

For Sentinel-1 the pair must share a relative orbit (identical viewing geometry). For
Sentinel-2 the pair must be clear over the target. Precision is about 0.1–0.2 pixel, i.e.
0.1 m/day for a 12-day S1 pair: enough to see the multi-fold acceleration that precedes
glacier and ice-rock collapse, but not creep in the mm/month range (that needs InSAR).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage

from himsat.geo.grid import Grid, to_lonlat


@dataclass
class VelocityField:
    grid: Grid  # image grid
    rows: np.ndarray  # lattice centre rows (1-D)
    cols: np.ndarray  # lattice centre cols (1-D)
    dx: np.ndarray  # eastward displacement, pixels (2-D lattice, NaN = no estimate)
    dy: np.ndarray  # northward displacement, pixels
    corr: np.ndarray
    snr: np.ndarray
    dt_days: float

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.dx) & np.isfinite(self.dy)

    @property
    def speed(self) -> np.ndarray:
        """Speed in metres per day (NaN where invalid)."""
        return np.hypot(self.dx, self.dy) * self.grid.res / self.dt_days

    def lattice_mask(self, mask: np.ndarray) -> np.ndarray:
        """Sample a full-resolution boolean mask at the lattice points."""
        return mask[np.ix_(self.rows, self.cols)]

    def points_lonlat(self, max_points: int = 5000) -> list[dict]:
        rr, cc = np.nonzero(self.valid)
        if rr.size > max_points:
            sel = np.linspace(0, rr.size - 1, max_points).astype(int)
            rr, cc = rr[sel], cc[sel]
        x, y = self.grid.xy(self.rows[rr], self.cols[cc])
        lon, lat = to_lonlat(np.asarray(x), np.asarray(y), self.grid.epsg)
        sp = self.speed
        return [{"lon": round(float(a), 6), "lat": round(float(b), 6), "v": round(float(sp[i, j]), 3),
                 "ve": round(float(self.dx[i, j] * self.grid.res / self.dt_days), 3),
                 "vn": round(float(self.dy[i, j] * self.grid.res / self.dt_days), 3)}
                for a, b, i, j in zip(lon, lat, rr, cc, strict=True)]


def highpass(img: np.ndarray, sigma: float) -> np.ndarray:
    """Remove low-frequency content; NaNs are filled with the local mean first."""
    a = img.astype("float32")
    nan = ~np.isfinite(a)
    if nan.all():
        return np.zeros_like(a)
    if nan.any():
        fill = np.nanmean(a)
        a = np.where(nan, fill, a)
    low = ndimage.gaussian_filter(a, sigma)
    out = a - low
    out[nan] = 0.0
    return out


def _norm_chips(ch: np.ndarray, window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = ch.mean(axis=(1, 2), keepdims=True)
    s = ch.std(axis=(1, 2), keepdims=True)
    ok = s[:, 0, 0] > 1e-6
    ch = (ch - m) / np.where(s > 1e-6, s, 1.0)
    return ch * window, ok


def _subpixel(surface: np.ndarray, pr: np.ndarray, pc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, h, w = surface.shape
    idx = np.arange(n)

    def at(dr, dc):
        return surface[idx, np.clip(pr + dr, 0, h - 1), np.clip(pc + dc, 0, w - 1)]

    c, up, dn, lf, rt = at(0, 0), at(-1, 0), at(1, 0), at(0, -1), at(0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        denr = up - 2 * c + dn
        denc = lf - 2 * c + rt
        orr = np.where(np.abs(denr) > 1e-9, 0.5 * (up - dn) / denr, 0.0)
        occ = np.where(np.abs(denc) > 1e-9, 0.5 * (lf - rt) / denc, 0.0)
    return np.clip(orr, -0.5, 0.5), np.clip(occ, -0.5, 0.5)


def offset_tracking(ref: np.ndarray, sec: np.ndarray, grid: Grid, dt_days: float, *,
                    where: np.ndarray | None = None, chip: int = 64, step: int = 16,
                    max_shift_px: float | None = None, min_corr: float = 0.15, min_snr: float = 4.0,
                    highpass_sigma: float | None = None, batch: int = 1024,
                    outlier_px: float = 1.0) -> VelocityField:
    """Track features from ``ref`` to ``sec`` (both on ``grid``). Returns displacement per lattice node."""
    assert ref.shape == sec.shape == grid.shape
    hs = highpass_sigma or chip / 6.0
    a, b = highpass(ref, hs), highpass(sec, hs)
    valid_a, valid_b = np.isfinite(ref), np.isfinite(sec)
    half = chip // 2
    rows = np.arange(half, grid.height - half, step)
    cols = np.arange(half, grid.width - half, step)
    shape = (rows.size, cols.size)
    dx = np.full(shape, np.nan, "float32")
    dy = np.full(shape, np.nan, "float32")
    corr = np.full(shape, np.nan, "float32")
    snr = np.full(shape, np.nan, "float32")
    if rows.size == 0 or cols.size == 0:
        return VelocityField(grid, rows, cols, dx, dy, corr, snr, dt_days)

    todo = np.ones(shape, bool) if where is None else where[np.ix_(rows, cols)]
    # skip chips with too much missing data in either image
    both_valid = (valid_a & valid_b).astype("float32")
    frac = ndimage.uniform_filter(both_valid, size=chip)[np.ix_(rows, cols)]
    todo &= frac > 0.9
    li, lj = np.nonzero(todo)
    if li.size == 0:
        return VelocityField(grid, rows, cols, dx, dy, corr, snr, dt_days)

    win = np.outer(np.hanning(chip), np.hanning(chip)).astype("float32")
    va = sliding_window_view(a, (chip, chip))
    vb = sliding_window_view(b, (chip, chip))
    max_shift = max_shift_px if max_shift_px is not None else chip / 4.0
    for s in range(0, li.size, batch):
        bi, bj = li[s:s + batch], lj[s:s + batch]
        r0, c0 = rows[bi] - half, cols[bj] - half
        ca, oka = _norm_chips(va[r0, c0].astype("float32"), win)
        cb, okb = _norm_chips(vb[r0, c0].astype("float32"), win)
        fa = np.fft.rfft2(ca)
        fb = np.fft.rfft2(cb)
        cc = np.fft.irfft2(np.conj(fa) * fb, s=(chip, chip))
        cc = np.fft.fftshift(cc, axes=(1, 2)) / float((win**2).sum())
        flat = cc.reshape(cc.shape[0], -1)
        k = flat.argmax(axis=1)
        pr, pc = np.unravel_index(k, (chip, chip))
        peak = flat[np.arange(flat.shape[0]), k]
        # SNR: peak over mean absolute correlation away from the peak
        mean_abs = np.abs(flat).mean(axis=1)
        sn = peak / np.maximum(mean_abs, 1e-6)
        orr, occ = _subpixel(cc, pr, pc)
        shift_r = pr + orr - half
        shift_c = pc + occ - half
        ok = oka & okb & (peak >= min_corr) & (sn >= min_snr) & (np.hypot(shift_r, shift_c) <= max_shift)
        dx[bi[ok], bj[ok]] = shift_c[ok]
        dy[bi[ok], bj[ok]] = -shift_r[ok]  # rows grow southward
        corr[bi, bj] = peak
        snr[bi, bj] = sn

    # spatial consistency: drop vectors that disagree with their neighbourhood median
    if outlier_px:
        for comp in (dx, dy):
            med = _nan_median_filter(comp, 3)
            bad = np.abs(comp - med) > outlier_px
            dx[bad] = np.nan
            dy[bad] = np.nan
    return VelocityField(grid, rows, cols, dx, dy, corr, snr, dt_days)


def _nan_median_filter(a: np.ndarray, size: int) -> np.ndarray:
    pad = size // 2
    p = np.pad(a, pad, mode="constant", constant_values=np.nan)
    win = sliding_window_view(p, (size, size))
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(win.reshape(*a.shape, -1), axis=-1)


def correct_stable_ground(vf: VelocityField, stable_mask: np.ndarray, min_points: int = 30) -> dict:
    """Subtract the median apparent motion of stable terrain (co-registration residual)."""
    st = vf.lattice_mask(stable_mask) & vf.valid
    info = {"stable_points": int(st.sum()), "offset_dx": 0.0, "offset_dy": 0.0, "stable_nmad_px": None}
    if st.sum() < min_points:
        return info
    ox, oy = float(np.median(vf.dx[st])), float(np.median(vf.dy[st]))
    vf.dx -= ox
    vf.dy -= oy
    resid = np.hypot(vf.dx[st], vf.dy[st])
    info.update(offset_dx=ox, offset_dy=oy, stable_nmad_px=float(1.4826 * np.median(np.abs(resid - np.median(resid)))))
    return info


def downslope(vf: VelocityField, aspect_deg: np.ndarray) -> np.ndarray:
    """Velocity component along the local downslope direction (m/day) at lattice nodes.

    Unlike speed magnitude this is unbiased under noise: random errors average to zero over a
    region, so coherent slow motion of a few cm/day becomes measurable.
    """
    a = np.radians(aspect_deg[np.ix_(vf.rows, vf.cols)])
    return (vf.dx * np.sin(a) + vf.dy * np.cos(a)) * vf.grid.res / vf.dt_days


def summarize_region(vf: VelocityField, region: np.ndarray, aspect_deg: np.ndarray,
                     noise_px: float | None = None) -> dict | None:
    """Robust regional motion statistics: downslope component (primary) and speed."""
    m = vf.lattice_mask(region)
    n_total = int(m.sum())
    if n_total == 0:
        return None
    ok = m & vf.valid
    n = int(ok.sum())
    if n < 3:
        return {"n_points": n, "coverage": n / n_total}
    d = downslope(vf, aspect_deg)[ok]
    sp = vf.speed[ok]
    # overlapping chips are correlated: ~ (chip/step)^2 nodes per independent sample
    se = None
    if noise_px is not None:
        n_eff = max(1.0, n / 16.0)
        se = noise_px * vf.grid.res / vf.dt_days / np.sqrt(n_eff)
    return {
        "n_points": n,
        "coverage": n / n_total,
        "v_down_median": float(np.median(d)),
        "v_down_mean": float(np.mean(d)),
        "v_down_p90": float(np.percentile(d, 90)),
        "median_m_per_day": float(np.median(sp)),
        "p90_m_per_day": float(np.percentile(sp, 90)),
        "mean_corr": float(np.nanmean(vf.corr[ok])),
        "dt_days": vf.dt_days,
        "noise_m_per_day": (noise_px * vf.grid.res / vf.dt_days) if noise_px is not None else None,
        "v_down_se": se,
    }


def summarize(vf: VelocityField, region: np.ndarray, noise_px: float | None = None) -> dict | None:
    """Speed statistics (m/day) over a full-resolution boolean region."""
    m = vf.lattice_mask(region)
    n_total = int(m.sum())
    if n_total == 0:
        return None
    ok = m & vf.valid
    n = int(ok.sum())
    if n == 0:
        return {"n_points": 0, "coverage": 0.0}
    sp = vf.speed[ok]
    res = {
        "n_points": n,
        "coverage": n / n_total,
        "median_m_per_day": float(np.median(sp)),
        "p90_m_per_day": float(np.percentile(sp, 90)),
        "max_m_per_day": float(sp.max()),
        "mean_corr": float(np.nanmean(vf.corr[ok])),
        "dt_days": vf.dt_days,
    }
    if noise_px is not None:
        res["noise_m_per_day"] = noise_px * vf.grid.res / vf.dt_days
    return res
