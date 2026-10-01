"""Spectral indices and Sentinel-2 scene-classification helpers."""

from __future__ import annotations

import numpy as np

# Sentinel-2 L2A Scene Classification (SCL) codes
SCL_NODATA, SCL_SATURATED, SCL_DARK, SCL_CLOUD_SHADOW = 0, 1, 2, 3
SCL_VEGETATION, SCL_BARE, SCL_WATER, SCL_UNCLASSIFIED = 4, 5, 6, 7
SCL_CLOUD_MED, SCL_CLOUD_HIGH, SCL_CIRRUS, SCL_SNOW = 8, 9, 10, 11


def nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / (a + b)
    out[~np.isfinite(out)] = np.nan
    return out.astype("float32")


def ndwi(b: dict[str, np.ndarray]) -> np.ndarray:
    """McFeeters NDWI (green, NIR): open water > ~0.2."""
    return nd(b["B03"], b["B08"])


def mndwi(b: dict[str, np.ndarray]) -> np.ndarray:
    """Modified NDWI (green, SWIR1). Numerically identical to NDSI."""
    return nd(b["B03"], b["B11"])


ndsi = mndwi


def ndvi(b: dict[str, np.ndarray]) -> np.ndarray:
    return nd(b["B08"], b["B04"])


def brightness(b: dict[str, np.ndarray]) -> np.ndarray:
    return ((b["B02"] + b["B03"] + b["B04"]) / 3.0).astype("float32")


def cloud_mask(scl: np.ndarray) -> np.ndarray:
    """Opaque cloud, cirrus, saturated or missing pixels."""
    return np.isin(scl, (SCL_NODATA, SCL_SATURATED, SCL_CLOUD_MED, SCL_CLOUD_HIGH, SCL_CIRRUS))


def turbidity_index(b: dict[str, np.ndarray]) -> np.ndarray:
    """Red/green ratio: rises as suspended sediment turns meltwater brown."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = b["B04"] / b["B03"]
    r[~np.isfinite(r)] = np.nan
    return r.astype("float32")
