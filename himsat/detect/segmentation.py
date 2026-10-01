"""Pixel-wise land-surface segmentation of Sentinel-2 scenes.

Classes: water (lakes), clean snow/ice, debris-covered ice, other, invalid (cloud / no data).

Implementations:

* :class:`SpectralSegmenter`: physics rules (NDWI / MNDWI / NDSI with DEM slope and shadow
  constraints). Needs no model, always available. It is also the weak-label teacher for
  fine-tuning.
* ``himsat.ml.prithvi.PrithviSegmenter``: fine-tuned NASA/IBM Prithvi-EO-2.0 foundation model.
* ``himsat.ml.sam.SamRefiner``: wraps any segmenter and sharpens lake outlines with Segment Anything.

``get_segmenter()`` picks the best available option and falls back safely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy import ndimage

from himsat.config import Settings, get_settings
from himsat.detect import indices as ix
from himsat.ingest.loader import S2Scene

log = logging.getLogger(__name__)

OTHER, WATER, SNOW_ICE, DEBRIS_ICE, INVALID = 0, 1, 2, 3, 255
CLASS_NAMES = {OTHER: "other", WATER: "water", SNOW_ICE: "snow_ice", DEBRIS_ICE: "debris_ice", INVALID: "invalid"}
N_CLASSES = 4  # trainable classes (INVALID excluded)


@dataclass
class TerrainContext:
    """Terrain layers resampled to the scene grid."""

    dem: np.ndarray
    slope: np.ndarray
    shadow: np.ndarray | None = None  # cast + self shadow for this acquisition's sun geometry
    glacier_prior: np.ndarray | None = None  # inventory glacier outlines (bool), optional


@dataclass
class Segmentation:
    classes: np.ndarray  # uint8 class codes
    prob: dict[int, np.ndarray] = field(default_factory=dict)  # per-class probability (float32)
    method: str = "spectral"

    def mask(self, cls: int) -> np.ndarray:
        return self.classes == cls

    def valid_fraction(self) -> float:
        return float((self.classes != INVALID).mean())


class Segmenter(Protocol):
    name: str

    def segment(self, scene: S2Scene, terrain: TerrainContext) -> Segmentation: ...


def _sig(x: np.ndarray, center: float, width: float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return (1.0 / (1.0 + np.exp(-(x - center) / width))).astype("float32")


class SpectralSegmenter:
    """Rule-based segmentation tuned for high-mountain Sentinel-2 L2A.

    Water: NDWI > 0.25 and MNDWI > 0.1, dark in NIR, on gentle terrain (lakes are flat, which
    removes most shadow and wet-rock false positives). In terrain shadow the NDWI threshold is
    raised, because diffuse sky light makes shadows look bluish.
    Snow/ice: NDSI > 0.4 with NIR > 0.15 (Hall et al. 1995).
    Debris-covered ice cannot be separated spectrally. Within the glacier inventory prior,
    non-snow, non-water pixels are labelled debris ice.
    """

    name = "spectral"

    def __init__(self, ndwi_thr: float = 0.25, max_lake_slope: float = 18.0):
        self.ndwi_thr = ndwi_thr
        self.max_lake_slope = max_lake_slope

    def segment(self, scene: S2Scene, terrain: TerrainContext) -> Segmentation:
        b = scene.bands
        w = ix.ndwi(b)
        m = ix.mndwi(b)
        nir = b["B08"]
        missing = ~np.isfinite(w) | ~np.isfinite(m) | ~np.isfinite(b["B11"])
        cloud = ix.cloud_mask(scene.scl)
        shadow = terrain.shadow if terrain.shadow is not None else np.zeros(w.shape, bool)
        scl_shadow = scene.scl == ix.SCL_CLOUD_SHADOW

        w0 = np.nan_to_num(w, nan=-1)
        m0 = np.nan_to_num(m, nan=-1)
        nir0 = np.nan_to_num(nir, nan=1)
        thr = np.where(shadow | scl_shadow, self.ndwi_thr + 0.2, self.ndwi_thr)
        flat = terrain.slope < self.max_lake_slope
        swir0 = np.nan_to_num(b["B11"], nan=1)
        vis0 = np.nan_to_num(ix.brightness(b), nan=1)
        p_flat = _sig(-terrain.slope, -self.max_lake_slope, 3.0)
        # 1) classic open water: positive NDWI/MNDWI, NIR-dark
        p_open = _sig(w0, thr, 0.05) * _sig(m0, 0.1, 0.05) * _sig(-nir0, -0.20, 0.03)
        # 2) deep clear lakes are almost black after atmospheric correction, so band ratios are
        #    noise there. Use absolute darkness in NIR+SWIR on flat, sunlit terrain.
        p_dark = _sig(-nir0, -0.04, 0.008) * _sig(-swir0, -0.05, 0.01) * _sig(-vis0, -0.10, 0.02) * (~shadow)
        p_water = np.maximum(p_open, p_dark) * p_flat
        water = (((w0 > thr) & (m0 > 0.1) & (nir0 < 0.2))
                 | ((nir0 < 0.04) & (swir0 < 0.05) & (vis0 < 0.10) & ~shadow)
                 | ((scene.scl == ix.SCL_WATER) & (nir0 < 0.08) & (m0 > 0.0))) & flat

        # frozen / partly frozen lakes: bright, flat, NDSI-high but with a water-like core are left
        # to the model; the rules only take open water
        p_snow = _sig(m0, 0.4, 0.05) * _sig(nir0, 0.15, 0.03)
        snow = (m0 > 0.4) & (nir0 > 0.15) & ~water

        classes = np.full(w.shape, OTHER, dtype=np.uint8)
        classes[snow] = SNOW_ICE
        if terrain.glacier_prior is not None:
            classes[terrain.glacier_prior & ~snow & ~water] = DEBRIS_ICE
        classes[water] = WATER

        # clean speckle: remove isolated water pixels (<3 px) and fill 1-px holes in lakes
        water_clean = ndimage.binary_opening(water, structure=np.ones((2, 2)))
        water_clean = ndimage.binary_fill_holes(water_clean) & (water | ndimage.binary_dilation(water))
        classes[water & ~water_clean] = OTHER
        classes[water_clean] = WATER

        # SCL cloud shadow is only invalid where we don't see a confident water signature
        invalid = missing | cloud | (scl_shadow & ~water_clean)
        classes[invalid] = INVALID
        p_other = np.clip(1 - p_water - p_snow, 0, 1)
        return Segmentation(classes, {WATER: p_water, SNOW_ICE: p_snow, OTHER: p_other}, self.name)


def get_segmenter(settings: Settings | None = None) -> Segmenter:
    """Best available segmenter: Prithvi (if a fine-tuned checkpoint is configured) → spectral.

    If ``sam_model`` is configured the result is wrapped with SAM lake-outline refinement.
    Any failure while loading models falls back to the spectral rules with a loud warning:
    the warning system must keep running.
    """
    settings = settings or get_settings()
    seg: Segmenter = SpectralSegmenter()
    want_prithvi = settings.segmenter == "prithvi" or (
        settings.segmenter == "auto" and settings.prithvi_checkpoint and settings.prithvi_checkpoint.exists())
    if want_prithvi:
        try:
            from himsat.ml.prithvi import PrithviSegmenter

            seg = PrithviSegmenter.from_settings(settings, fallback=seg)
        except Exception as e:  # pragma: no cover - depends on optional deps / files
            if settings.segmenter == "prithvi":
                raise
            log.error("Prithvi segmenter unavailable (%s); using spectral rules", e)
    if settings.sam_model:
        try:
            from himsat.ml.sam import SamRefiner

            seg = SamRefiner(seg, settings.sam_model, device=settings.resolved_device())
        except Exception as e:  # pragma: no cover
            log.error("SAM refiner unavailable (%s); continuing without it", e)
    log.info("segmenter: %s", seg.name)
    return seg
