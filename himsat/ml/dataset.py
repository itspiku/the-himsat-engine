"""Weakly supervised training data for the Prithvi segmenter.

There is no pixel-labelled Himalayan glacial-lake dataset under an open licence that matches
Sentinel-2. Labels are built from three sources that are each reliable *where they are confident*:

* physics rules (``SpectralSegmenter``): only pixels with probability ≥ 0.85 for water,
  snow/ice or other are kept
* glacier inventory outlines (OpenStreetMap natural=glacier, largely traced from GLIMS/RGI):
  non-snow, non-water pixels well inside an outline become *debris-covered ice*, a class the
  rules cannot see
* everything else, including class boundaries, clouds and shadows, gets the ignore label (255)

The model learns spatial context from the confident pixels and generalises to the uncertain
ones: turbid or partly frozen lakes, debris-covered glacier tongues, shadow edges. Patches are
split into train/validation **by tile** (spatial hold-out), so validation never shares pixels with
training.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import ndimage

from himsat.config import get_aoi
from himsat.detect.segmentation import DEBRIS_ICE, INVALID, OTHER, SNOW_ICE, WATER, SpectralSegmenter
from himsat.ingest.loader import load_s2
from himsat.ingest.stac import group_acquisitions
from himsat.ml.prithvi import BANDS, PATCH
from himsat.pipeline.context import AOIContext
from himsat.pipeline.s2 import terrain_for

log = logging.getLogger(__name__)
IGNORE = 255


def weak_labels(rules, glacier: np.ndarray, p_min: float = 0.85) -> np.ndarray:
    y = np.full(rules.classes.shape, IGNORE, np.uint8)
    pw, ps, po = (rules.prob.get(k) for k in (WATER, SNOW_ICE, OTHER))
    c = rules.classes
    y[(c == WATER) & (pw >= p_min)] = WATER
    y[(c == SNOW_ICE) & (ps >= p_min)] = SNOW_ICE
    inner_glacier = ndimage.binary_erosion(glacier, iterations=3)
    near_glacier = ndimage.binary_dilation(glacier, iterations=3)
    y[(c == OTHER) & (po >= p_min) & ~near_glacier] = OTHER
    y[((c == OTHER) | (c == DEBRIS_ICE)) & inner_glacier] = DEBRIS_ICE
    # boundaries between different labels are ambiguous at 10-20 m: ignore a 1-px seam
    for k in (WATER, SNOW_ICE, OTHER, DEBRIS_ICE):
        m = y == k
        seam = ndimage.binary_dilation(m) & ~m & (y != IGNORE) & (y != k)
        y[seam] = IGNORE
    y[c == INVALID] = IGNORE
    return y


def _split(aoi: str, tile: tuple[int, int], r: int, c: int, val_fraction: float, block: int = 4 * PATCH) -> str:
    """Spatial hold-out by ~9 km blocks; all dates of a block land in the same split."""
    h = int(hashlib.sha1(f"{aoi}:{tile}:{r // block}:{c // block}".encode()).hexdigest(), 16) % 1000
    return "val" if h < val_fraction * 1000 else "train"


def build_dataset(aoi_ids: list[str], start: datetime, end: datetime, out: Path, *, scenes_per_aoi: int = 3,
                  max_cloud: float = 10.0, val_fraction: float = 0.2, min_labelled: float = 0.3,
                  progress=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rules = SpectralSegmenter()
    stats = {"train": 0, "val": 0, "class_pixels": {k: 0 for k in (OTHER, WATER, SNOW_ICE, DEBRIS_ICE)}}
    for aoi_id in aoi_ids:
        cfg = get_aoi(aoi_id)
        ctx = AOIContext(cfg)
        acqs = sorted(group_acquisitions(ctx.catalog.search_s2(cfg.bbox, start, end, max_cloud=max_cloud)),
                      key=lambda a: a.cloud_cover or 0)[:scenes_per_aoi]
        for acq in acqs:
            for tile in ctx.tiles:
                scene = load_s2(acq, tile.grid, bands=tuple(sorted(set(BANDS) | {"B08"})), catalog=ctx.catalog,
                                min_valid_fraction=0.3)
                if scene is None:
                    continue
                st = ctx.tile_static(tile)
                terrain = terrain_for(ctx, tile, acq)
                y = weak_labels(rules.segment(scene, terrain), st.glacier)
                x = np.stack([scene.bands[b] for b in BANDS])
                h, w = y.shape
                n = 0
                for r in range(0, h - PATCH + 1, PATCH):
                    for c in range(0, w - PATCH + 1, PATCH):
                        yp = y[r:r + PATCH, c:c + PATCH]
                        if (yp != IGNORE).mean() < min_labelled:
                            continue
                        xp = x[:, r:r + PATCH, c:c + PATCH]
                        split = _split(aoi_id, tile.index, r, c, val_fraction)
                        if not np.isfinite(xp).all():
                            continue
                        name = f"{aoi_id}_{acq.key}_{tile.index[0]}_{tile.index[1]}_{r}_{c}.npz"
                        np.savez_compressed(out / split / name if (out / split).exists() else _mk(out / split) / name,
                                            x=np.clip(xp * 10000, 0, 65535).astype(np.uint16), y=yp)
                        for k in stats["class_pixels"]:
                            stats["class_pixels"][k] += int((yp == k).sum())
                        stats[split] += 1
                        n += 1
                progress(f"{aoi_id} {acq.key} tile {tile.index}: {n} patches")
    stats["class_pixels"] = {str(k): v for k, v in stats["class_pixels"].items()}
    (out / "dataset.json").write_text(json.dumps({**stats, "bands": BANDS, "patch": PATCH, "aois": aoi_ids,
                                                  "period": [start.isoformat(), end.isoformat()]}, indent=1))
    return stats


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
