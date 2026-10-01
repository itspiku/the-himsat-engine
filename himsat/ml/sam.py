"""Segment Anything (SAM) refinement of lake outlines.

The base segmenter (Prithvi or spectral rules) finds lakes. SAM then sharpens each outline, which
matters for area-change measurement where a 1-pixel ring is ~10 % of a small lake. Each detected
lake is cropped from a false-colour composite (NIR-red-green: water is near black, ice white,
vegetation red) and SAM is prompted with the lake's bounding box. The refined mask is accepted only
if it agrees with the coarse one (IoU ≥ 0.6) and stays on flat terrain. Otherwise the coarse
outline is kept, so SAM can sharpen a lake but never invent one.
"""

from __future__ import annotations

import logging

import numpy as np
from skimage import measure

from himsat.detect.segmentation import OTHER, WATER, Segmentation, TerrainContext
from himsat.ingest.loader import S2Scene

log = logging.getLogger(__name__)


def false_colour(scene: S2Scene, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
    chans = []
    for b in ("B08", "B04", "B03"):
        a = np.nan_to_num(scene.bands[b][r0:r1, c0:c1], nan=0.0)
        lo, hi = np.percentile(a, 2), np.percentile(a, 98)
        chans.append(np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1))
    return (np.dstack(chans) * 255).astype(np.uint8)


class SamRefiner:
    def __init__(self, base, model_id: str = "facebook/sam-vit-base", device: str = "cpu",
                 min_px: int = 20, max_px: int = 200_000, min_iou: float = 0.6):
        import torch
        from transformers import SamModel, SamProcessor

        self.base = base
        self.device = device
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(device).eval()
        self.min_px, self.max_px, self.min_iou = min_px, max_px, min_iou
        self.name = f"{base.name}+sam"
        self._torch = torch
        self.stats = {"refined": 0, "kept": 0}

    def _refine_one(self, img: np.ndarray, box: list[int]) -> np.ndarray:
        torch = self._torch
        inputs = self.processor(images=img, input_boxes=[[box]], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs, multimask_output=False)
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        return masks[0][0, 0].numpy().astype(bool)

    def segment(self, scene: S2Scene, terrain: TerrainContext) -> Segmentation:
        seg = self.base.segment(scene, terrain)
        if any(b not in scene.bands for b in ("B08", "B04", "B03")):
            return seg
        water = seg.classes == WATER
        lab = measure.label(water, connectivity=2)
        classes = seg.classes.copy()
        h, w = water.shape
        for rp in measure.regionprops(lab):
            if not (self.min_px <= rp.area <= self.max_px):
                continue
            r0, c0, r1, c1 = rp.bbox
            pad = max(16, (r1 - r0) // 2, (c1 - c0) // 2)
            wr0, wc0, wr1, wc1 = max(0, r0 - pad), max(0, c0 - pad), min(h, r1 + pad), min(w, c1 + pad)
            img = false_colour(scene, wr0, wr1, wc0, wc1)
            coarse = lab[wr0:wr1, wc0:wc1] == rp.label
            try:
                refined = self._refine_one(img, [c0 - wc0 - 1, r0 - wr0 - 1, c1 - wc0 + 1, r1 - wr0 + 1])
            except Exception as e:  # pragma: no cover - model/runtime issues must not stop monitoring
                log.warning("SAM failed on a lake: %s", e)
                continue
            refined &= terrain.slope[wr0:wr1, wc0:wc1] < 20
            inter = (refined & coarse).sum()
            union = (refined | coarse).sum()
            if union == 0 or inter / union < self.min_iou:
                self.stats["kept"] += 1
                continue
            sub = classes[wr0:wr1, wc0:wc1]
            sub[coarse & ~refined] = OTHER
            sub[refined & (sub != 255)] = WATER
            self.stats["refined"] += 1
        return Segmentation(classes, seg.prob, self.name)
