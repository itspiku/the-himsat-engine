"""NASA/IBM Prithvi-EO-2.0 foundation model, fine-tuned for Himalayan surface segmentation.

The encoder is the open-weight Prithvi-EO-2.0 ViT (HLS-pretrained, 6 bands: Blue, Green, Red,
NIR-narrow, SWIR1, SWIR2). Its official model code and weights are fetched from the Hugging Face
Hub. A light multi-level decoder (four transformer depths → progressive upsampling) predicts the
HimSat classes. Sentinel-2 L2A reflectance maps directly onto HLS S30 bands
(B02, B03, B04, B8A, B11, B12), the same sensor the model was pretrained on.

A checkpoint stores the decoder (and any fine-tuned encoder blocks) plus metadata: backbone
id, band order, normalisation, classes and validation metrics.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from himsat.detect.segmentation import (
    INVALID,
    N_CLASSES,
    OTHER,
    WATER,
    Segmentation,
    SpectralSegmenter,
    TerrainContext,
)
from himsat.ingest.loader import S2Scene

log = logging.getLogger(__name__)

BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")
PATCH = 224
FEATURE_LAYERS = (3, 5, 7, 11)  # 0-based transformer blocks fed to the decoder


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for the Prithvi segmenter: pip install 'himsat[ml]'") from e


def load_backbone_module(repo_id: str):
    """Import ``prithvi_mae.py`` from the model repository (official implementation)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, "prithvi_mae.py")
    name = "prithvi_mae_" + repo_id.replace("/", "_").replace("-", "_").replace(".", "_")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def backbone_config(repo_id: str) -> dict:
    import json

    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo_id, "config.json"), encoding="utf-8") as f:
        return json.load(f)["pretrained_cfg"]


def build_model(repo_id: str, n_classes: int = N_CLASSES, pretrained: bool = True):
    import torch

    cfg = backbone_config(repo_id)
    mod = load_backbone_module(repo_id)
    encoder = mod.PrithviViT(
        img_size=PATCH, patch_size=cfg["patch_size"], num_frames=1, in_chans=cfg["in_chans"],
        embed_dim=cfg["embed_dim"], depth=cfg["depth"], num_heads=cfg["num_heads"], mlp_ratio=cfg["mlp_ratio"],
        coords_encoding=[], coords_scale_learn=cfg.get("coords_scale_learn", False),
    )
    if pretrained:
        from huggingface_hub import hf_hub_download, list_repo_files

        wfile = next(f for f in list_repo_files(repo_id) if f.endswith(".pt"))
        state = torch.load(hf_hub_download(repo_id, wfile), map_location="cpu", weights_only=True)
        enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        # positional embeddings are rebuilt for single-frame input; temporal/location encoders unused
        enc_state = {k: v for k, v in enc_state.items() if k != "pos_embed" and "embed_enc" not in k}
        missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
        log.info("Prithvi weights loaded (%d missing, %d unexpected keys)", len(missing), len(unexpected))
    return SegModel(encoder, cfg["embed_dim"], n_classes), cfg


def _conv_block(cin: int, cout: int):
    from torch import nn

    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _make_decoder(embed_dim: int, n_classes: int):
    from torch import nn

    class Decoder(nn.Module):
        """Fuse four encoder depths at 1/16 resolution, then upsample ×16 with skip-free conv stages."""

        def __init__(self):
            super().__init__()
            self.proj = nn.ModuleList([nn.Conv2d(embed_dim, 128, 1) for _ in FEATURE_LAYERS])
            self.fuse = _conv_block(128 * len(FEATURE_LAYERS), 256)
            chans = [256, 128, 96, 64, 48]
            self.ups = nn.ModuleList([
                nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), _conv_block(a, b))
                for a, b in zip(chans[:-1], chans[1:], strict=True)])
            self.head = nn.Conv2d(chans[-1], n_classes, 1)

        def forward(self, feats):
            import torch

            x = torch.cat([p(f) for p, f in zip(self.proj, feats, strict=True)], dim=1)
            x = self.fuse(x)
            for up in self.ups:
                x = up(x)
            return self.head(x)

    return Decoder()


def SegModel(encoder, embed_dim: int, n_classes: int):  # noqa: N802 - factory mimicking a class
    from torch import nn

    class _SegModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.decoder = _make_decoder(embed_dim, n_classes)

        def forward(self, x):
            # x: (B, C, H, W) normalised reflectance; Prithvi wants (B, C, T, H, W)
            feats = self.encoder.forward_features(x.unsqueeze(2))
            feats = self.encoder.prepare_features_for_image_model([feats[i] for i in FEATURE_LAYERS])
            return self.decoder(feats)

        def freeze_encoder(self, trainable_last_blocks: int = 0) -> None:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if trainable_last_blocks:
                for blk in self.encoder.blocks[-trainable_last_blocks:]:
                    for p in blk.parameters():
                        p.requires_grad = True
                for p in self.encoder.norm.parameters():
                    p.requires_grad = True

    return _SegModel()


@dataclass
class Normaliser:
    mean: np.ndarray  # in HLS reflectance ×10000 units
    std: np.ndarray

    def __call__(self, stack_refl: np.ndarray) -> np.ndarray:
        """(C, H, W) reflectance (0..1) → normalised, NaN → 0."""
        x = (np.nan_to_num(stack_refl, nan=0.0) * 10000.0 - self.mean[:, None, None]) / self.std[:, None, None]
        return x.astype("float32")


def scene_stack(scene: S2Scene) -> np.ndarray:
    return np.stack([scene.bands[b] for b in BANDS])


def sliding_windows(h: int, w: int, size: int = PATCH, stride: int = 160) -> list[tuple[int, int]]:
    def starts(n: int) -> list[int]:
        if n <= size:
            return [0]
        s = list(range(0, n - size, stride))
        return s + [n - size]

    return [(r, c) for r in starts(h) for c in starts(w)]


class PrithviSegmenter:
    """Inference wrapper. Cloud/no-data masking still comes from the scene classification."""

    def __init__(self, model, normaliser: Normaliser, device: str, meta: dict, fallback=None):
        import torch

        self.model = model.to(device).eval()
        self.norm = normaliser
        self.device = device
        self.meta = meta
        self.fallback = fallback or SpectralSegmenter()
        self.name = f"prithvi:{meta.get('version', 'ft')}"
        self._amp = device.startswith("cuda")
        self._torch = torch

    @classmethod
    def from_checkpoint(cls, path: Path, device: str = "cpu", fallback=None) -> PrithviSegmenter:
        _require_torch()
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=False)
        meta = ck["meta"]
        model, _ = build_model(meta["backbone"], n_classes=meta.get("n_classes", N_CLASSES), pretrained=False)
        model.load_state_dict(ck["state_dict"])
        norm = Normaliser(np.asarray(meta["mean"], "float32"), np.asarray(meta["std"], "float32"))
        log.info("Prithvi segmenter %s (val mIoU %.3f) on %s", path.name, meta.get("val_miou", float("nan")), device)
        return cls(model, norm, device, meta, fallback)

    @classmethod
    def from_settings(cls, settings, fallback=None) -> PrithviSegmenter:
        if not settings.prithvi_checkpoint or not Path(settings.prithvi_checkpoint).exists():
            raise FileNotFoundError(f"Prithvi checkpoint not found: {settings.prithvi_checkpoint}")
        return cls.from_checkpoint(Path(settings.prithvi_checkpoint), settings.resolved_device(), fallback)

    def predict_proba(self, stack: np.ndarray, batch: int = 8) -> np.ndarray:
        """(C, H, W) reflectance → (K, H, W) class probabilities (overlapping windows, cosine-blended)."""
        torch = self._torch
        x = self.norm(stack)
        _, h, w = x.shape
        ph, pw = max(h, PATCH), max(w, PATCH)
        if (ph, pw) != (h, w):
            x = np.pad(x, ((0, 0), (0, ph - h), (0, pw - w)))
        k = self.meta.get("n_classes", N_CLASSES)
        acc = np.zeros((k, ph, pw), "float32")
        wsum = np.zeros((ph, pw), "float32")
        win = np.outer(np.hanning(PATCH + 2)[1:-1], np.hanning(PATCH + 2)[1:-1]).astype("float32") + 1e-3
        coords = sliding_windows(ph, pw)
        with torch.inference_mode():
            for i in range(0, len(coords), batch):
                cs = coords[i:i + batch]
                xb = torch.from_numpy(np.stack([x[:, r:r + PATCH, c:c + PATCH] for r, c in cs])).to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self._amp):
                    logits = self.model(xb)
                prob = torch.softmax(logits.float(), dim=1).cpu().numpy()
                for (r, c), p in zip(cs, prob, strict=True):
                    acc[:, r:r + PATCH, c:c + PATCH] += p * win
                    wsum[r:r + PATCH, c:c + PATCH] += win
        return (acc / wsum)[:, :h, :w]

    def segment(self, scene: S2Scene, terrain: TerrainContext) -> Segmentation:
        if any(b not in scene.bands for b in BANDS):
            log.warning("scene lacks Prithvi bands; using spectral rules")
            return self.fallback.segment(scene, terrain)
        rules = self.fallback.segment(scene, terrain)  # provides the invalid (cloud / no data) mask
        prob = self.predict_proba(scene_stack(scene))
        classes = prob.argmax(0).astype(np.uint8)
        classes[(classes == WATER) & (terrain.slope > 25)] = OTHER  # lakes are flat: physics veto
        classes[rules.classes == INVALID] = INVALID
        return Segmentation(classes, {c: prob[c] for c in range(prob.shape[0])}, self.name)
