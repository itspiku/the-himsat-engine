"""Fine-tune Prithvi-EO-2.0 on a HimSat weak-label dataset (see ``himsat.ml.dataset``)."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from himsat.detect.segmentation import CLASS_NAMES, N_CLASSES
from himsat.ml.dataset import IGNORE
from himsat.ml.prithvi import BANDS, backbone_config, build_model

log = logging.getLogger(__name__)


class PatchDataset:
    def __init__(self, files: list[Path], mean: np.ndarray, std: np.ndarray, augment: bool):
        self.files, self.mean, self.std, self.augment = files, mean, std, augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        import torch

        with np.load(self.files[i]) as z:
            x = z["x"].astype("float32")
            y = z["y"].astype("int64")
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        if self.augment:
            k = np.random.randint(4)
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            # mild radiometric jitter (illumination / atmosphere differences between scenes)
            x = x * np.float32(np.random.uniform(0.95, 1.05)) + np.float32(np.random.normal(0, 0.03))
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def confusion(pred: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    m = y != IGNORE
    return np.bincount(k * y[m] + pred[m], minlength=k * k).reshape(k, k)


def iou_from_confusion(cm: np.ndarray) -> np.ndarray:
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, np.nan)


def train(data: Path, out: Path, *, backbone: str = "ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL", epochs: int = 8,
          batch: int = 8, lr: float = 3e-4, unfreeze_blocks: int = 4, device: str = "cuda", workers: int = 2,
          progress=print) -> dict:
    import torch
    from torch.utils.data import DataLoader

    cfg = backbone_config(backbone)
    idx = [cfg["bands"].index(b) for b in ("B02", "B03", "B04", "B05", "B06", "B07")]  # HLS names, same order
    mean = np.asarray(cfg["mean"], "float32")[idx]
    std = np.asarray(cfg["std"], "float32")[idx]
    tr_files = sorted((data / "train").glob("*.npz"))
    va_files = sorted((data / "val").glob("*.npz"))
    if not tr_files or not va_files:
        raise RuntimeError(f"empty dataset in {data}")
    tr = DataLoader(PatchDataset(tr_files, mean, std, True), batch_size=batch, shuffle=True, num_workers=workers,
                    drop_last=True, persistent_workers=workers > 0)
    va = DataLoader(PatchDataset(va_files, mean, std, False), batch_size=batch, num_workers=workers)

    meta_ds = json.loads((data / "dataset.json").read_text())
    counts = np.array([meta_ds["class_pixels"].get(str(k), 0) for k in range(N_CLASSES)], "float64") + 1
    weights = torch.tensor((counts.sum() / counts) ** 0.5, dtype=torch.float32)
    weights = weights / weights.mean()

    model, _ = build_model(backbone, N_CLASSES, pretrained=True)
    model.freeze_encoder(unfreeze_blocks)
    model.to(device)
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": model.decoder.parameters(), "lr": lr},
                             {"params": enc_params, "lr": lr * 0.1}], weight_decay=0.01)
    steps = epochs * len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=IGNORE)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    best, history = -1.0, []
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.detach().item()
            n += 1
        cm = evaluate(model, va, device)
        iou = iou_from_confusion(cm)
        miou = float(np.nanmean(iou))
        rec = {"epoch": ep + 1, "loss": tot / max(n, 1), "val_miou": miou,
               "val_iou": {CLASS_NAMES[k]: (None if np.isnan(v) else round(float(v), 4)) for k, v in enumerate(iou)},
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        progress(json.dumps(rec))
        if miou > best:
            best = miou
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "meta": {
                "backbone": backbone, "bands": BANDS, "mean": mean.tolist(), "std": std.tolist(),
                "n_classes": N_CLASSES, "classes": [CLASS_NAMES[k] for k in range(N_CLASSES)],
                "val_miou": miou, "val_iou": rec["val_iou"], "epoch": ep + 1,
                "version": out.stem, "dataset": meta_ds, "confusion": cm.tolist()}}, out)
    (out.with_suffix(".json")).write_text(json.dumps({"best_val_miou": best, "history": history}, indent=1))
    return {"best_val_miou": best, "history": history}


def evaluate(model, loader, device: str) -> np.ndarray:
    import torch

    model.eval()
    cm = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    with torch.inference_mode():
        for x, y in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                pred = model(x.to(device)).argmax(1).cpu().numpy()
            cm += confusion(pred.ravel(), y.numpy().ravel(), N_CLASSES)
    return cm
