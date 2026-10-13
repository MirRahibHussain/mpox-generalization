"""Training and evaluation of one (split, model, seed) run. Requires torch + timm."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from . import metrics
from .labels import TASKS, positive_index, task_target

# Models compared in the paper (timm names). Learning rates follow common
# fine-tuning practice: lower for transformers.
MODELS = {
    "resnet50": 1e-4,
    "densenet121": 1e-4,
    "efficientnet_b0": 3e-4,
    "mobilenetv3_large_100": 3e-4,
    "convnext_tiny": 1e-4,
    "vit_small_patch16_224": 5e-5,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ImageTable(Dataset):
    def __init__(self, paths, targets, tf):
        self.paths, self.targets, self.tf = list(paths), list(targets), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            x = self.tf(im)
        return x, self.targets[i]


def transforms(size: int = 224):
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(20),
        # Mild brightness/contrast only; no hue/saturation change, so the
        # skin-tone analysis is not confounded by colour augmentation.
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test = T.Compose([
        T.Resize(int(size * 1.143)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, test


def _rows(images: pd.DataFrame, ids, task):
    df = images.set_index("image_id").loc[list(ids)]
    y = [task_target(l, task) for l in df["label"]]
    return df.index.to_list(), df["path"].to_list(), y


def _predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def run(spec: dict, images: pd.DataFrame, model_name: str, seed: int, out_dir: Path,
        epochs: int = 30, patience: int = 7, batch_size: int = 32, size: int = 224,
        num_workers: int = 2, save_checkpoint: bool = False, pretrained: bool = True) -> dict:
    """Train on spec['train'], early-stop on spec['val'], predict spec['test'].

    Writes predictions.csv and metrics.json into out_dir and returns the metrics.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = spec["task"]
    n_cls = len(TASKS[task])

    tf_train, tf_test = transforms(size)
    _, p_tr, y_tr = _rows(images, spec["train"], task)
    _, p_va, y_va = _rows(images, spec["val"], task)
    id_te, p_te, y_te = _rows(images, spec["test"], task)

    g = torch.Generator()
    g.manual_seed(seed)
    dl_tr = DataLoader(ImageTable(p_tr, y_tr, tf_train), batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True, drop_last=len(p_tr) > batch_size,
                       generator=g)
    dl_va = DataLoader(ImageTable(p_va, y_va, tf_test), batch_size=batch_size,
                       num_workers=num_workers, pin_memory=True)
    dl_te = DataLoader(ImageTable(p_te, y_te, tf_test), batch_size=batch_size,
                       num_workers=num_workers, pin_memory=True)

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=n_cls).to(device)

    counts = np.bincount(y_tr, minlength=n_cls).astype(float)
    weights = counts.sum() / np.maximum(counts, 1) / n_cls
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    lr = MODELS.get(model_name, 1e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    pos = positive_index(task)
    best, best_epoch, best_state, history = -1.0, -1, None, []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()
        yv, pv = _predict(model, dl_va, device)
        val = metrics.compute(yv, pv, pos)["balanced_accuracy"]
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_bal_acc": val})
        if val > best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    yt, pt = _predict(model, dl_te, device)
    res = metrics.compute(yt, pt, pos)
    ci = metrics.bootstrap(yt, pt, pos, n=1000, seed=seed)

    pred = pd.DataFrame({"image_id": id_te, "y_true": yt, "y_pred": pt.argmax(1)})
    for c in range(n_cls):
        pred[f"p{c}"] = pt[:, c]
    pred.to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    record = {
        "protocol": spec["protocol"], "task": task, "rep": spec["rep"],
        "train_dataset": spec["train_dataset"], "test_dataset": spec["test_dataset"],
        "model": model_name, "seed": seed, "lr": lr,
        "n_train": len(p_tr), "n_val": len(p_va), "n_test": len(p_te),
        "best_epoch": best_epoch, "val_bal_acc": best,
        "minutes": round((time.time() - t0) / 60, 2),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        **res,
        **{f"{k}_ci_low": v[0] for k, v in ci.items()},
        **{f"{k}_ci_high": v[1] for k, v in ci.items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    if save_checkpoint:
        torch.save(best_state, out_dir / "model.pt")
    return record
