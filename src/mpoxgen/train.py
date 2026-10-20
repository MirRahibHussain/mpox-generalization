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
    # Fixed seeds. By default cuDNN autotuning is on for speed, so GPU results can
    # differ slightly between repeated runs; MPOXGEN_DETERMINISTIC=1 switches to
    # deterministic cuDNN kernels (used for the FACTOR study).
    import os
    det = os.environ.get("MPOXGEN_DETERMINISTIC", "0") == "1"
    torch.backends.cudnn.benchmark = not det
    torch.backends.cudnn.deterministic = det


class ImageTable(Dataset):
    def __init__(self, paths, targets, tf, groups=None):
        self.paths, self.targets, self.tf = list(paths), list(targets), tf
        self.groups = None if groups is None else [int(g) for g in groups]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            x = self.tf(im)
        if self.groups is None:
            return x, self.targets[i]
        return x, self.targets[i], self.groups[i]


def transforms(size: int = 224, aug: str = "base"):
    """aug: "base" (study default), "hsv" (adds hue/saturation jitter) or
    "tone" (adds the lesion-preserving tone shift of methods.ToneShift)."""
    from .methods import ToneShift
    pre = [ToneShift(p=0.5)] if aug == "tone" else []
    jitter = (T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05) if aug == "hsv"
              # Mild brightness/contrast only; no hue/saturation change, so the
              # skin-tone analysis is not confounded by colour augmentation.
              else T.ColorJitter(brightness=0.2, contrast=0.2))
    train = T.Compose(pre + [
        T.RandomResizedCrop(size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(20),
        jitter,
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
        for batch in loader:
            x, y = batch[0], batch[1]
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
    keep = num_workers > 0  # keep worker processes alive across epochs
    dl_tr = DataLoader(ImageTable(p_tr, y_tr, tf_train), batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True, drop_last=len(p_tr) > batch_size,
                       generator=g, persistent_workers=keep)
    dl_va = DataLoader(ImageTable(p_va, y_va, tf_test), batch_size=batch_size,
                       num_workers=num_workers, pin_memory=True, persistent_workers=keep)
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


def _predict_head(model, loader, device, head):
    """Labels, (n, 2) [p_non_mpox, p_mpox] scores and head-specific extras."""
    from .factor import factor_scores, hod_scores
    model.eval()
    ys, p1, extra = [], [], {"pd": [], "pm": []}
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
            if head == "factor":
                pm_, pdis, pmd = factor_scores(logits)
                p1.append(pm_)
                extra["pd"].append(pdis)
                extra["pm"].append(pmd)
            elif head == "hod":
                p1.append(hod_scores(logits))
            else:
                p1.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
            ys.append(y.numpy())
    p1 = np.concatenate(p1)
    probs = np.stack([1 - p1, p1], 1)
    extra = {k: np.concatenate(v) for k, v in extra.items() if v}
    return np.concatenate(ys), probs, extra


def run_method(spec: dict, images: pd.DataFrame, model_name: str, seed: int, out_dir: Path,
               method: str, extra: pd.DataFrame | None = None, epochs: int = 30, patience: int = 7,
               batch_size: int = 32, size: int = 224, num_workers: int = 2,
               pretrained: bool = True) -> dict:
    """Like ``run`` but with one of the training methods in ``methods.CONFIGS``.

    ``images`` needs an ``ita_band`` column. ``extra`` holds the auxiliary pool
    for the method (look-alike, control or the whole atlas; label "other",
    plus a ``condition`` column); it is used for training only, never for
    validation or test. A static pool is added to every epoch; the atlas is
    sampled anew every epoch (factor.EpochSampler). Validation predictions of
    the best model are saved too, so decision thresholds can be chosen on
    validation data (scripts/11_operating_points.py).
    """
    from torch.utils.data import WeightedRandomSampler

    from .factor import EpochSampler, atlas_weights, factor_loss, hod_loss
    from .methods import GroupDRO, balanced_weights, config, group_ids, mixup

    cfg = config(method)
    head = cfg["head"]
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = spec["task"]
    if task != "binary" and (head != "binary" or cfg["extra"]):
        raise ValueError("auxiliary negatives and the factor/hod heads support the binary task only")
    n_cls = len(TASKS[task])

    table = images
    study_train = list(spec["train"])
    train_ids = list(study_train)
    pool_ids = []
    n_extra = 0
    if cfg["extra"]:
        if extra is None or extra.empty:
            raise ValueError(f"method {method} needs the {cfg['extra']} pool")
        cols = [c for c in images.columns if c in extra.columns] + (["condition"] if "condition" in extra else [])
        table = pd.concat([images, extra[list(dict.fromkeys(cols))]], ignore_index=True)
        pool_ids = list(extra["image_id"])
        train_ids += pool_ids
    tab = table.set_index("image_id")
    band = tab["ita_band"].fillna("unknown")
    dynamic = cfg["extra"] == "atlas"
    k_epoch = min(int(cfg["k"]), len(pool_ids)) if dynamic else len(pool_ids)
    n_extra = k_epoch

    tf_train, tf_test = transforms(size, cfg["aug"])
    _, p_tr, y_tr = _rows(table, train_ids, task)
    id_va, p_va, y_va = _rows(table, spec["val"], task)
    id_te, p_te, y_te = _rows(table, spec["test"], task)
    y_tr = np.asarray(y_tr)
    labels_tr = tab.loc[train_ids, "label"].astype(str).to_numpy()
    n_study = len(study_train)
    is_study = np.arange(len(train_ids)) < n_study

    # Per-sample auxiliary target: tone group (binary head), disease flag (factor)
    # or fine-grained class (hod).
    if head == "factor":
        aux = (labels_tr != "healthy").astype(int)
    elif head == "hod":
        cond = tab.loc[train_ids, "condition"].astype(str).to_numpy() if "condition" in tab else labels_tr
        fine_names = np.where(is_study, labels_tr, np.char.add("atlas:", cond.astype(str)))
        classes = ["mpox"] + sorted(set(fine_names) - {"mpox"})
        fine_index = {c: i for i, c in enumerate(classes)}
        aux = np.array([fine_index[c] for c in fine_names])
    else:
        aux = group_ids(y_tr, band.loc[train_ids].to_numpy())

    # Expected class composition of one epoch (study images + k pool images).
    frac_pool = (k_epoch / len(pool_ids)) if pool_ids else 0.0
    wts = np.where(is_study, 1.0, frac_pool)

    g = torch.Generator()
    g.manual_seed(seed)
    keep = num_workers > 0
    ds_tr = ImageTable(p_tr, y_tr, tf_train, aux)
    n_active = len(pool_ids)
    n_cond_active = int(tab.loc[pool_ids, "condition"].nunique()) if pool_ids and "condition" in tab else 0
    if dynamic:
        mpox_bands = band.loc[[i for i, yy in zip(study_train, y_tr[:n_study]) if yy == 1]].to_numpy()
        aw = atlas_weights(tab.loc[pool_ids, "condition"].to_numpy(), band.loc[pool_ids].to_numpy(),
                           mode=cfg["draw"] or "cond", mpox_bands=mpox_bands,
                           n_conditions=cfg["n_conditions"], seed=seed)
        sampler = EpochSampler(n_study, n_study, aw, k_epoch, seed=seed)
        n_active = int((aw > 0).sum())
        n_cond_active = int(pd.Series(tab.loc[pool_ids, "condition"].to_numpy()[aw > 0]).nunique())
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                           pin_memory=True, drop_last=len(sampler) > batch_size, persistent_workers=keep)
    elif cfg["sampler"] == "group":
        sampler = WeightedRandomSampler(torch.as_tensor(balanced_weights(aux), dtype=torch.double),
                                        num_samples=len(p_tr), replacement=True, generator=g)
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                           pin_memory=True, drop_last=len(p_tr) > batch_size, persistent_workers=keep)
    else:
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                           pin_memory=True, drop_last=len(p_tr) > batch_size, generator=g,
                           persistent_workers=keep)
    dl_va = DataLoader(ImageTable(p_va, y_va, tf_test), batch_size=batch_size,
                       num_workers=num_workers, pin_memory=True, persistent_workers=keep)
    dl_te = DataLoader(ImageTable(p_te, y_te, tf_test), batch_size=batch_size,
                       num_workers=num_workers, pin_memory=True)

    out_dim = len(classes) if head == "hod" else (2 if head == "factor" else n_cls)
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=out_dim).to(device)
    counts = np.array([wts[y_tr == c].sum() for c in range(n_cls)], dtype=float)
    weights = counts.sum() / np.maximum(counts, 1e-9) / n_cls
    criterion, dro = None, None
    if head == "factor":
        d_pos, d_neg = wts[aux == 1].sum(), wts[aux == 0].sum()
        m_mask = aux == 1
        m_pos, m_neg = wts[m_mask & (y_tr == 1)].sum(), wts[m_mask & (y_tr == 0)].sum()
        pw_d = torch.tensor(d_neg / max(d_pos, 1e-9), device=device)
        pw_m = torch.tensor(m_neg / max(m_pos, 1e-9), device=device)
    elif head == "hod":
        w_coarse = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        # Class weights only when nothing else rebalances the data.
        use_w = cfg["sampler"] is None and cfg["loss"] == "ce"
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device)
                                        if use_w else None)
        dro = GroupDRO(n_cls * 4, device=device) if cfg["loss"] == "gdro" else None
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
        for x, y, gb in dl_tr:
            x, y, gb = (t.to(device, non_blocking=True) for t in (x, y, gb))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                if head == "factor":
                    loss = factor_loss(model(x), y, gb, pw_d, pw_m)
                elif head == "hod":
                    loss = hod_loss(model(x), gb, y, w_coarse)
                elif cfg["mixup"] > 0:
                    xm, ya, yb, lam = mixup(x, y, cfg["mixup"])
                    out = model(xm)
                    loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
                else:
                    out = model(x)
                    loss = dro(out.float(), y, gb) if dro is not None else criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
            n += len(y)
        sched.step()
        yv, pv, _ = _predict_head(model, dl_va, device, head)
        val = metrics.compute(yv, pv, pos)["balanced_accuracy"]
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_bal_acc": val})
        if val > best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    yt, pt, ext = _predict_head(model, dl_te, device, head)
    res = metrics.compute(yt, pt, pos)
    ci = metrics.bootstrap(yt, pt, pos, n=1000, seed=seed)
    pred = pd.DataFrame({"image_id": id_te, "y_true": yt, "y_pred": pt.argmax(1)})
    for c in range(pt.shape[1]):
        pred[f"p{c}"] = pt[:, c]
    for k_, v_ in ext.items():
        pred[k_] = v_
    pred.to_csv(out_dir / "predictions.csv", index=False)
    yv, pv, extv = _predict_head(model, dl_va, device, head)
    vpred = pd.DataFrame({"image_id": id_va, "y_true": yv, "y_pred": pv.argmax(1)})
    for c in range(pv.shape[1]):
        vpred[f"p{c}"] = pv[:, c]
    for k_, v_ in extv.items():
        vpred[k_] = v_
    vpred.to_csv(out_dir / "val_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    record = {
        "protocol": spec["protocol"], "task": task, "rep": spec["rep"],
        "train_dataset": spec["train_dataset"], "test_dataset": spec["test_dataset"],
        "model": model_name, "method": method, "seed": seed, "lr": lr, "n_extra": n_extra,
        "n_pool": len(pool_ids), "n_pool_active": n_active, "n_conditions_active": n_cond_active, "head": head,
        "n_train": n_study + n_extra, "n_val": len(p_va), "n_test": len(p_te),
        "best_epoch": best_epoch, "val_bal_acc": best,
        "minutes": round((time.time() - t0) / 60, 2),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        **res,
        **{f"{k}_ci_low": v[0] for k, v in ci.items()},
        **{f"{k}_ci_high": v[1] for k, v in ci.items()},
    }
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    return record
