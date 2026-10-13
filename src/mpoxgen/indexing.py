"""Build a clean image index from a dataset folder.

The index has one row per *unique file content* (MD5), so an image that the
dataset ships in several folds is counted once. Official fold membership, if
the folder layout has it (e.g. MSLD v2.0 ``fold1/Train/Mpox/...``), is kept in
a separate table.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pandas as pd
from PIL import Image

from .labels import canonical_class

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = {"train": "train", "training": "train", "valid": "val", "val": "val",
               "validation": "val", "test": "test", "testing": "test"}
FOLD_RE = re.compile(r"fold[\s_\-]?(\d+)", re.IGNORECASE)


def md5_of(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _class_from_parts(parts: tuple[str, ...]) -> str | None:
    # Nearest class-like folder above the file wins.
    for p in reversed(parts[:-1]):
        c = canonical_class(p)
        if c is not None:
            return c
    return None


def _fold_split(parts: tuple[str, ...]) -> tuple[int, str]:
    fold, split = -1, ""
    for p in parts[:-1]:
        m = FOLD_RE.search(p)
        if m:
            fold = int(m.group(1))
        s = SPLIT_NAMES.get(p.lower())
        if s:
            split = s
    return fold, split


def patient_hint(stem: str) -> str:
    """Best-effort patient key from a file stem: drop a trailing image counter.

    e.g. ``MKP_017_02`` -> ``MKP_017``. Only used for sanity checks and for the
    early-stopping split in cross-dataset training; official folds are used for
    the patient-level protocol whenever a dataset provides them.
    """
    m = re.match(r"^(.*?)[_\-\s]?\d{1,3}$", stem)
    return m.group(1) if m and m.group(1) else stem


def scan(root: str | Path, dataset: str, exclude_keywords=("aug",)) -> pd.DataFrame:
    """Walk ``root`` and return one row per image file (duplicates kept)."""
    root = Path(root)
    rows = []
    for dirpath, _, files in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        low = str(rel_dir).lower()
        if any(k in low for k in exclude_keywords):
            continue
        for fn in files:
            if Path(fn).suffix.lower() not in IMG_EXT:
                continue
            rel = rel_dir / fn
            parts = rel.parts
            label = _class_from_parts(parts)
            if label is None:
                continue
            fold, split = _fold_split(parts)
            full = root / rel
            rows.append({
                "dataset": dataset,
                "path": str(full),
                "rel_path": str(rel),
                "label": label,
                "fold": fold,
                "split": split,
                "md5": md5_of(full),
                "patient_hint": patient_hint(Path(fn).stem),
            })
    if not rows:
        raise RuntimeError(f"No labelled images found under {root}. Check the path.")
    return pd.DataFrame(rows)


def build(root: str | Path, dataset: str, exclude_keywords=("aug",)):
    """Return (images, official_folds, info).

    images: one row per unique MD5 with columns
        image_id, dataset, path, label, md5, patient_hint, width, height
    official_folds: rows (image_id, fold, split) for files found inside foldN/ dirs.
    """
    raw = scan(root, dataset, exclude_keywords)

    # A single MD5 must not carry two different labels.
    conflict = raw.groupby("md5")["label"].nunique()
    bad = conflict[conflict > 1]
    if len(bad):
        print(f"[warn] {dataset}: {len(bad)} identical files carry different labels; "
              "they are dropped and listed in label_conflicts.csv")
    n_files = len(raw)
    conflicts = raw[raw["md5"].isin(bad.index)].copy()
    raw = raw[~raw["md5"].isin(bad.index)].copy()

    raw["image_id"] = dataset + ":" + raw["md5"].str[:12]
    first = raw.sort_values("rel_path").drop_duplicates("md5", keep="first").copy()

    sizes = []
    for p in first["path"]:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
        except Exception:  # unreadable image
            sizes.append((0, 0))
    first["width"] = [s[0] for s in sizes]
    first["height"] = [s[1] for s in sizes]
    unreadable = (first["width"] == 0).sum()
    if unreadable:
        print(f"[warn] {dataset}: {unreadable} unreadable images dropped")
        first = first[first["width"] > 0]

    images = first[["image_id", "dataset", "path", "label", "md5", "patient_hint",
                    "width", "height"]].reset_index(drop=True)

    folds = raw[(raw["fold"] >= 0) & (raw["split"] != "")]
    folds = folds[folds["image_id"].isin(images["image_id"])]
    official = folds[["image_id", "fold", "split"]].drop_duplicates().reset_index(drop=True)

    info = {"n_files": int(n_files), "label_conflicts": conflicts}
    return images, official, info


def describe(images: pd.DataFrame, official: pd.DataFrame) -> str:
    lines = []
    for ds, g in images.groupby("dataset"):
        lines.append(f"{ds}: {len(g)} unique images")
        for lab, n in g["label"].value_counts().items():
            lines.append(f"    {lab:<11} {n}")
        o = official[official["image_id"].isin(g["image_id"])]
        if len(o):
            for f, fo in o.groupby("fold"):
                c = fo["split"].value_counts().to_dict()
                lines.append(f"    fold {f}: train={c.get('train', 0)} val={c.get('val', 0)} "
                             f"test={c.get('test', 0)}")
    return "\n".join(lines)
