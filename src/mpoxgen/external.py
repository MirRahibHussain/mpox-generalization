"""Auxiliary "look-alike" negatives from Fitzpatrick17k (Groh et al., CVPR-W 2021).

Fitzpatrick17k has 16,577 clinical images of 114 skin conditions, none of which
is mpox, chickenpox, measles, cowpox or hand, foot and mouth disease. We take
conditions whose lesions are papules, pustules, vesicles, bullae, crusts or
erosions, which can resemble poxvirus lesions, and use them only as extra
non-mpox TRAINING images. They never enter a validation or test set.

Images are named ``<md5hash>.jpg``; labels and Fitzpatrick types come from the
dataset's ``fitzpatrick17k.csv`` (github.com/mattgroh/fitzpatrick17k).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .indexing import md5_of

# Our selection of papulovesicular / pustular / bullous / crusted conditions.
LOOKALIKE_CONDITIONS = [
    "folliculitis", "scabies", "acne vulgaris", "acne", "erythema multiforme",
    "drug eruption", "pityriasis rosea", "prurigo nodularis", "tungiasis", "urticaria",
    "pustular psoriasis", "stevens johnson syndrome", "fixed eruptions",
    "neurotic excoriations", "tick bite", "myiasis", "pityriasis lichenoides chronica",
    "epidermolysis bullosa", "acquired autoimmune bullous diseaseherpes gestationis",
    "hailey hailey disease", "dariers disease", "nematode infection", "keratosis pilaris",
    "milia",
]

# Never allowed in the pool: the classes we evaluate on.
EVAL_CLASSES = {"mpox", "monkeypox", "chickenpox", "varicella", "measles", "cowpox",
                "hand foot and mouth disease", "hand foot mouth disease", "hfmd"}


def find_image_dir(root: str | Path) -> Path:
    """Folder that holds most of the ``<md5>.jpg`` files."""
    root = Path(root)
    best, n_best = root, -1
    for d in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        n = sum(1 for f in d.glob("*.jpg") if len(f.stem) == 32)
        if n > n_best:
            best, n_best = d, n
    if n_best <= 0:
        raise RuntimeError(f"No <md5>.jpg files found under {root}")
    return best


def build_pool(image_root: str | Path, csv_path: str | Path, per_condition: int = 50,
               conditions=None, seed: int = 0) -> pd.DataFrame:
    """Sample up to ``per_condition`` images per look-alike condition.

    Within a condition, images are sampled with weights inversely proportional to
    the frequency of their Fitzpatrick type, so darker skin types are not
    under-represented. Images flagged "3 Wrongly labelled" in the QC column are
    dropped.
    """
    conditions = conditions or LOOKALIKE_CONDITIONS
    bad = {c for c in conditions if c in EVAL_CLASSES}
    if bad:
        raise ValueError(f"evaluation classes cannot be pool conditions: {bad}")
    img_dir = find_image_dir(image_root)
    meta = pd.read_csv(csv_path)
    meta = meta[meta["label"].isin(conditions)]
    meta = meta[~meta["qc"].fillna("").astype(str).str.startswith("3")]
    meta = meta[[(img_dir / f"{h}.jpg").exists() for h in meta["md5hash"]]]
    rng = np.random.default_rng(seed)
    picked = []
    for cond, g in meta.groupby("label", sort=True):
        if len(g) <= per_condition:
            picked.append(g)
            continue
        freq = g["fitzpatrick_scale"].map(g["fitzpatrick_scale"].value_counts())
        w = (1.0 / freq).to_numpy()
        idx = rng.choice(len(g), size=per_condition, replace=False, p=w / w.sum())
        picked.append(g.iloc[np.sort(idx)])
    pool = pd.concat(picked, ignore_index=True)
    paths = [str(img_dir / f"{h}.jpg") for h in pool["md5hash"]]
    md5 = [md5_of(p) for p in paths]
    out = pd.DataFrame({
        "image_id": ["fitz17k:" + m[:12] for m in md5],
        "dataset": "fitz17k",
        "path": paths,
        "label": "other",
        "condition": pool["label"].to_numpy(),
        "fitzpatrick": pool["fitzpatrick_scale"].to_numpy(),
        "md5": md5,
    })
    return out.drop_duplicates("md5").reset_index(drop=True)
