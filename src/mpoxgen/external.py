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


# Words that mark a condition as vesicular, pustular, bullous or otherwise
# pox-like; such conditions are kept out of the control pool even if they are
# not on our look-alike list.
CONTROL_EXCLUDE_WORDS = ("pox", "varicella", "herpes", "zoster", "pemphig", "bullous", "blister",
                         "impetigo", "molluscum", "hand foot", "vesic", "pustul", "eczema herpeticum")


def control_candidates(meta: pd.DataFrame, lookalike=None) -> pd.DataFrame:
    """Fitzpatrick17k rows from conditions that are neither look-alikes nor evaluation classes."""
    lookalike = set(lookalike or LOOKALIKE_CONDITIONS)
    lab = meta["label"].astype(str)
    bad = lab.isin(lookalike) | lab.isin(EVAL_CLASSES)
    bad |= lab.str.contains("|".join(CONTROL_EXCLUDE_WORDS), case=False, regex=True)
    return meta[~bad]


def build_control_pool(image_root: str | Path, csv_path: str | Path, match: pd.DataFrame,
                       per_condition: int = 50, oversample: float = 1.05, seed: int = 1) -> pd.DataFrame:
    """Control negatives for the look-alike pool: same size and Fitzpatrick-type mix.

    For every Fitzpatrick value in ``match["fitzpatrick"]`` we draw that many images
    (times ``oversample``, so the duplicate check can drop a few) from control
    conditions, spreading them over conditions (weight 1 / condition frequency)
    and taking at most ``per_condition`` images from any one condition. The
    returned frame has a ``rank`` column; keep the lowest ranks per Fitzpatrick
    value to trim to the target size after duplicates are removed.
    """
    img_dir = find_image_dir(image_root)
    meta = pd.read_csv(csv_path)
    meta = meta[~meta["qc"].fillna("").astype(str).str.startswith("3")]
    meta = control_candidates(meta)
    meta = meta[[(img_dir / f"{h}.jpg").exists() for h in meta["md5hash"]]].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    target = match["fitzpatrick"].value_counts()
    used = {}
    rows = []
    for fz in sorted(target.index):
        want = int(np.ceil(target[fz] * oversample))
        cand = meta[meta["fitzpatrick_scale"] == fz]
        if cand.empty:
            continue
        freq = cand["label"].map(cand["label"].value_counts()).to_numpy(dtype=float)
        w = 1.0 / freq
        order = rng.choice(len(cand), size=len(cand), replace=False, p=w / w.sum())
        k = 0
        for j in order:
            r = cand.iloc[j]
            if used.get(r["label"], 0) >= per_condition:
                continue
            used[r["label"]] = used.get(r["label"], 0) + 1
            rows.append((r, k))
            k += 1
            if k >= want:
                break
    pool = pd.DataFrame([r for r, _ in rows]).reset_index(drop=True)
    rank = [k for _, k in rows]
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
        "rank": rank,
    })
    return out.drop_duplicates("md5").reset_index(drop=True)


def trim_to_match(pool: pd.DataFrame, match: pd.DataFrame) -> pd.DataFrame:
    """Keep, per Fitzpatrick value, as many images as ``match`` has (lowest rank first)."""
    target = match["fitzpatrick"].value_counts()
    parts = [g.sort_values("rank").head(int(target.get(fz, 0))) for fz, g in pool.groupby("fitzpatrick")]
    return pd.concat(parts, ignore_index=True) if parts else pool.iloc[:0]


def build_atlas_pool(image_root: str | Path, csv_path: str | Path) -> pd.DataFrame:
    """Every usable Fitzpatrick17k image (the whole atlas) for FACTOR.

    Keeps all conditions except the evaluation classes, drops images flagged
    "3 Wrongly labelled" and exact file duplicates. Duplicates of study images
    are removed later (scripts/15_atlas_pool.py).
    """
    img_dir = find_image_dir(image_root)
    meta = pd.read_csv(csv_path)
    meta = meta[~meta["label"].astype(str).isin(EVAL_CLASSES)]
    meta = meta[~meta["qc"].fillna("").astype(str).str.startswith("3")]
    meta = meta[[(img_dir / f"{h}.jpg").exists() for h in meta["md5hash"]]].reset_index(drop=True)
    paths = [str(img_dir / f"{h}.jpg") for h in meta["md5hash"]]
    md5 = [md5_of(p) for p in paths]
    out = pd.DataFrame({
        "image_id": ["fitz17k:" + m[:12] for m in md5],
        "dataset": "fitz17k",
        "path": paths,
        "label": "other",
        "condition": meta["label"].to_numpy(),
        "fitzpatrick": meta["fitzpatrick_scale"].to_numpy(),
        "md5": md5,
    })
    return out.drop_duplicates("md5").drop_duplicates("image_id").reset_index(drop=True)
