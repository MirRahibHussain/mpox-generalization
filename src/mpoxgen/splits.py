"""The three evaluation protocols.

P1 naive        : random image-level 70/20/10 splits (what many papers do).
P2 patient-level: official patient-independent folds when the dataset has
                  them, otherwise group splits by duplicate group; in both
                  cases no duplicate group crosses train/val and test.
P3 cross-dataset: train on dataset A, test on dataset B minus any image that
                  duplicates an image of A.

Each split is a plain dict, saved as JSON so others can reuse it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit, train_test_split

from .labels import task_target


def _task_frame(images: pd.DataFrame, task: str) -> pd.DataFrame:
    df = images.copy()
    df["target"] = [task_target(l, task) for l in df["label"]]
    return df[df["target"].notna()].astype({"target": int})


def _spec(protocol, task, rep, train_ds, test_ds, tr, va, te, notes=None):
    return {
        "protocol": protocol, "task": task, "rep": int(rep),
        "train_dataset": train_ds, "test_dataset": test_ds,
        "train": sorted(tr), "val": sorted(va), "test": sorted(te),
        "notes": notes or {},
    }


def naive_splits(images, dataset, task, n_reps=5, seed=0):
    df = _task_frame(images[images["dataset"] == dataset], task)
    out = []
    for r in range(n_reps):
        rest, te = train_test_split(df, test_size=0.10, stratify=df["target"], random_state=seed + r)
        tr, va = train_test_split(rest, test_size=2 / 9, stratify=rest["target"], random_state=seed + r)
        out.append(_spec("P1", task, r, dataset, dataset, tr["image_id"], va["image_id"], te["image_id"]))
    return out


def _drop_group_overlap(df, tr_ids, va_ids, te_ids):
    """Remove from train/val any image sharing a duplicate group with test."""
    g = df.set_index("image_id")["dup_group"]
    test_groups = set(g.loc[list(te_ids)])
    tr2 = [i for i in tr_ids if g.loc[i] not in test_groups]
    va2 = [i for i in va_ids if g.loc[i] not in test_groups]
    # And keep val groups out of train, so early stopping is honest.
    val_groups = set(g.loc[va2])
    tr2 = [i for i in tr2 if g.loc[i] not in val_groups]
    return tr2, va2, (len(tr_ids) - len(tr2)) + (len(va_ids) - len(va2))


def patient_splits(images, official, dataset, task, n_reps=5, seed=0):
    df = _task_frame(images[images["dataset"] == dataset], task)
    ids = set(df["image_id"])
    off = official[official["image_id"].isin(ids)]
    out = []
    if len(off) and off["fold"].nunique() >= 2:
        for k, (fold, fo) in enumerate(sorted(off.groupby("fold"), key=lambda kv: kv[0])):
            if k >= n_reps:
                break
            tr = set(fo.loc[fo["split"] == "train", "image_id"])
            va = set(fo.loc[fo["split"] == "val", "image_id"])
            te = set(fo.loc[fo["split"] == "test", "image_id"])
            # Defensive: a file listed in two splits of the same fold stays only in test.
            tr -= te
            va -= te
            tr -= va
            tr2, va2, dropped = _drop_group_overlap(df, tr, va, te)
            out.append(_spec("P2", task, k, dataset, dataset, tr2, va2, te,
                             {"source": f"official fold {fold}", "dropped_for_duplicates": dropped}))
        return out

    # No official folds: 10 group-stratified parts; rep k -> test part k,
    # val parts k+1,k+2, train the rest (70/20/10).
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    part = np.empty(len(df), dtype=int)
    for p, (_, idx) in enumerate(sgkf.split(df, df["target"], groups=df["dup_group"])):
        part[idx] = p
    for k in range(n_reps):
        te_p = {k % 10}
        va_p = {(k + 1) % 10, (k + 2) % 10}
        te = df.loc[np.isin(part, list(te_p)), "image_id"]
        va = df.loc[np.isin(part, list(va_p)), "image_id"]
        tr = df.loc[~np.isin(part, list(te_p | va_p)), "image_id"]
        out.append(_spec("P2", task, k, dataset, dataset, tr, va, te,
                         {"source": "group split by duplicate group (no patient ids)"}))
    return out


def cross_splits(images, train_ds, test_ds, task, n_reps=3, seed=0):
    df = _task_frame(images, task)
    a = df[df["dataset"] == train_ds]
    b = df[df["dataset"] == test_ds]
    a_groups = set(a["dup_group"])
    b_clean = b[~b["dup_group"].isin(a_groups)]
    removed = len(b) - len(b_clean)
    out = []
    for r in range(n_reps):
        # 90/10 group-aware split of A for early stopping only.
        gss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=seed + r)
        grp = a.groupby("dup_group")["target"].first()
        g_tr, g_va = next(gss.split(grp.index.to_numpy().reshape(-1, 1), grp.to_numpy()))
        va_groups = set(grp.index[g_va])
        va = a.loc[a["dup_group"].isin(va_groups), "image_id"]
        tr = a.loc[~a["dup_group"].isin(va_groups), "image_id"]
        out.append(_spec("P3", task, r, train_ds, test_ds, tr, va, b_clean["image_id"],
                         {"test_removed_as_duplicates_of_train": int(removed)}))
    return out


def check_no_leak(spec: dict, images: pd.DataFrame) -> dict:
    """Counts of shared images / duplicate groups between train(+val) and test."""
    g = images.set_index("image_id")["dup_group"]
    trv = set(spec["train"]) | set(spec["val"])
    te = set(spec["test"])
    return {
        "shared_images": len(trv & te),
        "shared_groups": len(set(g.loc[list(trv)]) & set(g.loc[list(te)])),
    }
