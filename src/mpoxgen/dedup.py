"""Duplicate audit: find duplicate pairs and merge them into duplicate groups."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hashing import hamming_matrix, hash_image, pack


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def compute_hashes(images: pd.DataFrame, verbose: bool = True) -> dict[str, np.ndarray]:
    keys = ["phash", "dhash", "phash_flip", "dhash_flip"]
    store = {k: [] for k in keys}
    for i, p in enumerate(images["path"]):
        h = hash_image(p)
        for k in keys:
            store[k].append(h[k])
        if verbose and (i + 1) % 250 == 0:
            print(f"  hashed {i + 1}/{len(images)}")
    return {k: pack(np.stack(v)) for k, v in store.items()}


def hash_pairs(images: pd.DataFrame, hashes: dict[str, np.ndarray], max_dist: int = 6) -> pd.DataFrame:
    """Pairs (i<j) whose pHash AND dHash are both within ``max_dist`` bits,
    either directly or after mirroring one image.

    Requiring both hashes to agree keeps false positives low; the embedding
    step catches heavier edits.
    """
    n = len(images)
    direct = np.maximum(hamming_matrix(hashes["phash"], hashes["phash"]),
                        hamming_matrix(hashes["dhash"], hashes["dhash"]))
    mirror = np.maximum(hamming_matrix(hashes["phash"], hashes["phash_flip"]),
                        hamming_matrix(hashes["dhash"], hashes["dhash_flip"]))
    dist = np.minimum(direct, mirror)
    iu = np.triu_indices(n, k=1)
    mask = dist[iu] <= max_dist
    i, j = iu[0][mask], iu[1][mask]
    return pairs_frame(images, i, j, "hash", dist[i, j].astype(float))


def pairs_frame(images: pd.DataFrame, i, j, method: str, score) -> pd.DataFrame:
    ids = images["image_id"].to_numpy()
    ds = images["dataset"].to_numpy()
    lab = images["label"].to_numpy()
    return pd.DataFrame({
        "a": ids[i], "b": ids[j], "method": method, "score": score,
        "dataset_a": ds[i], "dataset_b": ds[j],
        "label_a": lab[i], "label_b": lab[j],
        "cross_dataset": ds[i] != ds[j],
        "label_mismatch": lab[i] != lab[j],
    })


def assign_groups(images: pd.DataFrame, pairs: pd.DataFrame) -> pd.Series:
    """Duplicate-group id per image (connected components of the pair graph)."""
    pos = {iid: k for k, iid in enumerate(images["image_id"])}
    uf = UnionFind(len(images))
    for a, b in zip(pairs["a"], pairs["b"]):
        if a in pos and b in pos:
            uf.union(pos[a], pos[b])
    roots = [uf.find(k) for k in range(len(images))]
    # Stable, readable ids: g00000, g00001, ...
    remap = {r: f"g{n:05d}" for n, r in enumerate(dict.fromkeys(roots))}
    return pd.Series([remap[r] for r in roots], index=images.index, name="dup_group")


def summary(images: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sizes = images.groupby("dup_group").size()
    multi = sizes[sizes > 1]
    for ds, g in images.groupby("dataset"):
        in_multi = g["dup_group"].isin(multi.index).sum()
        rows.append({"scope": ds, "images": len(g),
                     "images_in_duplicate_groups": int(in_multi),
                     "pairs": int(((pairs["dataset_a"] == ds) & (pairs["dataset_b"] == ds)).sum()),
                     "label_mismatch_pairs": int(((pairs["dataset_a"] == ds) & (pairs["dataset_b"] == ds)
                                                  & pairs["label_mismatch"]).sum())})
    cross = pairs[pairs["cross_dataset"]]
    rows.append({"scope": "cross-dataset", "images": np.nan,
                 "images_in_duplicate_groups": int(len(set(cross["a"]) | set(cross["b"]))),
                 "pairs": int(len(cross)),
                 "label_mismatch_pairs": int(cross["label_mismatch"].sum())})
    return pd.DataFrame(rows)
