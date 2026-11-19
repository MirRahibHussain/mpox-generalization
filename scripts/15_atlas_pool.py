"""Step 15 - the whole Fitzpatrick17k atlas as auxiliary negatives for FACTOR.

1. Take every usable Fitzpatrick17k image (all conditions except the
   evaluation classes; QC "wrongly labelled" dropped; exact file duplicates
   dropped).
2. Remove every atlas image that duplicates an MSLD v2.0 or MSID image:
   pHash AND dHash within --max-dist bits (mirrored copies included), or
   ResNet-50 embedding cosine >= --cos (default 0.92, stricter than the
   duplicate audit of the study data). Duplicates inside the atlas are kept:
   they cannot leak into a test set.
3. Estimate ITA for every atlas image (in parallel).
4. Write pool.csv, pool_report.txt and pool_examples.png.

    python scripts/15_atlas_pool.py --fitz /kaggle/input/fitzpatrick17k-original \
        --csv fitzpatrick17k.csv --index data/index --out data/pool_atlas
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import pandas as pd

from mpoxgen import dedup, external, ita
from mpoxgen.hashing import hamming_matrix


def _ita(path):
    try:
        e = ita.estimate(path)
        return e["ita"], e["ita_band"]
    except Exception:
        return np.nan, "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fitz", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--out", default="data/pool_atlas")
    ap.add_argument("--max-dist", type=int, default=6)
    ap.add_argument("--cos", type=float, default=0.92)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    study = pd.read_csv(Path(a.index) / "images.csv")

    atlas = external.build_atlas_pool(a.fitz, a.csv)
    n0, c0 = len(atlas), atlas["condition"].nunique()
    print(f"atlas: {n0} images from {c0} conditions", flush=True)

    # Hash duplicates of study images (atlas x study only).
    hs = dedup.compute_hashes(study, verbose=False)
    ha = dedup.compute_hashes(atlas, verbose=False)
    direct = np.maximum(hamming_matrix(ha["phash"], hs["phash"]), hamming_matrix(ha["dhash"], hs["dhash"]))
    mirror = np.maximum(hamming_matrix(ha["phash_flip"], hs["phash"]), hamming_matrix(ha["dhash_flip"], hs["dhash"]))
    hit_hash = (np.minimum(direct, mirror) <= a.max_dist).any(axis=1)
    n_hash = int(hit_hash.sum())

    # Embedding duplicates (second line of defence).
    hit_emb = np.zeros(len(atlas), bool)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("lp", Path(__file__).with_name("08_lookalike_pool.py"))
        lp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lp)
        emb_file = Path(a.index) / "embeddings.npy"
        f_main = np.load(emb_file) if emb_file.exists() else lp.embed(study["path"].tolist())
        f_atlas = lp.embed(atlas["path"].tolist())
        hit_emb = (f_atlas @ f_main.T).max(1) >= a.cos
    except Exception as e:
        print(f"[warn] embedding check skipped: {e}")
    n_emb = int((hit_emb & ~hit_hash).sum())
    atlas = atlas[~(hit_hash | hit_emb)].reset_index(drop=True)

    with Pool(a.workers) as p:
        res = p.map(_ita, atlas["path"].tolist(), chunksize=64)
    atlas["ita"] = [r[0] for r in res]
    atlas["ita_band"] = [r[1] for r in res]
    atlas.to_csv(out / "pool.csv", index=False)

    per = atlas["condition"].value_counts()
    report = [
        f"atlas images (usable, exact duplicates dropped): {n0} from {c0} conditions",
        f"removed as hash duplicates of study images: {n_hash}",
        f"removed by embedding cosine >= {a.cos} with a study image (not already caught by hashes): {n_emb}",
        f"final atlas pool: {len(atlas)} images from {atlas['condition'].nunique()} conditions",
        f"images per condition: median {int(per.median())}, min {int(per.min())}, max {int(per.max())}",
        "", "by Fitzpatrick type:", atlas["fitzpatrick"].value_counts().sort_index().to_string(),
        "", "by ITA band:", atlas["ita_band"].value_counts().to_string(),
        "", "conditions (largest 30):", per.head(30).to_string(),
    ]
    (out / "pool_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    try:
        spec = importlib.util.spec_from_file_location("lp", Path(__file__).with_name("08_lookalike_pool.py"))
        lp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lp)
        ex = atlas.groupby("condition").head(1).sample(min(48, atlas["condition"].nunique()), random_state=0)
        lp.sheet([(p, c) for p, c in zip(ex["path"], ex["condition"])], 8, out / "pool_examples.png")
    except Exception as e:
        print(f"[warn] example sheet skipped: {e}")


if __name__ == "__main__":
    main()
