"""Step 3 (CPU) - duplicate audit + skin-tone (ITA) estimate.

Hash pairs (pHash AND dHash within --max-dist bits, mirrored copies included)
are merged with accepted embedding pairs from step 2, then grouped. Adds
columns dup_group, ita, ita_band to images.csv.
"""

import argparse
from pathlib import Path

import _paths  # noqa: F401
import pandas as pd

from mpoxgen import dedup, ita
from mpoxgen.viz import pair_sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--max-dist", type=int, default=6)
    ap.add_argument("--no-ita", action="store_true")
    a = ap.parse_args()
    idx = Path(a.index)
    images = pd.read_csv(idx / "images.csv")

    print(f"Hashing {len(images)} images ...")
    hashes = dedup.compute_hashes(images)
    pairs = dedup.hash_pairs(images, hashes, a.max_dist)
    emb_file = idx / "embedding_pairs.csv"
    if emb_file.exists():
        emb = pd.read_csv(emb_file)
        emb = emb[emb["accept"] == 1].drop(columns=["accept"])
        print(f"  + {len(emb)} accepted embedding pairs")
        pairs = pd.concat([pairs, emb], ignore_index=True)
    else:
        print("  (no embedding_pairs.csv - hash pairs only)")

    images["dup_group"] = dedup.assign_groups(images, pairs)
    pairs.to_csv(idx / "duplicate_pairs.csv", index=False)
    summ = dedup.summary(images, pairs)
    summ.to_csv(idx / "duplicate_summary.csv", index=False)
    print(summ.to_string(index=False))

    pair_sheet(pairs[pairs["cross_dataset"]], images, str(idx / "cross_dataset_duplicates.png"))
    pair_sheet(pairs[pairs["label_mismatch"]], images, str(idx / "label_mismatch_duplicates.png"))

    if not a.no_ita:
        print("Estimating ITA ...")
        est = [ita.estimate(p) for p in images["path"]]
        images["ita"] = [e["ita"] for e in est]
        images["ita_band"] = [e["ita_band"] for e in est]
        print(images.groupby(["dataset", "label", "ita_band"]).size().unstack(fill_value=0).to_string())

    images.to_csv(idx / "images.csv", index=False)


if __name__ == "__main__":
    main()
