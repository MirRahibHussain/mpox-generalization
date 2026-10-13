"""Step 1 - index the datasets (one row per unique image file).

Example:
    python scripts/01_build_index.py \
        --data msld_v2=/kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20 \
        --data msid=/kaggle/input/monkeypoxskinimagedataset \
        --out data/index
"""

import argparse
from pathlib import Path

import _paths  # noqa: F401
import pandas as pd

from mpoxgen import indexing

EXPECTED = {"msld_v2": 755}  # from the MSLD v2.0 paper (Ali et al., 2023)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True, help="name=path (repeatable)")
    ap.add_argument("--out", default="data/index")
    ap.add_argument("--exclude", nargs="*", default=["aug"],
                    help="skip folders whose relative path contains any of these words")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    all_img, all_off, conflicts = [], [], []
    for item in a.data:
        name, path = item.split("=", 1)
        print(f"Indexing {name} from {path}")
        img, off, info = indexing.build(path, name, tuple(a.exclude))
        print(f"  {info['n_files']} files -> {len(img)} unique images")
        if name in EXPECTED and len(img) != EXPECTED[name]:
            print(f"  [check] expected {EXPECTED[name]} unique images for {name}, found {len(img)}. "
                  "Inspect index_report.txt before continuing.")
        if name in EXPECTED and off.empty:
            print(f"  [WARNING] no official foldN/Train|Valid|Test folders found for {name}. "
                  "P2 would fall back to duplicate-group splits, which are NOT patient-level. "
                  "Point --data at the folder that contains the original-image FOLDS.")
        all_img.append(img)
        all_off.append(off)
        conflicts.append(info["label_conflicts"])

    images = pd.concat(all_img, ignore_index=True)
    official = pd.concat(all_off, ignore_index=True)
    images.to_csv(out / "images.csv", index=False)
    official.to_csv(out / "official_folds.csv", index=False)
    pd.concat(conflicts).to_csv(out / "label_conflicts.csv", index=False)

    report = indexing.describe(images, official)
    examples = "\n".join(f"  {r.dataset}: {Path(r.path).name}  ->  patient_hint={r.patient_hint}"
                         for r in images.groupby("dataset").head(3).itertuples())
    text = report + "\n\nExample file names:\n" + examples + "\n"
    (out / "index_report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
