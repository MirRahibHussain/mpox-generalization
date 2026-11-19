"""Step 21 - verify the released split files against the released predictions (CPU, no images).

For every split file in release/v2_evaluation/splits/ this script checks that
  * no image or duplicate group is shared between training(+validation) and test (P2, P3),
  * the test ids equal the image ids of every released predictions.csv of that split
    (evaluation study and all method sessions), and
  * the validation ids equal those of every released val_predictions.csv of that split,
and it reports which splits `scripts/04_make_splits.py` regenerates identically from the
released index. MSID P2 splits use scikit-learn's StratifiedGroupKFold with shuffling, whose
output depends on the scikit-learn version, so the released files are the reference.

Usage:  python scripts/21_verify_splits.py   (writes release/v2_evaluation/splits/verification.csv)
"""

import glob
import json
import sys
from pathlib import Path

import _paths  # noqa: F401
import pandas as pd

from mpoxgen import splits

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "release"


def released_ids():
    out = {}
    for kind, pat in [("test", "predictions.csv"), ("val", "val_predictions.csv")]:
        for f in glob.glob(str(REL / "*" / "results*" / "*" / "*" / pat)) + \
                 glob.glob(str(REL / "*" / "results*" / "*" / "*" / "*" / pat)):
            name = next(p for p in Path(f).parts if p.startswith(("binary_", "4class_")))
            ids = {i for i in pd.read_csv(f, usecols=["image_id"]).image_id if not str(i).startswith("mcsi")}
            out.setdefault(name, {"test": [], "val": []})[kind].append(ids)
    return out


def main():
    idx = REL / "v2_evaluation" / "index"
    images = pd.read_csv(idx / "images.csv")
    official = pd.read_csv(idx / "official_folds.csv")
    rel = released_ids()

    regen = {}
    for task in ["binary", "4class"]:
        specs = []
        for ds in ["msld_v2", "msid"]:
            specs += splits.naive_splits(images, ds, task, 5)
            specs += splits.patient_splits(images, official, ds, task, 5)
        specs += splits.cross_splits(images, "msld_v2", "msid", task, 3)
        specs += splits.cross_splits(images, "msid", "msld_v2", task, 3)
        for s in specs:
            regen[f"{s['task']}_{s['protocol']}_{s['train_dataset']}_to_{s['test_dataset']}_rep{s['rep']}"] = s

    rows, failed = [], 0
    for f in sorted((REL / "v2_evaluation" / "splits").glob("*.json")):
        s = json.loads(f.read_text())
        name = f.stem
        r = rel.get(name, {"test": [], "val": []})
        leak = splits.check_no_leak(s, images)
        strict = s["protocol"] in ("P2", "P3")
        test_ok = all(t == set(s["test"]) for t in r["test"])
        val_ok = all(v == set(s["val"]) for v in r["val"])
        g = regen.get(name)
        same = bool(g) and all(set(g[k]) == set(s[k]) for k in ("train", "val", "test"))
        ok = test_ok and val_ok and (not strict or (leak["shared_images"] == 0 and leak["shared_groups"] == 0))
        failed += not ok
        rows.append({"name": name, "n_train": len(s["train"]), "n_val": len(s["val"]), "n_test": len(s["test"]),
                     "released_test_files": len(r["test"]), "released_val_files": len(r["val"]),
                     "test_ids_match": test_ok, "val_ids_match": val_ok, **leak,
                     "regenerated_identically": same, "ok": bool(ok)})
    man = pd.DataFrame(rows)
    man.to_csv(REL / "v2_evaluation" / "splits" / "verification.csv", index=False)
    print(man.groupby(man.name.str.extract(r"^(\w+?_P\d)")[0]).agg(
        files=("name", "size"), checked_against=("released_test_files", "sum"),
        ok=("ok", "all"), regenerated_identically=("regenerated_identically", "sum")).to_string())
    if failed:
        sys.exit(f"{failed} split file(s) failed verification")
    print(f"All {len(man)} split files verified; {int(man.released_test_files.sum())} prediction files and "
          f"{int(man.released_val_files.sum())} validation-prediction files match their splits.")


if __name__ == "__main__":
    main()
