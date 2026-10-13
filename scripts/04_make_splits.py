"""Step 4 - write P1/P2/P3 splits as JSON, with a leakage check."""

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
import pandas as pd

from mpoxgen import splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--out", default="splits")
    ap.add_argument("--datasets", nargs=2, default=["msld_v2", "msid"])
    ap.add_argument("--tasks", nargs="+", default=["binary", "4class"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cross-seeds", type=int, default=3)
    a = ap.parse_args()
    idx, out = Path(a.index), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    images = pd.read_csv(idx / "images.csv")
    official = pd.read_csv(idx / "official_folds.csv")
    if "dup_group" not in images:
        raise SystemExit("Run 03_duplicate_audit.py first (dup_group missing).")

    specs = []
    for task in a.tasks:
        for ds in a.datasets:
            specs += splits.naive_splits(images, ds, task, a.reps)
            specs += splits.patient_splits(images, official, ds, task, a.reps)
        d1, d2 = a.datasets
        specs += splits.cross_splits(images, d1, d2, task, a.cross_seeds)
        specs += splits.cross_splits(images, d2, d1, task, a.cross_seeds)

    rows = []
    for s in specs:
        name = f"{s['task']}_{s['protocol']}_{s['train_dataset']}_to_{s['test_dataset']}_rep{s['rep']}"
        s["name"] = name
        (out / f"{name}.json").write_text(json.dumps(s))
        leak = splits.check_no_leak(s, images)
        rows.append({"name": name, "task": s["task"], "protocol": s["protocol"],
                     "train_dataset": s["train_dataset"], "test_dataset": s["test_dataset"],
                     "rep": s["rep"], "n_train": len(s["train"]), "n_val": len(s["val"]),
                     "n_test": len(s["test"]), **leak, **s["notes"]})
    man = pd.DataFrame(rows)
    man.to_csv(out / "manifest.csv", index=False)

    strict = man[man["protocol"].isin(["P2", "P3"])]
    if (strict["shared_images"] > 0).any() or (strict["shared_groups"] > 0).any():
        raise SystemExit("Leak found in a P2/P3 split - see manifest.csv")
    p1 = man[man["protocol"] == "P1"]
    print(f"Wrote {len(man)} splits. P2/P3: no shared images or duplicate groups.")
    print("P1 (naive): duplicate groups shared by train(+val) and test, mean per split:")
    print(p1.groupby(["task", "train_dataset"])["shared_groups"].mean().to_string())


if __name__ == "__main__":
    main()
