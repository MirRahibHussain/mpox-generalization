"""Step 9 (GPU) - train and evaluate the proposed method and the baselines.

Every method in --methods runs on the cross-dataset splits (P3, both
directions, 3 seeds); methods in --p2-methods also run on the patient-level
splits (P2) to check in-dataset accuracy. Output:
results_methods/<split>/<model>/<method>/{metrics.json,predictions.csv,history.csv}.
Finished runs are skipped, so the script can be restarted.

    python scripts/09_run_methods.py --smoke
    python scripts/09_run_methods.py --max-hours 10
"""

import argparse
import json
import os
import time
from pathlib import Path

import _paths  # noqa: F401
import pandas as pd

from mpoxgen import cache
from mpoxgen.methods import CONFIGS
from mpoxgen.train import run_method

ORDER = ["erm", "tale", "la", "tone", "hsv", "mixup", "gdro"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--pool", default="data/pool/pool.csv")
    ap.add_argument("--control-pool", default="data/pool_control/pool.csv")
    ap.add_argument("--atlas-pool", default="data/pool_atlas/pool.csv")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--out", default="results_methods")
    ap.add_argument("--models", nargs="+", default=["resnet50", "efficientnet_b0", "vit_small_patch16_224"])
    ap.add_argument("--methods", nargs="*", default=ORDER)
    ap.add_argument("--p2-methods", nargs="*", default=["erm", "tale"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache-dir", default="cache/img288")
    ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--smoke", action="store_true", help="1 epoch, 1 model, 1 split per protocol")
    ap.add_argument("--train-dataset", default="", help="only P3 splits trained on this dataset")
    ap.add_argument("--order", choices=["method", "split"], default="method",
                    help="job order: all splits of one method first, or all methods of one split first "
                         "(with a time budget, 'split' leaves every method with the same finished splits)")
    a = ap.parse_args()
    for m in a.methods + a.p2_methods:
        if m not in CONFIGS:
            raise SystemExit(f"unknown method {m}; choose from {list(CONFIGS)}")

    images = pd.read_csv(Path(a.index) / "images.csv")
    pools = {"lookalike": a.pool, "control": a.control_pool, "atlas": a.atlas_pool}
    needed = {CONFIGS[m]["extra"] for m in a.methods + a.p2_methods if CONFIGS[m]["extra"]}
    pools = {k: pd.read_csv(v) for k, v in pools.items() if k in needed and Path(v).exists()}
    if a.cache_dir:
        images = cache.build(images, a.cache_dir)
        pools = {k: cache.build(v, a.cache_dir) for k, v in pools.items()}
    man = pd.read_csv(Path(a.splits) / "manifest.csv")
    man = man[man["task"] == "binary"]
    p3 = man[man["protocol"] == "P3"].sort_values(["train_dataset", "rep"])
    if a.train_dataset:
        p3 = p3[p3["train_dataset"] == a.train_dataset]
    p2 = man[man["protocol"] == "P2"].sort_values(["train_dataset", "rep"])
    out = Path(a.out)
    models, methods, p2_methods = a.models, a.methods, a.p2_methods
    if a.smoke:
        out = Path("results_methods_smoke")
        models = ["mobilenetv3_large_100"]
        p3, p2 = p3.groupby("train_dataset").head(1), p2.head(1)
        a.epochs, a.patience = 1, 1

    jobs = []
    for split_rows, meths in ((p3, methods), (p2, p2_methods)):  # P3 first: the question the method is for
        if a.order == "method":
            jobs += [(r.name, mdl, m) for m in meths for r in split_rows.itertuples() for mdl in models]
        else:
            jobs += [(r.name, mdl, m) for r in split_rows.itertuples() for mdl in models for m in meths]

    start = float(os.environ.get("PIPELINE_START", time.time()))
    deadline = start + a.max_hours * 3600 if a.max_hours > 0 else None
    print(f"{len(jobs)} runs -> {out}/  (pools: {', '.join(f'{k} {len(v)}' for k, v in pools.items()) or 'none'})", flush=True)
    skipped = 0
    for k, (name, mdl, meth) in enumerate(jobs, 1):
        run_dir = out / name / mdl / meth
        if (run_dir / "metrics.json").exists():
            continue
        if deadline and time.time() > deadline:
            skipped += 1
            continue
        spec = json.loads((Path(a.splits) / f"{name}.json").read_text())
        extra = pools.get(CONFIGS[meth]["extra"]) if CONFIGS[meth]["extra"] else None
        rec = run_method(spec, images, mdl, seed=spec["rep"], out_dir=run_dir, method=meth, extra=extra,
                         epochs=a.epochs, patience=a.patience, num_workers=a.workers)
        el = (time.time() - start) / 3600
        print(f"[{k}/{len(jobs)}] {name} {mdl} {meth}: bal_acc={rec['balanced_accuracy']:.3f} "
              f"sens={rec['mpox_sensitivity']:.3f} spec={rec['mpox_specificity']:.3f} "
              f"({rec['minutes']} min; {el:.2f} h)", flush=True)
    if skipped:
        print(f"[time budget] {skipped} runs not started")
    recs = [json.loads(p.read_text()) for p in out.glob("*/*/*/metrics.json")]
    pd.DataFrame(recs).to_csv(out / "all_runs.csv", index=False)
    print(f"Collected {len(recs)} runs into {out}/all_runs.csv")


if __name__ == "__main__":
    main()
