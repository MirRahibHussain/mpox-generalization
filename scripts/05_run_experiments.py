"""Step 5 (GPU) - train and evaluate every (split, model). Resumable.

Each run writes results/<split>/<model>/{metrics.json,predictions.csv,history.csv}.
Runs whose metrics.json exists are skipped, so you can stop and restart.

Smoke test first (about 5 minutes):
    python scripts/05_run_experiments.py --smoke
Full binary study:
    python scripts/05_run_experiments.py --tasks binary
"""

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
import pandas as pd

from mpoxgen.train import MODELS, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--tasks", nargs="+", default=["binary"])
    ap.add_argument("--protocols", nargs="+", default=["P1", "P2", "P3"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save-ckpt", action="store_true",
                    help="save weights for rep 0 of P2 and P3 (needed for Grad-CAM)")
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch, 1 model, first split of each protocol, written to results_smoke/")
    a = ap.parse_args()

    images = pd.read_csv(Path(a.index) / "images.csv")
    man = pd.read_csv(Path(a.splits) / "manifest.csv")
    man = man[man["task"].isin(a.tasks) & man["protocol"].isin(a.protocols)]
    out = Path(a.out)
    models = a.models
    if a.smoke:
        out = Path("results_smoke")
        models = ["mobilenetv3_large_100"]
        man = man.groupby(["protocol"]).head(1)
        a.epochs, a.patience = 1, 1

    todo = [(r.name, m) for r in man.itertuples() for m in models]
    print(f"{len(todo)} runs ({len(man)} splits x {len(models)} models) -> {out}/")
    for k, (name, model) in enumerate(todo, 1):
        run_dir = out / name / model
        if (run_dir / "metrics.json").exists():
            continue
        spec = json.loads((Path(a.splits) / f"{name}.json").read_text())
        ckpt = a.save_ckpt and spec["rep"] == 0 and spec["protocol"] in ("P2", "P3")
        rec = run(spec, images, model, seed=spec["rep"], out_dir=run_dir, epochs=a.epochs,
                  patience=a.patience, batch_size=a.batch_size, num_workers=a.workers,
                  save_checkpoint=ckpt)
        print(f"[{k}/{len(todo)}] {name} {model}: bal_acc={rec['balanced_accuracy']:.3f} "
              f"auroc={rec['auroc']:.3f} ({rec['minutes']} min)")

    recs = [json.loads(p.read_text()) for p in out.glob("*/*/metrics.json")]
    pd.DataFrame(recs).to_csv(out / "all_runs.csv", index=False)
    print(f"Collected {len(recs)} finished runs into {out}/all_runs.csv")


if __name__ == "__main__":
    main()
