"""Step 17 (CPU) - seen vs unseen negative classes, and what each factor of FACTOR learned.

For every cross-dataset (P3) run the test images are split by whether their
class occurs in the training dataset:

* closed_bal_acc - balanced accuracy on mpox + non-mpox classes seen in training;
* unseen_spec   - specificity on non-mpox classes never seen in training
  (MSLD v2.0 cowpox and HFMD for MSID-trained models; empty the other way);
* disease_spec / healthy_spec - specificity on diseased non-mpox / healthy images;
* auroc_vs_disease - AUROC of the mpox score, mpox vs other diseases only;
* auroc_vs_healthy - AUROC of the mpox score, mpox vs healthy only.

For FACTOR runs (predictions with ``pd``/``pm`` columns) it also reports the
AUROC of the disease factor (diseased vs healthy) and of the mpox factor
(mpox vs other diseases, diseased images only).

Writes open_set_runs.csv (one row per run, the format scripts/13_paired_stats.py
reads) and open_set.md.

    python scripts/17_open_set.py --results results_factor --index data/index
"""

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DIRS = {"msid->msld_v2": "MSID → MSLD v2.0", "msld_v2->msid": "MSLD v2.0 → MSID"}
METRICS = ["closed_bal_acc", "unseen_spec", "disease_spec", "healthy_spec",
           "auroc_vs_disease", "auroc_vs_healthy", "factor_disease_auroc", "factor_mpox_auroc"]


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def run_row(run_dir: Path, rec: dict, lab: pd.Series, train_classes: set) -> dict:
    p = pd.read_csv(run_dir / "predictions.csv")
    p["label"] = p["image_id"].map(lab)
    mp = p["label"] == "mpox"
    neg = ~mp
    seen = p["label"].isin(train_classes)
    healthy = p["label"] == "healthy"
    call = p["y_pred"] == 1
    sens = float(call[mp].mean())
    spec_seen = float((~call[neg & seen]).mean()) if (neg & seen).any() else np.nan
    row = {k: rec[k] for k in ["protocol", "model", "rep", "method"]}
    row["direction"] = f"{rec['train_dataset']}->{rec['test_dataset']}"
    row.update({
        "sens": sens,
        "closed_bal_acc": (sens + spec_seen) / 2,
        "unseen_spec": float((~call[neg & ~seen]).mean()) if (neg & ~seen).any() else np.nan,
        "n_unseen": int((neg & ~seen).sum()),
        "disease_spec": float((~call[neg & ~healthy]).mean()) if (neg & ~healthy).any() else np.nan,
        "healthy_spec": float((~call[healthy]).mean()) if healthy.any() else np.nan,
        "auroc_vs_disease": auc(mp[mp | (neg & ~healthy)], p.loc[mp | (neg & ~healthy), "p1"]),
        "auroc_vs_healthy": auc(mp[mp | healthy], p.loc[mp | healthy, "p1"]),
        "factor_disease_auroc": np.nan, "factor_mpox_auroc": np.nan,
    })
    if "pd" in p and "pm" in p:
        row["factor_disease_auroc"] = auc(~healthy, p["pd"])
        dis = ~healthy
        row["factor_mpox_auroc"] = auc(mp[dis], p.loc[dis, "pm"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_factor")
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--order", nargs="*", default=["erm", "rand", "atlas", "div5", "div20", "hod",
                                                   "factor_static", "factor_notone", "factor"])
    a = ap.parse_args()
    res = Path(a.results)
    images = pd.read_csv(Path(a.index) / "images.csv")
    lab = images.set_index("image_id")["label"]
    classes = images.groupby("dataset")["label"].apply(set).to_dict()
    rows = []
    for m in sorted(res.glob("*/*/*/metrics.json")):
        rec = json.loads(m.read_text())
        if rec.get("protocol") != "P3" or rec.get("task", "binary") != "binary":
            continue
        rows.append(run_row(m.parent, rec, lab, classes.get(rec["train_dataset"], set())))
    runs = pd.DataFrame(rows)
    runs.to_csv(res / "open_set_runs.csv", index=False)

    meths = [x for x in a.order if x in set(runs["method"])] + sorted(set(runs["method"]) - set(a.order))
    text = ["# Seen vs unseen negative classes (P3, mean ± SD over backbones and seeds)", ""]
    for d, dn in DIRS.items():
        sub = runs[runs["direction"] == d]
        if sub.empty:
            continue
        unseen = sorted(set(images.loc[images["dataset"] == d.split("->")[1], "label"])
                        - classes.get(d.split("->")[0], set()) - {"mpox"})
        text += [f"## {dn} (classes not seen in training: {', '.join(unseen) or 'none'})", "",
                 "| method | runs | sens. | closed-set bal. acc. | unseen-class spec. | disease spec. | "
                 "healthy spec. | AUROC vs disease | AUROC vs healthy | disease-factor AUROC | mpox-factor AUROC |",
                 "| --- " * 11 + "|"]
        for meth in meths:
            g = sub[sub["method"] == meth]
            if g.empty:
                continue
            cells = []
            for c in ["sens"] + METRICS:
                x = g[c].dropna()
                cells.append(f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1
                             else (f"{x.mean():.3f}" if len(x) else "–"))
            text.append(f"| {meth} | {len(g)} | " + " | ".join(cells) + " |")
        text.append("")
    (res / "open_set.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
