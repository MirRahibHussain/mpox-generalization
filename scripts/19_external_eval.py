"""Step 19 (CPU) - results on the external MCSI test set and the disease-presence shortcut.

For every P3-trained run with predictions_<name>.csv (default name "mcsi"):

* balanced accuracy, sensitivity, specificity and AUROC on the clean MCSI set;
* the share of each MCSI class called mpox (acne is a disease that neither
  MSLD v2.0 nor MSID contains);
* the disease-presence shortcut: AUROC of the mpox score for mpox against
  healthy skin only and against other diseases only (chickenpox, acne). A model
  that has learned "visible disease = mpox" separates mpox from healthy skin
  well but mpox from other diseases poorly; the gap measures the shortcut;
* sensitivity and specificity per ITA band.

Writes external_runs.csv (one row per run; "direction" = training dataset ->
name, the format scripts/13_paired_stats.py reads) and external.md.

    python scripts/19_external_eval.py --results results_external --external data/external_mcsi/images_all.csv
"""

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

NAMES = {"msid": "MSID", "msld_v2": "MSLD v2.0", "mcsi": "MCSI"}
COLS = ["balanced_accuracy", "sens", "spec", "auroc", "auroc_vs_healthy", "auroc_vs_disease", "shortcut_gap",
        "called_acne", "called_chickenpox", "called_healthy", "sens_dark", "sens_light", "spec_dark", "spec_light"]


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def run_row(pred: pd.DataFrame, info: pd.DataFrame) -> dict:
    p = pred.merge(info[["image_id", "label", "ita_band"]], on="image_id", how="left")
    mp, healthy = p["label"] == "mpox", p["label"] == "healthy"
    dis = ~mp & ~healthy
    call = p["y_pred"] == 1
    out = {"sens": float(call[mp].mean()), "spec": float((~call[~mp]).mean())}
    out["balanced_accuracy"] = (out["sens"] + out["spec"]) / 2
    out["auroc"] = auc(mp, p["p1"])
    out["auroc_vs_healthy"] = auc(mp[mp | healthy], p.loc[mp | healthy, "p1"])
    out["auroc_vs_disease"] = auc(mp[mp | dis], p.loc[mp | dis, "p1"])
    out["shortcut_gap"] = out["auroc_vs_healthy"] - out["auroc_vs_disease"]
    for c in ["acne", "chickenpox", "healthy"]:
        s = p["label"] == c
        out[f"called_{c}"] = float(call[s].mean()) if s.any() else np.nan
    for b in ["dark", "intermediate", "light"]:
        s = p["ita_band"] == b
        out[f"sens_{b}"] = float(call[s & mp].mean()) if (s & mp).any() else np.nan
        out[f"spec_{b}"] = float((~call[s & ~mp]).mean()) if (s & ~mp).any() else np.nan
    if "pd" in p and "pm" in p:
        out["factor_disease_auroc"] = auc(~healthy, p["pd"])
        out["factor_mpox_auroc"] = auc(mp[~healthy], p.loc[~healthy, "pm"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_external")
    ap.add_argument("--external", default="data/external_mcsi/images_all.csv")
    ap.add_argument("--name", default="mcsi")
    ap.add_argument("--order", nargs="*", default=["erm", "rand", "atlas", "hod", "factor_notone", "factor"])
    ap.add_argument("--clean", choices=["common", "direction"], default="common",
                    help="common: drop MCSI images that duplicate any study image (pre-registered primary set); "
                         "direction: drop only duplicates of the model's training dataset (amendment 1, secondary)")
    ap.add_argument("--suffix", default="", help="appended to the output file names")
    a = ap.parse_args()
    res = Path(a.results)
    info = pd.read_csv(a.external)
    rows = []
    for m in sorted(res.glob("*/*/*/metrics.json")):
        f = m.parent / f"predictions_{a.name}.csv"
        rec = json.loads(m.read_text())
        if rec.get("protocol") != "P3" or not f.exists():
            continue
        row = {"protocol": "P3", "direction": f"{rec['train_dataset']}->{a.name}", "model": rec["model"],
               "rep": rec["rep"], "method": rec["method"]}
        if a.clean == "common":
            keep = info[~info["duplicate_any"]] if "duplicate_any" in info else info
        else:
            col = f"dup_{rec['train_dataset']}"
            keep = info[~info[col]] if col in info else info
        pred = pd.read_csv(f)
        pred = pred[pred["image_id"].isin(set(keep["image_id"]))]
        row["n_mpox"] = int((keep.set_index("image_id").loc[pred["image_id"], "label"] == "mpox").sum())
        row.update(run_row(pred, keep))
        rows.append(row)
    runs = pd.DataFrame(rows)
    if runs.empty:
        raise SystemExit(f"no runs with predictions_{a.name}.csv under {res}")
    runs.to_csv(res / f"external_runs{a.suffix}.csv", index=False)

    base = info[~info["duplicate_any"]] if (a.clean == "common" and "duplicate_any" in info) else info
    counts = base["label"].value_counts().to_dict()
    what = ("pre-registered clean set" if a.clean == "common"
            else "direction-specific clean sets (amendment 1; counts below are before removing training-set copies)")
    text = [f"# External test on {NAMES.get(a.name, a.name)} ({what}: {len(base)} images, {counts}); "
            "mean ± SD over backbones and seeds", ""]
    meths = [x for x in a.order if x in set(runs["method"])] + sorted(set(runs["method"]) - set(a.order))
    extra = [c for c in ["factor_disease_auroc", "factor_mpox_auroc"] if c in runs]
    for d in sorted(runs["direction"].unique()):
        src = d.split("->")[0]
        sub = runs[runs["direction"] == d]
        nm = sorted(set(sub["n_mpox"]))
        text += [f"## Trained on {NAMES.get(src, src)} (mpox images scored: {nm})", "",
                 "| method | runs | " + " | ".join(COLS + extra) + " |", "| --- " * (len(COLS + extra) + 2) + "|"]
        for meth in meths:
            g = sub[sub["method"] == meth]
            if g.empty:
                continue
            cells = []
            for c in COLS + extra:
                x = g[c].dropna() if c in g else pd.Series(dtype=float)
                cells.append(f"{x.mean():.3f} ± {x.std(ddof=1):.3f}" if len(x) > 1
                             else (f"{x.mean():.3f}" if len(x) else "–"))
            text.append(f"| {meth} | {len(g)} | " + " | ".join(cells) + " |")
        text.append("")
    (res / f"external{a.suffix}.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
