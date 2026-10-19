"""Step 11 (CPU) - decision thresholds chosen on validation data, and ROC operating points.

The default decision rule (argmax, i.e. a threshold of 0.5 on the mpox
probability) mixes two things when methods are compared: how well a model ranks
images, and where its threshold happens to fall. For every P3 run that saved
validation predictions this script reports

* AUROC on the test set (threshold-free);
* test specificity at 90% sensitivity and test sensitivity at 90% specificity,
  read off the test ROC curve (an optimistic reading, because the threshold
  is fitted on test data);
* test sensitivity, specificity and balanced accuracy when the threshold is
  chosen on the validation set to give at least 90% mpox sensitivity there
  (a rule that could be used in practice);
* the same for the Youden threshold of the validation set.

It then runs paired Wilcoxon signed-rank tests within each direction for the
chosen method pairs, with Holm's correction within each (direction, metric).

    python scripts/11_operating_points.py --results results_control
"""

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

TARGET = 0.90
DIRS = {"msid->msld_v2": "MSID → MSLD v2.0", "msld_v2->msid": "MSLD v2.0 → MSID"}


def thr_for_sens(y, p, target=TARGET):
    """Largest threshold whose sensitivity (p >= t) is at least ``target``."""
    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        return np.nan
    k = int(np.ceil(target * len(pos)))
    return float(pos[max(k, 1) - 1])


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def rates(y, p, t):
    pred = p >= t
    sens = pred[y == 1].mean() if (y == 1).any() else np.nan
    spec = (~pred[y == 0]).mean() if (y == 0).any() else np.nan
    return float(sens), float(spec), float((sens + spec) / 2)


def roc_points(y, p, target=TARGET):
    fpr, tpr, _ = roc_curve(y, p)
    spec_at_sens = float((1 - fpr[tpr >= target]).max())
    sens_at_spec = float(tpr[fpr <= 1 - target].max())
    return spec_at_sens, sens_at_spec


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def wilcoxon(a, b):
    from scipy.stats import wilcoxon as w
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return float(w(a, b, zero_method="wilcox").pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_control")
    ap.add_argument("--pairs", nargs="*", default=["la:erm", "rand:erm", "tale:erm", "la:rand", "tale:rand", "tale:la"])
    a = ap.parse_args()
    res = Path(a.results)
    rows = []
    for m in sorted(res.glob("*/*/*/metrics.json")):
        rec = json.loads(m.read_text())
        if rec.get("protocol") != "P3" or not (m.parent / "val_predictions.csv").exists():
            continue
        te = pd.read_csv(m.parent / "predictions.csv")
        va = pd.read_csv(m.parent / "val_predictions.csv")
        yt, pt = te["y_true"].to_numpy(), te["p1"].to_numpy()
        yv, pv = va["y_true"].to_numpy(), va["p1"].to_numpy()
        t90, ty = thr_for_sens(yv, pv), youden(yv, pv)
        s90, sp90, ba90 = rates(yt, pt, t90)
        sy, spy, bay = rates(yt, pt, ty)
        spec_at, sens_at = roc_points(yt, pt)
        rows.append({
            "direction": f"{rec['train_dataset']}->{rec['test_dataset']}", "model": rec["model"],
            "rep": rec["rep"], "method": rec["method"],
            "auroc": roc_auc_score(yt, pt), "spec_at_sens90": spec_at, "sens_at_spec90": sens_at,
            "thr_val90": t90, "val_sens_val90": rates(yv, pv, t90)[0],
            "test_sens_val90": s90, "test_spec_val90": sp90, "test_bal_acc_val90": ba90,
            "thr_youden": ty, "test_sens_youden": sy, "test_spec_youden": spy, "test_bal_acc_youden": bay,
            "test_bal_acc_argmax": rec["balanced_accuracy"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no P3 runs with val_predictions.csv under {res}")
    df.to_csv(res / "operating_points_runs.csv", index=False)

    metrics = ["auroc", "spec_at_sens90", "sens_at_spec90", "test_sens_val90", "test_spec_val90",
               "test_bal_acc_val90", "test_bal_acc_youden"]
    summ = df.groupby(["direction", "method"])[metrics].agg(["mean", "std"])
    summ.to_csv(res / "operating_points_summary.csv")

    tests = []
    for d in df["direction"].unique():
        sub = df[df["direction"] == d]
        for metric in ["auroc", "test_bal_acc_val90", "test_spec_val90", "test_bal_acc_youden"]:
            piv = sub.pivot_table(index=["model", "rep"], columns="method", values=metric)
            fam = []
            for pr in a.pairs:
                x, y = pr.split(":")
                if x not in piv or y not in piv:
                    continue
                pp = piv[[x, y]].dropna()
                fam.append({"direction": d, "metric": metric, "comparison": f"{x} vs {y}", "pairs": len(pp),
                            "mean_diff": (pp[x] - pp[y]).mean(), "wins": int((pp[x] > pp[y]).sum()),
                            "p": wilcoxon(pp[x], pp[y])})
            if fam:
                adj = holm([f["p"] for f in fam])
                for f, q in zip(fam, adj):
                    f["p_holm"] = float(q)
                tests += fam
    tt = pd.DataFrame(tests)
    tt.to_csv(res / "operating_points_tests.csv", index=False)

    lines = ["## Operating points (P3), mean ± SD over backbones and seeds", ""]
    for d, dn in DIRS.items():
        sub = df[df["direction"] == d]
        if sub.empty:
            continue
        lines += [f"### {dn}", "", "| method | AUROC | spec. @ 90% sens. (test ROC) | sens. @ 90% spec. (test ROC) | "
                  "test sens. (val. thr. for 90% sens.) | test spec. (same thr.) | test bal. acc. (same thr.) | "
                  "test bal. acc. (val. Youden thr.) | runs |", "| --- " * 9 + "|"]
        for meth, g in sub.groupby("method"):
            f = lambda c: f"{g[c].mean():.3f} ± {g[c].std(ddof=1):.3f}"
            lines.append(f"| {meth} | {f('auroc')} | {f('spec_at_sens90')} | {f('sens_at_spec90')} | "
                         f"{f('test_sens_val90')} | {f('test_spec_val90')} | {f('test_bal_acc_val90')} | "
                         f"{f('test_bal_acc_youden')} | {len(g)} |")
        lines.append("")
    if not tt.empty:
        lines += ["## Paired Wilcoxon tests within each direction (Holm-adjusted within direction and metric)", "",
                  "| direction | metric | comparison | pairs | mean diff | wins | p | p (Holm) |", "| --- " * 8 + "|"]
        for r in tt.itertuples():
            lines.append(f"| {DIRS.get(r.direction, r.direction)} | {r.metric} | {r.comparison} | {r.pairs} | "
                         f"{r.mean_diff:+.3f} | {r.wins} | {r.p:.4f} | {r.p_holm:.4f} |")
    (res / "operating_points.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
