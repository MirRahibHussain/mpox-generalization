"""Step 10 (CPU) - tables, statistics and a figure for the method comparison.

Reads results_methods/*/*/*/{metrics.json,predictions.csv} and writes
results_methods/methods_tables.md plus CSV files and fig_methods.png.
"""

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import pandas as pd

NAMES = {"erm": "ERM (standard fine-tuning)", "hsv": "Colour jitter (hue/saturation)",
         "mixup": "Mixup", "gdro": "Group DRO", "la": "Look-alike exposure only",
         "tone": "Tone balancing + tone shift only", "tale": "TALE (proposed)"}
ORDER = ["erm", "hsv", "mixup", "gdro", "la", "tone", "tale"]
DIRS = {"msid->msld_v2": "MSID → MSLD v2.0", "msld_v2->msid": "MSLD v2.0 → MSID"}


def ms(x, d=3):
    x = pd.Series(x).dropna()
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else (f"{x.mean():.{d}f}" if len(x) else "")


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(map(str, r)) + " |")
    return "\n".join(lines)


def load(res: Path):
    recs = []
    for m in res.glob("*/*/*/metrics.json"):
        r = json.loads(m.read_text())
        r["run_dir"] = str(m.parent)
        recs.append(r)
    df = pd.DataFrame(recs)
    df["direction"] = df["train_dataset"] + "->" + df["test_dataset"]
    return df


def per_band(runs, images, metric):
    """Sensitivity (mpox) or specificity (non-mpox) per ITA band, one row per run."""
    band = images.set_index("image_id")["ita_band"]
    rows = []
    for r in runs.itertuples():
        p = pd.read_csv(Path(r.run_dir) / "predictions.csv")
        p["band"] = p["image_id"].map(band).fillna("unknown")
        sel = p[p["y_true"] == 1] if metric == "sens" else p[p["y_true"] == 0]
        out = {"method": r.method, "model": r.model, "rep": r.rep}
        for b in ["dark", "intermediate", "light"]:
            s = sel[sel["band"] == b]
            hit = (s["y_pred"] == 1) if metric == "sens" else (s["y_pred"] == 0)
            out[b] = hit.mean() if len(s) else np.nan
        out["gap"] = out["dark"] - out["light"]
        out["worst"] = min(out["dark"], out["intermediate"], out["light"])
        rows.append(out)
    return pd.DataFrame(rows)


def called_mpox(runs, images):
    lab = images.set_index("image_id")["label"]
    rows = []
    for r in runs.itertuples():
        p = pd.read_csv(Path(r.run_dir) / "predictions.csv")
        p["label"] = p["image_id"].map(lab)
        for l, g in p.groupby("label"):
            rows.append({"method": r.method, "label": l, "called": (g["y_pred"] == 1).mean(), "n": len(g)})
    return pd.DataFrame(rows)


def wilcoxon(a, b):
    from scipy.stats import wilcoxon as w
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return float(w(a, b, zero_method="wilcox").pvalue)


def figure(p3, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"msid->msld_v2": "#2a78d6", "msld_v2->msid": "#1baf7a"}
    meths = [m for m in ORDER if m in set(p3["method"])]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    x = np.arange(len(meths))
    for k, (d, c) in enumerate(cols.items()):
        g = p3[p3["direction"] == d].groupby("method")["balanced_accuracy"]
        mu, sd = g.mean().reindex(meths), g.std(ddof=1).reindex(meths)
        ax.bar(x + (k - 0.5) * 0.38, mu, 0.36, yerr=sd, color=c, label=DIRS[d],
               error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": "#555"})
    ax.set_xticks(x, [NAMES[m].split(" (")[0].replace("Tone balancing + tone shift only", "Tone only")
                      .replace("Look-alike exposure only", "Look-alike only") for m in meths],
                  rotation=20, ha="right", fontsize=7)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Balanced accuracy (P3)", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="#e6e5e0", lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_methods")
    ap.add_argument("--index", default="data/index")
    a = ap.parse_args()
    res = Path(a.results)
    images = pd.read_csv(Path(a.index) / "images.csv")
    df = load(res)
    df.to_csv(res / "all_runs.csv", index=False)
    p3 = df[df["protocol"] == "P3"]
    p2 = df[df["protocol"] == "P2"]
    meths = [m for m in ORDER if m in set(p3["method"])]
    text = []

    # A. main P3 table
    rows = []
    for m in meths:
        row = {"Method": NAMES[m]}
        for d, dn in DIRS.items():
            g = p3[(p3["method"] == m) & (p3["direction"] == d)]
            row[f"{dn}: bal. acc."] = ms(g["balanced_accuracy"])
            row[f"{dn}: spec."] = ms(g["mpox_specificity"])
            row[f"{dn}: sens."] = ms(g["mpox_sensitivity"])
        g = p3[p3["method"] == m]
        row["Mean bal. acc. (both)"] = ms(g["balanced_accuracy"])
        row["n runs"] = len(g)
        rows.append(row)
    tA = pd.DataFrame(rows)
    tA.to_csv(res / "table_p3_methods.csv", index=False)
    text += ["## P3: cross-dataset, mean ± SD over models and seeds", md_table(tA), ""]

    # B. per-backbone
    pb = p3.groupby(["method", "model"])["balanced_accuracy"].mean().unstack().reindex(meths)
    pb.to_csv(res / "table_p3_by_model.csv")
    text += ["## P3 balanced accuracy by backbone (mean over directions and seeds)",
             md_table(pb.round(3).reset_index()), ""]

    # C. skin tone
    sens = per_band(p3[p3["direction"] == "msld_v2->msid"], images, "sens")
    spec = per_band(p3[p3["direction"] == "msid->msld_v2"], images, "spec")
    sens.to_csv(res / "tone_sensitivity_runs.csv", index=False)
    spec.to_csv(res / "tone_specificity_runs.csv", index=False)
    rows = []
    for m in meths:
        s, c = sens[sens["method"] == m], spec[spec["method"] == m]
        rows.append({"Method": NAMES[m], "Sens. dark": f"{s['dark'].mean():.3f}",
                     "Sens. light": f"{s['light'].mean():.3f}", "Sens. gap (dark-light)": f"{s['gap'].mean():+.3f}",
                     "Worst-band sens.": f"{s['worst'].mean():.3f}",
                     "Spec. dark": f"{c['dark'].mean():.3f}", "Spec. light": f"{c['light'].mean():.3f}",
                     "Spec. gap (dark-light)": f"{c['gap'].mean():+.3f}"})
    tC = pd.DataFrame(rows)
    tC.to_csv(res / "table_tone_methods.csv", index=False)
    text += ["## Skin tone (sensitivity: MSLD v2.0 → MSID; specificity: MSID → MSLD v2.0)", md_table(tC), ""]

    # D. which images are called mpox (MSID -> MSLD v2.0)
    cm = called_mpox(p3[p3["direction"] == "msid->msld_v2"], images)
    tD = (cm.groupby(["method", "label"])["called"].mean().unstack().reindex(meths) * 100).round(1)
    tD.to_csv(res / "table_called_mpox_methods.csv")
    text += ["## % of MSLD v2.0 images called mpox by MSID-trained models",
             md_table(tD.reset_index()), ""]

    # E. P2
    if len(p2):
        rows = []
        for m in [x for x in ORDER if x in set(p2["method"])]:
            row = {"Method": NAMES[m]}
            for ds in ["msid", "msld_v2"]:
                g = p2[(p2["method"] == m) & (p2["train_dataset"] == ds)]
                row[f"{ds} P2 bal. acc."] = ms(g["balanced_accuracy"])
            rows.append(row)
        tE = pd.DataFrame(rows)
        tE.to_csv(res / "table_p2_methods.csv", index=False)
        text += ["## P2: patient-level, mean ± SD over models and folds", md_table(tE), ""]

    # F. paired tests on P3 balanced accuracy
    key = ["direction", "model", "rep"]
    piv = p3.pivot_table(index=key, columns="method", values="balanced_accuracy")
    rows = []
    for ref in ["tale", "la", "tone"]:
        if ref not in piv:
            continue
        for m in meths:
            if m == ref or (ref != "tale" and m != "erm"):
                continue
            pr = piv[[ref, m]].dropna()
            rows.append({"Comparison": f"{NAMES[ref]} vs {NAMES[m]}", "pairs": len(pr),
                         "mean diff": f"{(pr[ref] - pr[m]).mean():+.3f}",
                         "wins": int((pr[ref] > pr[m]).sum()),
                         "Wilcoxon p": f"{wilcoxon(pr[ref], pr[m]):.4f}"})
    tF = pd.DataFrame(rows)
    tF.to_csv(res / "table_tests.csv", index=False)
    text += ["## Paired Wilcoxon signed-rank tests (P3 balanced accuracy)", md_table(tF), ""]

    (res / "methods_tables.md").write_text("\n".join(text))
    figure(p3, res / "fig_methods.png")
    print("\n".join(text))


if __name__ == "__main__":
    main()
