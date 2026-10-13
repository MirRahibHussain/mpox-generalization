"""Step 6 - turn finished runs into the paper's tables and figure.

Outputs (in --results):
  summary_by_model.csv   mean +/- sd per task x model x protocol x direction
  inflation_gap.csv      inflation = P1 - P2, generalization gap = P2 - P3
  ita_breakdown.csv      mpox sensitivity/specificity per skin-tone band
  paper_tables.md        the same, as Markdown tables
  fig_protocols.png      balanced accuracy by protocol for each model
"""

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mpoxgen.labels import positive_index  # noqa: E402

METRICS = ["balanced_accuracy", "macro_f1", "auroc", "mpox_sensitivity", "mpox_specificity"]
# Validated categorical slots 1-3 (dataviz reference palette, light mode).
COLORS = {"P1": "#2a78d6", "P2": "#eb6834", "P3": "#1baf7a"}
LABELS = {"P1": "P1 naive split", "P2": "P2 patient-level", "P3": "P3 cross-dataset"}
INK, INK2 = "#0b0b0b", "#52514e"


def load(results: Path) -> pd.DataFrame:
    recs = [json.loads(p.read_text()) for p in results.glob("*/*/metrics.json")]
    if not recs:
        raise SystemExit(f"No finished runs under {results}/")
    df = pd.DataFrame(recs)
    df["direction"] = df["train_dataset"] + " -> " + df["test_dataset"]
    return df


def summary(df):
    g = df.groupby(["task", "model", "protocol", "direction"])
    s = g[METRICS].agg(["mean", "std"])
    s.columns = [f"{m}_{st}" for m, st in s.columns]
    s["n_runs"] = g.size()
    return s.reset_index()


def inflation_gap(s):
    rows = []
    key = "balanced_accuracy_mean"
    for (task, model), g in s.groupby(["task", "model"]):
        for ds in sorted({d.split(" -> ")[0] for d in g["direction"]}):
            within = f"{ds} -> {ds}"
            p1 = g[(g.protocol == "P1") & (g.direction == within)][key]
            p2 = g[(g.protocol == "P2") & (g.direction == within)][key]
            p3 = g[(g.protocol == "P3") & (g.direction.str.startswith(ds + " -> "))
                   & (g.direction != within)]
            for _, r3 in p3.iterrows() if len(p3) else [(None, None)]:
                rows.append({
                    "task": task, "model": model, "train_dataset": ds,
                    "P1_naive": p1.iloc[0] if len(p1) else np.nan,
                    "P2_patient": p2.iloc[0] if len(p2) else np.nan,
                    "P3_cross": r3[key] if r3 is not None else np.nan,
                    "P3_direction": r3["direction"] if r3 is not None else "",
                })
    out = pd.DataFrame(rows)
    out["inflation_P1_minus_P2"] = out["P1_naive"] - out["P2_patient"]
    out["gap_P2_minus_P3"] = out["P2_patient"] - out["P3_cross"]
    return out


def ita_breakdown(results, df, images):
    if "ita_band" not in images:
        return pd.DataFrame()
    band = images.set_index("image_id")["ita_band"]
    rows = []
    for r in df[df.protocol.isin(["P2", "P3"])].itertuples():
        pred = pd.read_csv(results / f"{r.task}_{r.protocol}_{r.train_dataset}_to_{r.test_dataset}_rep{r.rep}"
                           / r.model / "predictions.csv")
        pred["band"] = band.reindex(pred["image_id"]).to_numpy()
        pred["task"], pred["protocol"], pred["model"] = r.task, r.protocol, r.model
        pred["test_dataset"] = r.test_dataset
        rows.append(pred)
    if not rows:
        return pd.DataFrame()
    p = pd.concat(rows)
    out = []
    for (task, prot, model, tds, b), g in p.groupby(["task", "protocol", "model", "test_dataset", "band"]):
        pos = positive_index(task)
        is_pos, pred_pos = g.y_true == pos, g.y_pred == pos
        out.append({"task": task, "protocol": prot, "model": model, "test_dataset": tds, "ita_band": b,
                    "n_mpox": int(is_pos.sum()), "n_other": int((~is_pos).sum()),
                    "sensitivity": (pred_pos & is_pos).sum() / is_pos.sum() if is_pos.sum() else np.nan,
                    "specificity": (~pred_pos & ~is_pos).sum() / (~is_pos).sum() if (~is_pos).sum() else np.nan})
    return pd.DataFrame(out)


def figure(s, out_png, task="binary"):
    d = s[s.task == task]
    if d.empty:
        return
    train_sets = sorted({x.split(" -> ")[0] for x in d.direction})
    fig, axes = plt.subplots(1, len(train_sets), figsize=(6.5 * len(train_sets), 3.6), sharey=True,
                             squeeze=False)
    for ax, ds in zip(axes[0], train_sets):
        models = sorted(d.model.unique())
        x = np.arange(len(models))
        w = 0.26
        for k, prot in enumerate(["P1", "P2", "P3"]):
            if prot == "P3":
                sel = d[(d.protocol == prot) & d.direction.str.startswith(ds + " -> ")
                        & (d.direction != f"{ds} -> {ds}")]
            else:
                sel = d[(d.protocol == prot) & (d.direction == f"{ds} -> {ds}")]
            sel = sel.set_index("model").reindex(models)
            ax.bar(x + (k - 1) * w, sel["balanced_accuracy_mean"], w - 0.02,
                   yerr=sel["balanced_accuracy_std"], color=COLORS[prot], label=LABELS[prot],
                   error_kw={"elinewidth": 1, "ecolor": INK2, "capsize": 2})
        ax.set_xticks(x, models, rotation=20, ha="right", color=INK, fontsize=8)
        ax.set_title(f"Trained on {ds}", color=INK, fontsize=10, loc="left")
        ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("Balanced accuracy", color=INK)
    axes[0][0].text(-0.45, 0.515, "chance", color=INK2, fontsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_png, dpi=300)


def md(df, cols, floatfmt="{:.3f}"):
    head = "| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n"
    body = ""
    for r in df[cols].itertuples(index=False):
        body += "| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |\n"
    return head + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--index", default="data/index")
    a = ap.parse_args()
    res = Path(a.results)
    df = load(res)
    images = pd.read_csv(Path(a.index) / "images.csv")

    s = summary(df)
    s.to_csv(res / "summary_by_model.csv", index=False)
    ig = inflation_gap(s)
    ig.to_csv(res / "inflation_gap.csv", index=False)
    ita = ita_breakdown(res, df, images)
    ita.to_csv(res / "ita_breakdown.csv", index=False)
    figure(s, res / "fig_protocols.png")

    text = "# Results tables (generated by 06_summarize.py)\n\n## Balanced accuracy by protocol\n\n"
    text += md(s, ["task", "model", "protocol", "direction", "balanced_accuracy_mean",
                   "balanced_accuracy_std", "auroc_mean", "mpox_sensitivity_mean",
                   "mpox_specificity_mean", "n_runs"])
    text += "\n## Inflation and generalization gap (balanced accuracy)\n\n"
    text += md(ig, ["task", "model", "train_dataset", "P1_naive", "P2_patient", "P3_cross",
                    "P3_direction", "inflation_P1_minus_P2", "gap_P2_minus_P3"])
    if len(ita):
        text += "\n## Skin-tone (ITA band) breakdown, pooled over reps\n\n"
        text += md(ita, ["task", "protocol", "model", "test_dataset", "ita_band", "n_mpox",
                         "n_other", "sensitivity", "specificity"])
    (res / "paper_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
