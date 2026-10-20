"""Step 16 (CPU) - how well does the ITA proxy track skin type, and does tone matching work?

The study uses an image-based Individual Typology Angle (ITA) as a skin-tone
proxy because the mpox datasets carry no skin-type labels. Fitzpatrick17k
images do carry a Fitzpatrick skin type (FST, 1-6, annotated by people), so the
atlas pool lets us check the proxy on about 16,000 images:

* Spearman correlation between ITA and FST (a valid proxy gives a clearly
  negative value: darker types, lower ITA);
* ITA distribution per FST;
* agreement between the three ITA bands and FST groups (1-2 light, 3-4
  intermediate, 5-6 dark): crosstab, accuracy, linearly weighted Cohen's kappa.

It also reports the ITA-band mix of the mpox training images of each dataset
and the expected band mix of the atlas negatives with condition-uniform draws
("cond") and with tone-matched draws ("cond_tone", FACTOR), which shows
whether tone matching does what it should.

    python scripts/16_ita_validation.py --pool data/pool_atlas/pool.csv --index data/index --out results_factor
"""

import argparse
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from mpoxgen.factor import atlas_weights

BANDS = ["light", "intermediate", "dark"]
FST_GROUP = {1: "light", 2: "light", 3: "intermediate", 4: "intermediate", 5: "dark", 6: "dark"}


def md(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/pool_atlas/pool.csv")
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--out", default="results_factor")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(a.pool)
    pool["fitzpatrick"] = pd.to_numeric(pool["fitzpatrick"], errors="coerce")
    ok = pool[pool["fitzpatrick"].between(1, 6) & pool["ita"].notna()].copy()
    ok["fitzpatrick"] = ok["fitzpatrick"].astype(int)
    text = ["# ITA proxy check on Fitzpatrick17k (atlas pool)", "",
            f"Images with both an FST label and an ITA estimate: {len(ok)} of {len(pool)} "
            f"(ITA could not be estimated for {int(pool['ita'].isna().sum())}; "
            f"FST missing for {int((~pool['fitzpatrick'].between(1, 6)).sum())}).", ""]

    rho, p = spearmanr(ok["ita"], ok["fitzpatrick"])
    rng = np.random.default_rng(0)
    x, y = ok["ita"].to_numpy(), ok["fitzpatrick"].to_numpy()
    boots = []
    for _ in range(a.n_boot):
        i = rng.integers(0, len(x), len(x))
        boots.append(spearmanr(x[i], y[i])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    text += [f"Spearman rho(ITA, FST) = {rho:.3f} (95% bootstrap CI {lo:.3f} to {hi:.3f}; p = {p:.2g}).", ""]

    per = ok.groupby("fitzpatrick")["ita"].describe(percentiles=[0.25, 0.5, 0.75])
    per = per[["count", "25%", "50%", "75%"]].round(1)
    per["count"] = per["count"].astype(int)
    per = per.reset_index()
    per.columns = ["FST", "n", "ITA Q1", "ITA median", "ITA Q3"]
    per.to_csv(out / "ita_by_fst.csv", index=False)
    text += ["## ITA by Fitzpatrick skin type", md(per), ""]

    ok["fst_group"] = ok["fitzpatrick"].map(FST_GROUP)
    known = ok[ok["ita_band"].isin(BANDS)]
    ct = pd.crosstab(known["fst_group"], known["ita_band"]).reindex(index=BANDS, columns=BANDS, fill_value=0)
    ct.to_csv(out / "ita_band_vs_fst_group.csv")
    code = {b: k for k, b in enumerate(BANDS)}
    kappa = cohen_kappa_score(known["fst_group"].map(code), known["ita_band"].map(code), weights="linear")
    acc = float((known["fst_group"] == known["ita_band"]).mean())
    far = float(((known["fst_group"] == "dark") & (known["ita_band"] == "light")).sum()
                + ((known["fst_group"] == "light") & (known["ita_band"] == "dark")).sum()) / len(known)
    text += ["## ITA band vs FST group (rows: FST 1-2 / 3-4 / 5-6; columns: ITA band)",
             md(ct.reset_index().rename(columns={"fst_group": "FST group"})), "",
             f"Agreement {acc:.3f}; linearly weighted kappa {kappa:.3f}; "
             f"light/dark swapped in {far:.3f} of images.", ""]

    # Band mix of mpox training images vs atlas draws.
    rows = []
    images = pd.read_csv(Path(a.index) / "images.csv")
    for ds, g in images.groupby("dataset"):
        mp = g.loc[g["label"] == "mpox", "ita_band"].fillna("unknown").to_numpy()
        if not len(mp):
            continue
        row = {"dataset": ds, "draw": "mpox images (target)"}
        row.update(pd.Series(mp).value_counts(normalize=True).reindex(BANDS + ["unknown"]).fillna(0).round(3))
        rows.append(row)
        for mode in ["cond", "cond_tone"]:
            w = atlas_weights(pool["condition"].to_numpy(), pool["ita_band"].fillna("unknown").to_numpy(),
                              mode=mode, mpox_bands=mp)
            mix = pd.Series(w).groupby(pool["ita_band"].fillna("unknown").to_numpy()).sum()
            row = {"dataset": ds, "draw": f"atlas, {mode}"}
            row.update(mix.reindex(BANDS + ["unknown"]).fillna(0).round(3))
            rows.append(row)
    mix = pd.DataFrame(rows)
    mix.to_csv(out / "atlas_tone_mix.csv", index=False)
    text += ["## ITA-band mix: mpox images vs expected atlas draw "
             "(all mpox images of the dataset; training splits are subsets)", md(mix), ""]
    (out / "ita_validation.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        data = [ok.loc[ok["fitzpatrick"] == f, "ita"].to_numpy() for f in range(1, 7)]
        ax.boxplot(data, showfliers=False, widths=0.55, medianprops={"color": "#2a78d6", "lw": 1.4},
                   boxprops={"lw": 0.7}, whiskerprops={"lw": 0.7}, capprops={"lw": 0.7})
        ax.set_xticks(range(1, 7), [str(f) for f in range(1, 7)])
        ax.axhline(41, color="#999", lw=0.5, ls="--")
        ax.axhline(10, color="#999", lw=0.5, ls="--")
        ax.set_xlabel("Fitzpatrick skin type (annotated)", fontsize=7.5)
        ax.set_ylabel("Image ITA (degrees)", fontsize=7.5)
        ax.tick_params(labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "fig_ita_vs_fst.png", dpi=400)
    except Exception as e:
        print(f"[warn] figure skipped: {e}")


if __name__ == "__main__":
    main()
