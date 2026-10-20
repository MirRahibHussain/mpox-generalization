"""Step 14 (CPU) - blinded human rating of the Grad-CAM maps and inter-rater agreement.

The Grad-CAM file names written by step 7 start with the prediction outcome
(TP_, FP_, TN_, FN_), so a rater who sees them is not blind. This script

1. ``make``: copies every rated Grad-CAM image into ``<out>/images`` under a
   random name (img_001.png, ...), writes an empty ``rating_sheet.csv`` for each
   rater and a separate ``key.csv`` that maps the random names back to the runs.
   Give raters only the ``images`` folder, ``rating_sheet.csv`` and
   ``INSTRUCTIONS.txt``; keep ``key.csv`` away from them.
2. ``agree``: reads the filled sheets, computes percentage agreement and
   Cohen's kappa for every pair of raters (and against the ratings released in
   paper_results/gradcam_ratings.csv, if given), and rebuilds Table 8 for each
   rater.

    python scripts/14_gradcam_rating_kit.py make --results results \
        --ratings paper_results/gradcam_ratings.csv --out gradcam_blind
    python scripts/14_gradcam_rating_kit.py agree --key gradcam_blind/key.csv \
        --sheets rater_A.csv rater_B.csv --ai paper_results/gradcam_ratings.csv
"""

import argparse
import itertools
import shutil
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import pandas as pd

CATS = ["lesion", "mixed", "off-lesion", "no lesion"]
OUTCOMES = {"TP": "True positive (mpox)", "FP": "False positive", "TN": "True negative",
            "FN": "False negative (missed mpox)"}
INSTRUCTIONS = """Grad-CAM rating instructions
============================

Each image shows a skin photograph (left) and the same photograph with the
model's Grad-CAM heat map (right). For every image, look at the hottest (red)
region of the map and write ONE of these words in the 'attention' column of
rating_sheet.csv:

  lesion      the red region lies on a visible lesion or on lesion-bearing skin
  mixed       the red region is split between lesion skin and other content
  off-lesion  the red region lies on other content: background, clothing,
              hands or skin without lesions, eyes, hair, text, watermarks,
              censor bars or image edges
  no lesion   the photograph shows no visible lesion at all (healthy skin)

You may add a short reason in the 'note' column. Rate every image on your own,
without discussing it with the other rater, and do not try to guess whether
the model's prediction was right. Write your name or initials in the 'rater'
column. When you are done, save the file as CSV.
"""


def kappa(a, b, cats=CATS):
    a, b = np.asarray(a), np.asarray(b)
    po = float((a == b).mean())
    pe = float(sum((a == c).mean() * (b == c).mean() for c in cats))
    return po, (po - pe) / (1 - pe) if pe < 1 else np.nan


def norm(x):
    x = str(x).strip().lower().replace("_", "-")
    return {"off lesion": "off-lesion", "offlesion": "off-lesion", "none": "no lesion",
            "no-lesion": "no lesion", "healthy": "no lesion", "n/a (healthy)": "no lesion", "n/a": "no lesion"}.get(x, x)


def make(a):
    rat = pd.read_csv(a.ratings)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(rat))
    key = []
    for k, i in enumerate(order, 1):
        r = rat.iloc[i]
        src = Path(a.results) / r["run"] / "gradcam" / r["file"]
        if not src.exists():
            raise SystemExit(f"missing {src}")
        bid = f"img_{k:03d}"
        shutil.copy(src, out / "images" / f"{bid}.png")
        key.append({"blind_id": bid, "run": r["run"], "file": r["file"], "image_id": r["image_id"],
                    "kind": r["kind"], "label": r["label"], "ita_band": r["ita_band"]})
    pd.DataFrame(key).to_csv(out / "key.csv", index=False)
    pd.DataFrame({"blind_id": [k["blind_id"] for k in key], "attention": "", "note": "", "rater": ""}) \
        .to_csv(out / "rating_sheet.csv", index=False)
    (out / "INSTRUCTIONS.txt").write_text(INSTRUCTIONS)
    print(f"Wrote {len(key)} blinded images to {out}/images; give raters images/, rating_sheet.csv and "
          f"INSTRUCTIONS.txt, and keep key.csv apart.")


def agree(a):
    key = pd.read_csv(a.key)
    raters = {}
    for s in a.sheets:
        df = pd.read_csv(s)
        name = str(df["rater"].dropna().iloc[0]) if "rater" in df and df["rater"].notna().any() else Path(s).stem
        df["attention"] = df["attention"].map(norm)
        bad = sorted(set(df["attention"]) - set(CATS))
        if bad:
            raise SystemExit(f"{s}: unknown ratings {bad}; use one of {CATS}")
        raters[name] = df.set_index("blind_id")["attention"]
    if a.ai:
        ai = pd.read_csv(a.ai).merge(key[["blind_id", "run", "file"]], on=["run", "file"])
        raters["AI rater (released)"] = ai.set_index("blind_id")["attention"].map(norm)
    ids = sorted(set.intersection(*[set(v.index) for v in raters.values()]))
    lines = [f"## Grad-CAM inter-rater agreement ({len(ids)} images rated by all)", "",
             "| rater pair | images | agreement | Cohen's kappa | kappa without 'no lesion' images |",
             "| --- | --- | --- | --- | --- |"]
    for (n1, r1), (n2, r2) in itertools.combinations(raters.items(), 2):
        x, y = r1.loc[ids], r2.loc[ids]
        po, k = kappa(x, y)
        keep = (x != "no lesion") & (y != "no lesion")
        po2, k2 = kappa(x[keep], y[keep], CATS[:3])
        lines.append(f"| {n1} vs {n2} | {len(ids)} | {po * 100:.1f}% | {k:.2f} | {k2:.2f} (n = {int(keep.sum())}) |")
    lines.append("")
    kk = key.set_index("blind_id")
    for n, r in raters.items():
        t = pd.crosstab(kk.loc[ids, "kind"].map(OUTCOMES), r.loc[ids]).reindex(columns=CATS, fill_value=0)
        lines += [f"### Table 8 as rated by {n}", "", t.to_markdown(), ""]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--results", default="results")
    m.add_argument("--ratings", default="paper_results/gradcam_ratings.csv")
    m.add_argument("--out", default="gradcam_blind")
    m.add_argument("--seed", type=int, default=2026)
    g = sub.add_parser("agree")
    g.add_argument("--key", default="gradcam_blind/key.csv")
    g.add_argument("--sheets", nargs="+", required=True)
    g.add_argument("--ai", default=None)
    g.add_argument("--out", default="gradcam_agreement.md")
    a = ap.parse_args()
    make(a) if a.cmd == "make" else agree(a)


if __name__ == "__main__":
    main()
