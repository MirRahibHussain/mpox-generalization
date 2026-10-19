"""Step 12 (CPU) - does the lesion-preserving tone shift do what it claims?

For a sample of study images (up to --per-group per dataset and ITA band), each
image is shifted once toward a random target ITA in every band. We record

* how often the shift is skipped, and why;
* the ITA of the healthy skin after the shift, against the target (absolute
  error and how often it lands in the target band);
* the median change of L* on lesion pixels and on healthy pixels;
* how often the wanted change hit the 35-unit cap;
* the share of image pixels that the YCrCb rule detects as skin, by the image's
  original ITA band (to see whether very dark skin is missed).

    python scripts/12_tone_shift_check.py --index data/index --out results_control
"""

import argparse
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.color import rgb2lab

from mpoxgen import ita
from mpoxgen.methods import BAND_RANGES, shift_tone


def ita_of(lab, mask):
    L, b = lab[..., 0][mask], lab[..., 2][mask]
    return float(np.degrees(np.arctan2(np.median(L) - 50.0, np.median(b))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--out", default="results_control")
    ap.add_argument("--per-group", type=int, default=60)
    ap.add_argument("--side", type=int, default=288)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    images = pd.read_csv(Path(a.index) / "images.csv")
    images = images[images["ita_band"].isin(["dark", "intermediate", "light"])]
    sample = pd.concat([g.sample(min(a.per_group, len(g)), random_state=a.seed)
                        for _, g in images.groupby(["dataset", "ita_band"])], ignore_index=True)
    rows = []
    for r in sample.itertuples():
        with Image.open(r.path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((a.side * 2, a.side * 2))
            w, h = im.size
            s = min(w, h)
            if s > a.side:
                im = im.resize((max(1, round(w * a.side / s)), max(1, round(h * a.side / s))), Image.Resampling.BICUBIC)
            rgb = np.asarray(im)
        lab_in = rgb2lab(rgb)
        for tb, (lo, hi) in BAND_RANGES.items():
            target = float(rng.uniform(lo, hi))
            res, info = shift_tone(im, target, return_info=True)
            row = {"image_id": r.image_id, "dataset": r.dataset, "label": r.label, "orig_band": r.ita_band,
                   "orig_ita": r.ita, "target_band": tb, "target_ita": target,
                   "skin_fraction": float(info["skin"].mean()) if "skin" in info else np.nan,
                   "skipped": res is None, "reason": info["reason"]}
            if res is not None:
                lab_out = rgb2lab(np.asarray(res))
                healthy = info["healthy"]
                lesion = info["skin"] & ~healthy
                achieved = ita_of(lab_out, healthy)
                dL = lab_out[..., 0] - lab_in[..., 0]
                row.update(achieved_ita=achieved, abs_error=abs(achieved - target),
                           achieved_band=ita.band(achieved), band_hit=ita.band(achieved) == tb,
                           dL_healthy=float(np.median(dL[healthy])),
                           abs_dL_lesion=float(np.median(np.abs(dL[lesion]))) if lesion.any() else np.nan,
                           delta=info["delta"], capped=abs(info["wanted"]) > 35.0,
                           lesion_fraction_of_skin=float(lesion.sum() / max(info["skin"].sum(), 1)))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "tone_shift_check.csv", index=False)

    ok = df[~df["skipped"]]
    lines = [
        "## Tone-shift check on study images",
        "",
        f"images: {df['image_id'].nunique()} ({', '.join(f'{k}: {v}' for k, v in df.groupby('dataset')['image_id'].nunique().items())}); "
        f"shifts attempted: {len(df)}",
        f"skipped: {df['skipped'].mean() * 100:.1f}% "
        f"({'; '.join(f'{k}: {v}' for k, v in df['reason'].value_counts().items())})",
        f"healthy-skin ITA after shift vs target: median absolute error {ok['abs_error'].median():.1f} deg; "
        f"lands in the target band {ok['band_hit'].mean() * 100:.1f}% of the time",
        f"wanted change hit the 35-unit cap: {ok['capped'].mean() * 100:.1f}%",
        f"median dL* healthy skin: {ok['dL_healthy'].abs().median():.1f}; median |dL*| lesion pixels: "
        f"{ok['abs_dL_lesion'].median():.1f}",
        "",
        "### Band hit rate (%), original band (rows) x target band (columns)",
        (ok.pivot_table(index="orig_band", columns="target_band", values="band_hit", aggfunc="mean") * 100)
        .round(1).to_string(),
        "",
        "### Skipped (%) by original band",
        (df.groupby("orig_band")["skipped"].mean() * 100).round(1).to_string(),
        "",
        "### Share of pixels detected as skin (median) by original band",
        df.drop_duplicates("image_id").groupby("orig_band")["skin_fraction"].median().round(3).to_string(),
        "",
        "### Median absolute ITA error (deg) by target band",
        ok.groupby("target_band")["abs_error"].median().round(1).to_string(),
    ]
    (out / "tone_shift_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
