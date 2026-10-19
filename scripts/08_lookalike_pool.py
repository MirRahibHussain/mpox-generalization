"""Step 8 - build the look-alike pool from Fitzpatrick17k and check it.

1. Sample up to --per-condition images of each look-alike condition
   (mpoxgen.external.LOOKALIKE_CONDITIONS), balancing Fitzpatrick types.
2. Remove any pool image that duplicates an MSLD v2.0 or MSID image
   (pHash/dHash, mirrored copies included, plus ResNet-50 embedding cosine
   >= --cos), and duplicates inside the pool.
3. Estimate ITA for every pool image.
4. Draw two contact sheets for checking by eye: pool examples, and the
   tone-shift augmentation applied to study images.
5. With --control-out, also build the control pool for the "rand" method: the
   same number of images with the same Fitzpatrick-type mix, from conditions
   that are not look-alikes, checked for duplicates in the same way.

    python scripts/08_lookalike_pool.py --fitz /kaggle/input/fitzpatrick17k-original \
        --csv fitzpatrick17k.csv --index data/index --out data/pool
"""

import argparse
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from mpoxgen import dedup, external, ita
from mpoxgen.methods import BAND_RANGES, shift_tone


def embed(paths, model_name="resnet50"):
    import timm
    import torch
    from timm.data import create_transform, resolve_data_config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
    tf = create_transform(**resolve_data_config({}, model=model))
    out = []
    with torch.no_grad():
        for k in range(0, len(paths), 64):
            xs = []
            for p in paths[k:k + 64]:
                with Image.open(p) as im:
                    xs.append(tf(ImageOps.exif_transpose(im).convert("RGB")))
            f = model(torch.stack(xs).to(device)).float()
            out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(out)


def thumb(path_or_img, side=150):
    im = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return ImageOps.fit(im, (side, side))


def sheet(cells, cols, out_png, side=150, label_h=16):
    rows = (len(cells) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * side, rows * (side + label_h)), "white")
    d = ImageDraw.Draw(canvas)
    for k, (img, text) in enumerate(cells):
        x, y = (k % cols) * side, (k // cols) * (side + label_h)
        canvas.paste(thumb(img, side), (x, y + label_h))
        d.text((x + 3, y + 2), text[:24], fill="black")
    canvas.save(out_png)


def remove_duplicates(pool, main_img, a, emb_file, f_main=None):
    """Drop pool images that duplicate a study image (or each other)."""
    both = pd.concat([main_img[["image_id", "dataset", "path", "label"]],
                      pool[["image_id", "dataset", "path", "label"]]], ignore_index=True)
    hashes = dedup.compute_hashes(both, verbose=False)
    pairs = dedup.hash_pairs(both, hashes, a.max_dist)
    is_pool = pairs["a"].str.startswith("fitz17k:") | pairs["b"].str.startswith("fitz17k:")
    pairs = pairs[is_pool]
    cross = pairs[pairs["dataset_a"] != pairs["dataset_b"]]
    drop = set(cross["a"]) | set(cross["b"])
    within = pairs[(pairs["dataset_a"] == "fitz17k") & (pairs["dataset_b"] == "fitz17k")]
    drop |= set(within["b"])  # keep the first image of each within-pool pair
    n_hash = int(pool["image_id"].isin(drop).sum())
    n_emb = 0
    try:  # embeddings are a second line of defence; hashes already ran
        f_pool = embed(pool["path"].tolist())
        if f_main is None:
            f_main = np.load(emb_file) if emb_file.exists() else embed(main_img["path"].tolist())
        hit = (f_pool @ f_main.T).max(1) >= a.cos
        n_emb = int(hit.sum())
        drop |= set(pool.loc[hit, "image_id"])
    except Exception as e:
        print(f"[warn] embedding check skipped: {e}")
    return pool[~pool["image_id"].isin(drop)].reset_index(drop=True), n_hash, n_emb, f_main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fitz", required=True, help="Fitzpatrick17k image folder (md5-named jpgs)")
    ap.add_argument("--csv", required=True, help="fitzpatrick17k.csv")
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--out", default="data/pool")
    ap.add_argument("--per-condition", type=int, default=50)
    ap.add_argument("--max-dist", type=int, default=6)
    ap.add_argument("--cos", type=float, default=0.92)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control-out", default="", help="also build the matched control pool here")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    main_img = pd.read_csv(Path(a.index) / "images.csv")

    pool = external.build_pool(a.fitz, a.csv, a.per_condition, seed=a.seed)
    n0 = len(pool)
    print(f"Sampled {n0} look-alike images from {pool['condition'].nunique()} conditions")

    f_main = None
    emb_file = Path(a.index) / "embeddings.npy"
    pool, n_hash_cross, n_emb, f_main = remove_duplicates(pool, main_img, a, emb_file, f_main)
    est = [ita.estimate(p) for p in pool["path"]]
    pool["ita"] = [e["ita"] for e in est]
    pool["ita_band"] = [e["ita_band"] for e in est]
    pool.to_csv(out / "pool.csv", index=False)

    report = [
        f"sampled: {n0} images, {pool['condition'].nunique()} conditions, up to {a.per_condition} each",
        f"removed as hash duplicates of study images or of each other: {n_hash_cross}",
        f"removed by embedding cosine >= {a.cos} with a study image: {n_emb}",
        f"final pool: {len(pool)} images",
        "", "by condition:", pool["condition"].value_counts().to_string(),
        "", "by Fitzpatrick type:", pool["fitzpatrick"].value_counts().sort_index().to_string(),
        "", "by ITA band:", pool["ita_band"].value_counts().to_string(),
    ]
    (out / "pool_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))

    ex = pool.groupby("condition").head(1).head(48)
    sheet([(p, c) for p, c in zip(ex["path"], ex["condition"])], 8, out / "pool_examples.png")

    # Tone-shift examples on study images: original + one target per band.
    rng = np.random.default_rng(0)
    cells = []
    for band in ["dark", "intermediate", "light"]:
        cand = main_img[(main_img["ita_band"] == band) & (main_img["label"] == "mpox")]
        for p in cand.sample(min(3, len(cand)), random_state=0)["path"]:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((384, 384))
                cells.append((im, f"original ({band})"))
                for tb, (lo, hi) in BAND_RANGES.items():
                    t = float(rng.uniform(lo, hi))
                    s = shift_tone(im, t)
                    cells.append((s if s is not None else im, f"-> {tb} ({t:.0f})" if s is not None else "skipped"))
    sheet(cells, 4, out / "tone_shift_examples.png", side=170)

    if a.control_out:
        cout = Path(a.control_out)
        cout.mkdir(parents=True, exist_ok=True)
        ctl = external.build_control_pool(a.fitz, a.csv, pool, a.per_condition, seed=a.seed + 1)
        c0 = len(ctl)
        ctl, c_hash, c_emb, f_main = remove_duplicates(ctl, main_img, a, emb_file, f_main)
        overlap = set(ctl["md5"]) & set(pool["md5"])
        ctl = ctl[~ctl["md5"].isin(overlap)]
        ctl = external.trim_to_match(ctl, pool)
        est = [ita.estimate(p) for p in ctl["path"]]
        ctl["ita"] = [e["ita"] for e in est]
        ctl["ita_band"] = [e["ita_band"] for e in est]
        ctl.drop(columns=["rank"]).to_csv(cout / "pool.csv", index=False)
        fz = pd.DataFrame({"look-alike pool": pool["fitzpatrick"].value_counts(),
                           "control pool": ctl["fitzpatrick"].value_counts()}).fillna(0).astype(int).sort_index()
        crep = [
            f"control candidates drawn: {c0} (target {len(pool)} plus 5% margin)",
            f"removed as duplicates of study images or of each other: {c_hash} (hash), {c_emb} (embedding)",
            f"final control pool: {len(ctl)} images from {ctl['condition'].nunique()} conditions",
            "", "Fitzpatrick type, look-alike vs control:", fz.to_string(),
            "", "control by ITA band:", ctl["ita_band"].value_counts().to_string(),
            "", "control conditions (top 40):", ctl["condition"].value_counts().head(40).to_string(),
        ]
        (cout / "pool_report.txt").write_text("\n".join(crep) + "\n")
        print("\n".join(crep))
        ex = ctl.groupby("condition").head(1).head(48)
        sheet([(p, c) for p, c in zip(ex["path"], ex["condition"])], 8, cout / "pool_examples.png")


if __name__ == "__main__":
    main()
