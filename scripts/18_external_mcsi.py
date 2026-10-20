"""Step 18 - a third public dataset, MCSI, as an external test set, with a crop-aware duplicate audit.

MCSI (Mpox Close Skin Images; Campana et al., Pervasive Mob. Comput. 2024;
Zenodo record 8360076, CC BY 4.0) has 400 images: 100 mpox, 100 chickenpox,
100 acne (from ACNE04) and 100 healthy. Its mpox and healthy images come from
the first MSLD release, which MSLD v2.0 extends, and its images were cropped and
zoomed onto the lesion. Perceptual hashes do not survive cropping, so besides
pHash/dHash and embedding similarity every MCSI image is compared with its 10
nearest study images (ResNet-50 embedding) by ORB keypoint matching with a
RANSAC homography, which finds crops, zooms and re-encodings of the same photo.

An MCSI image is a duplicate of a study image if any of these holds:
pHash and dHash within --max-dist bits (mirrored copies included), embedding
cosine >= --cos, or at least --min-inliers RANSAC inliers. Only images that
duplicate no MSLD v2.0 or MSID image form the clean external test set, so no
model is tested on a copy of an image it could have been trained on, whatever
its training dataset.

Writes <out>/images.csv (all MCSI images with labels, duplicate flags and ITA),
images_clean.csv (the external test set), pairs.csv, report.txt and
duplicates_review.png (every flagged pair and the borderline ORB matches, for
checking by eye).

    python scripts/18_external_mcsi.py --index data/index --out data/external_mcsi
"""

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from mpoxgen import dedup, ita
from mpoxgen.hashing import hamming_matrix
from mpoxgen.indexing import md5_of

URL = "https://zenodo.org/records/8360076/files/MCSI.zip?download=1"
MD5 = "e8e490856d0a7e4871ae9e2dc15e81fe"
LABELS = {"monkeypox": "mpox", "chickenpox": "chickenpox", "normal": "healthy",
          "acne_l1": "acne", "acne_l2": "acne", "acne_l3": "acne"}


def fetch(dest: Path) -> Path:
    zp = dest / "MCSI.zip"
    if not zp.exists():
        dest.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}", flush=True)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(URL, timeout=120) as r:
                    zp.write_bytes(r.read())
                break
            except Exception as e:  # network hiccup
                print(f"  attempt {attempt + 1} failed: {e}")
        if not zp.exists():
            raise SystemExit("could not download MCSI")
    md5 = hashlib.md5(zp.read_bytes()).hexdigest()
    if md5 != MD5:
        raise SystemExit(f"MCSI.zip md5 {md5} != {MD5}")
    root = dest / "MCSI"
    if not (root / "metadata.csv").exists():
        with zipfile.ZipFile(zp) as z:
            z.extractall(root)
    return root


def load_mcsi(root: Path) -> pd.DataFrame:
    meta = pd.read_csv(root / "metadata.csv")
    meta.columns = [c.strip() for c in meta.columns]
    rows = []
    for r in meta.itertuples():
        p = root / "images" / str(r.img_id).strip()
        if not p.exists():
            continue
        diag = str(r.diagnostic).strip()
        rows.append({"dataset": "mcsi", "path": str(p), "file": p.name, "label": LABELS.get(diag, diag),
                     "mcsi_class": diag, "source": str(r.source).strip(), "md5": md5_of(p)})
    df = pd.DataFrame(rows)
    df["image_id"] = "mcsi:" + df["md5"].str[:12]
    tone_csv = root / "skin_tone_labels.csv"
    if tone_csv.exists():
        t = pd.read_csv(tone_csv)
        t.columns = [c.strip() for c in t.columns]
        t = t.rename(columns={"Filename": "file", "ITA": "mcsi_ita", "Type": "mcsi_tone", "CLS": "mcsi_ita_class"})
        df = df.merge(t[["file", "mcsi_ita", "mcsi_tone", "mcsi_ita_class"]], on="file", how="left")
    n0 = len(df)
    df = df.drop_duplicates("md5").reset_index(drop=True)
    print(f"MCSI: {n0} images listed and found, {n0 - len(df)} exact duplicates inside MCSI dropped", flush=True)
    return df


def _gray(path, side=512):
    import cv2
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        im.thumbnail((side, side))
        return cv2.equalizeHist(np.asarray(im))


def orb_inliers(path_a, path_b, orb, matcher):
    """RANSAC homography inliers between ORB matches of two images (0 if too few)."""
    import cv2
    ga, gb = _gray(path_a), _gray(path_b)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    good = []
    for m in matcher.knnMatch(da, db, k=2):
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            good.append(m[0])
    if len(good) < 8:
        return 0
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def thumb(path, side=160):
    with Image.open(path) as im:
        return ImageOps.fit(ImageOps.exif_transpose(im).convert("RGB"), (side, side))


def review_sheet(pairs: pd.DataFrame, ext: pd.DataFrame, study: pd.DataFrame, out_png: Path, side=160):
    pairs = pairs.head(80)
    if pairs.empty:
        Image.new("RGB", (400, 40), "white").save(out_png)
        return
    ep, sp = ext.set_index("image_id")["path"], study.set_index("image_id")["path"]
    cols = 4  # pairs per row
    rows = (len(pairs) + cols - 1) // cols
    W, H = cols * (2 * side + 12), rows * (side + 18)
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    for k, r in enumerate(pairs.itertuples()):
        x, y = (k % cols) * (2 * side + 12), (k // cols) * (side + 18)
        canvas.paste(thumb(ep[r.mcsi_id], side), (x, y + 16))
        canvas.paste(thumb(sp[r.study_id], side), (x + side, y + 16))
        d.text((x + 2, y + 2), f"{r.reason} cos={r.cos:.2f} inl={r.inliers}"[:44], fill="black")
    canvas.save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--out", default="data/external_mcsi")
    ap.add_argument("--root", default="", help="already extracted MCSI folder (skips the download)")
    ap.add_argument("--max-dist", type=int, default=6)
    ap.add_argument("--cos", type=float, default=0.92)
    ap.add_argument("--min-inliers", type=int, default=25)
    ap.add_argument("--neighbours", type=int, default=10)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.root) if a.root else fetch(out)
    ext = load_mcsi(root)
    study = pd.read_csv(Path(a.index) / "images.csv")

    # 1. perceptual hashes (direct and mirrored)
    hs = dedup.compute_hashes(study, verbose=False)
    he = dedup.compute_hashes(ext, verbose=False)
    direct = np.maximum(hamming_matrix(he["phash"], hs["phash"]), hamming_matrix(he["dhash"], hs["dhash"]))
    mirror = np.maximum(hamming_matrix(he["phash_flip"], hs["phash"]), hamming_matrix(he["dhash_flip"], hs["dhash"]))
    hdist = np.minimum(direct, mirror)

    # 2. embeddings (ResNet-50, as in the duplicate audit) and 3. ORB + RANSAC on nearest neighbours
    cos = np.zeros((len(ext), len(study)), np.float32)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("lp", Path(__file__).with_name("08_lookalike_pool.py"))
        lp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lp)
        emb_file = Path(a.index) / "embeddings.npy"
        f_s = np.load(emb_file) if emb_file.exists() else lp.embed(study["path"].tolist())
        f_e = lp.embed(ext["path"].tolist())
        cos = f_e @ f_s.T
    except Exception as e:
        print(f"[warn] embedding check skipped: {e}")
    inl = {}
    try:
        import cv2
        orb = cv2.ORB_create(nfeatures=1500)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        nn = np.argsort(-cos, axis=1)[:, :a.neighbours]
        hn = np.argsort(hdist, axis=1)[:, :3]  # also the closest hash matches
        for i in range(len(ext)):
            for j in set(nn[i].tolist()) | set(hn[i].tolist()):
                inl[(i, j)] = orb_inliers(ext["path"][i], study["path"][j], orb, matcher)
            if (i + 1) % 100 == 0:
                print(f"  ORB checked {i + 1}/{len(ext)}", flush=True)
    except Exception as e:
        print(f"[warn] ORB check skipped: {e}")

    rows = []
    for i in range(len(ext)):
        cand = set(np.nonzero(hdist[i] <= a.max_dist)[0]) | set(np.nonzero(cos[i] >= a.cos)[0])
        cand |= {j for (ii, j), v in inl.items() if ii == i and v >= 10}
        for j in cand:
            v = inl.get((i, j), -1)
            reason = [n for n, ok in (("hash", hdist[i, j] <= a.max_dist), ("emb", cos[i, j] >= a.cos),
                                      ("orb", v >= a.min_inliers)) if ok]
            rows.append({"mcsi_id": ext["image_id"][i], "mcsi_label": ext["label"][i], "study_id": study["image_id"][j],
                         "study_dataset": study["dataset"][j], "study_label": study["label"][j],
                         "hash_dist": int(hdist[i, j]), "cos": float(cos[i, j]), "inliers": int(v),
                         "duplicate": bool(reason), "reason": "+".join(reason) or "borderline"})
    pairs = pd.DataFrame(rows, columns=["mcsi_id", "mcsi_label", "study_id", "study_dataset", "study_label",
                                        "hash_dist", "cos", "inliers", "duplicate", "reason"])
    pairs = pairs.sort_values(["duplicate", "inliers", "cos"], ascending=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    dup = pairs[pairs["duplicate"]]
    for ds in sorted(study["dataset"].unique()):
        ext[f"dup_{ds}"] = ext["image_id"].isin(set(dup.loc[dup["study_dataset"] == ds, "mcsi_id"]))
    ext["duplicate_any"] = ext["image_id"].isin(set(dup["mcsi_id"]))

    est = [ita.estimate(p) for p in ext["path"]]
    ext["ita"] = [e["ita"] for e in est]
    ext["ita_band"] = [e["ita_band"] for e in est]
    ext.to_csv(out / "images.csv", index=False)
    clean = ext[~ext["duplicate_any"]].reset_index(drop=True)
    clean.to_csv(out / "images_clean.csv", index=False)
    review_sheet(pairs[pairs["duplicate"] | (pairs["inliers"] >= 10)], ext, study, out / "duplicates_review.png")

    tab = ext.groupby("label").agg(images=("image_id", "size"), dup_msld=("dup_msld_v2", "sum"),
                                   dup_msid=("dup_msid", "sum"), duplicate_any=("duplicate_any", "sum"))
    tab["clean"] = tab["images"] - tab["duplicate_any"]
    rho = ext[["ita", "mcsi_ita"]].corr(method="spearman").iloc[0, 1] if "mcsi_ita" in ext else np.nan
    report = [
        f"MCSI images: {len(ext)}; duplicates of a study image: {int(ext['duplicate_any'].sum())} "
        f"(MSLD v2.0 {int(ext.get('dup_msld_v2', pd.Series(dtype=bool)).sum())}, MSID {int(ext.get('dup_msid', pd.Series(dtype=bool)).sum())})",
        f"flagged pairs by reason: {dup['reason'].value_counts().to_dict()}",
        f"borderline ORB pairs (10 <= inliers < {a.min_inliers}, not flagged otherwise): "
        f"{int(((pairs['reason'] == 'borderline')).sum())}",
        f"clean external test set: {len(clean)} images", "", tab.to_string(), "",
        f"clean set by ITA band: {clean['ita_band'].value_counts().to_dict()}",
        f"Spearman rho between our ITA and the ITA shipped with MCSI: {rho:.3f}",
    ]
    (out / "report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
