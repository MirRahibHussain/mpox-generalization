"""CPU-only checks of the non-training pipeline on synthetic images.

Builds two fake datasets with known duplicates, then runs steps 1, 3, 4 and
the summary step (on random fake predictions) and asserts the expected
behaviour. Nothing here is a research result.

    python tests/run_cpu_checks.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mpoxgen import ita, metrics  # noqa: E402
from mpoxgen.labels import canonical_class  # noqa: E402

RNG = np.random.default_rng(0)


def fake_image(seed: int, tone: int) -> Image.Image:
    """Skin-coloured background with a few random lesion blobs."""
    r = np.random.default_rng(seed)
    base = np.array([[225, 185, 160], [190, 140, 110], [120, 80, 60]][tone], dtype=float)
    img = np.ones((180, 220, 3)) * base + r.normal(0, 6, (180, 220, 3))
    yy, xx = np.mgrid[:180, :220]
    for _ in range(r.integers(3, 8)):
        cy, cx, rad = r.integers(20, 160), r.integers(20, 200), r.integers(6, 18)
        m = (yy - cy) ** 2 + (xx - cx) ** 2 < rad ** 2
        img[m] = [170 + r.integers(0, 40), 60, 60]
    # Large random structure so different seeds give clearly different hashes.
    img += np.kron(r.normal(0, 25, (6, 6, 1)), np.ones((30, 37, 3)))[:180, :220]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def run(cmd, cwd):
    print("$", " ".join(cmd))
    res = subprocess.run([sys.executable] + cmd, cwd=cwd, capture_output=True, text=True)
    print(res.stdout[-2500:])
    if res.returncode != 0:
        print(res.stderr[-4000:])
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return res.stdout


def build_fake(base: Path):
    msld = base / "msld_v2"
    classes = ["Mpox", "Chickenpox", "Measles", "Cowpox", "HFMD", "Healthy"]
    per_class = 14
    files = []
    for c in classes:
        for p in range(per_class):
            files.append((c, f"{c[:3].upper()}_{p:03d}_01", RNG.integers(1e9), int(p % 3)))
    # Two official folds; every image is in each fold exactly once.
    order = RNG.permutation(len(files))
    for fold in (1, 2):
        perm = np.roll(order, fold * 17)
        for k, i in enumerate(perm):
            c, stem, seed, tone = files[i]
            split = "Test" if k < len(perm) * 0.1 else ("Valid" if k < len(perm) * 0.3 else "Train")
            d = msld / "Original Images" / "FOLDS" / f"fold{fold}" / split / c
            d.mkdir(parents=True, exist_ok=True)
            fake_image(seed, tone).save(d / f"{stem}.jpg", quality=95)
    # Augmented copies must be ignored.
    aug = msld / "Augmented Images" / "fold1_AUG" / "Train" / "Mpox"
    aug.mkdir(parents=True, exist_ok=True)
    fake_image(1, 0).rotate(30).save(aug / "aug_0.jpg")

    msid = base / "msid"
    names = {"Monkeypox": "mpox", "Chickenpox": "chickenpox", "Measles": "measles", "Normal": "healthy"}
    for c in names:
        (msid / c).mkdir(parents=True, exist_ok=True)
        for p in range(18):
            fake_image(int(RNG.integers(1e9)), p % 3).save(msid / c / f"{c.lower()}{p}.jpg", quality=92)

    # Planted duplicates: 3 MSLD mpox images re-encoded / mirrored into MSID.
    src = sorted((msld / "Original Images" / "FOLDS" / "fold1").rglob("Mpox/*.jpg"))[:3]
    with Image.open(src[0]) as im:
        im.save(msid / "Monkeypox" / "dup_reencoded.jpg", quality=70)
    with Image.open(src[1]) as im:
        ImageOps.mirror(im).save(msid / "Monkeypox" / "dup_mirrored.jpg", quality=90)
    with Image.open(src[2]) as im:
        im.resize((300, 245)).save(msid / "Monkeypox" / "dup_resized.png")
    # One within-MSID duplicate with a different label (label conflict case).
    with Image.open(msid / "Measles" / "measles0.jpg") as im:
        im.save(msid / "Chickenpox" / "dup_measles_as_chickenpox.jpg", quality=80)
    return msld, msid


def main():
    assert canonical_class("Monkeypox") == "mpox"
    assert canonical_class("Hand, Foot and Mouth Disease") == "hfmd"
    assert canonical_class("Normal") == "healthy"
    assert canonical_class("Train") is None

    tmp = Path(tempfile.mkdtemp(prefix="mpoxgen_check_"))
    try:
        msld, msid = build_fake(tmp / "data")
        work = tmp / "work"
        shutil.copytree(ROOT / "scripts", work / "scripts")
        shutil.copytree(ROOT / "src", work / "src")

        run(["scripts/01_build_index.py", "--data", f"msld_v2={msld}", "--data", f"msid={msid}",
             "--out", "data/index"], work)
        images = pd.read_csv(work / "data/index/images.csv")
        n_msld = (images.dataset == "msld_v2").sum()
        assert n_msld == 84, f"expected 84 unique MSLD images (augmented ignored), got {n_msld}"
        assert (images.dataset == "msid").sum() == 72 + 4
        off = pd.read_csv(work / "data/index/official_folds.csv")
        assert set(off.fold) == {1, 2} and set(off.split) == {"train", "val", "test"}

        run(["scripts/03_duplicate_audit.py", "--index", "data/index"], work)
        images = pd.read_csv(work / "data/index/images.csv")
        pairs = pd.read_csv(work / "data/index/duplicate_pairs.csv")
        cross = pairs[pairs.cross_dataset]
        print("cross-dataset pairs found:", len(cross))
        assert len(cross) >= 3, "planted cross-dataset duplicates were not all found"
        assert pairs.label_mismatch.sum() >= 1, "planted label-mismatch duplicate not found"
        assert images["ita"].notna().mean() > 0.9

        run(["scripts/04_make_splits.py", "--index", "data/index", "--out", "splits",
             "--reps", "2", "--cross-seeds", "2"], work)
        man = pd.read_csv(work / "splits/manifest.csv")
        strict = man[man.protocol.isin(["P2", "P3"])]
        assert (strict.shared_groups == 0).all() and (strict.shared_images == 0).all()
        p3 = man[(man.protocol == "P3") & (man.train_dataset == "msld_v2") & (man.task == "binary")]
        assert (p3["test_removed_as_duplicates_of_train"] >= 3).all()

        # Fake runs (random predictions) only to exercise 06_summarize.py.
        for r in man.itertuples():
            spec = json.loads((work / "splits" / f"{r.name}.json").read_text())
            n_cls = 2 if r.task == "binary" else 4
            lab = images.set_index("image_id").loc[spec["test"], "label"]
            from mpoxgen.labels import task_target
            y = np.array([task_target(l, r.task) for l in lab])
            prob = RNG.dirichlet(np.ones(n_cls), len(y))
            for model in ("modelA", "modelB"):
                d = work / "results" / r.name / model
                d.mkdir(parents=True, exist_ok=True)
                pr = pd.DataFrame({"image_id": spec["test"], "y_true": y, "y_pred": prob.argmax(1)})
                for c in range(n_cls):
                    pr[f"p{c}"] = prob[:, c]
                pr.to_csv(d / "predictions.csv", index=False)
                rec = {"protocol": r.protocol, "task": r.task, "rep": int(r.rep),
                       "train_dataset": r.train_dataset, "test_dataset": r.test_dataset,
                       "model": model, **metrics.compute(y, prob, 1 if r.task == "binary" else 0)}
                (d / "metrics.json").write_text(json.dumps(rec))
        run(["scripts/06_summarize.py", "--results", "results", "--index", "data/index"], work)
        for f in ("summary_by_model.csv", "inflation_gap.csv", "ita_breakdown.csv", "paper_tables.md",
                  "fig_protocols.png"):
            assert (work / "results" / f).exists(), f
        shutil.copy(work / "results" / "fig_protocols.png", ROOT / "tests" / "_fig_protocols_FAKE_DATA.png")

        # Metric sanity.
        y = np.array([0, 0, 1, 1])
        p = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.45, 0.55]])
        m = metrics.compute(y, p, 1)
        assert m["accuracy"] == 1.0 and m["auroc"] == 1.0 and m["mpox_sensitivity"] == 1.0
        assert ita.band(50) == "light" and ita.band(20) == "intermediate" and ita.band(-5) == "dark"
        print("\nALL CPU CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
