"""CPU checks of the method-study code on synthetic images (no training).

Checks the tone-shift augmentation, tone-balanced sampling weights, the
look-alike pool builder (including removal of a planted duplicate of a study
image) and the summary script on fake predictions. If PyTorch is not
installed, a tiny stand-in module is used so the non-training code can be
imported. Nothing here is a research result.

    python tests/run_method_checks.py
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

STUB = None
if importlib.util.find_spec("torch") is None:
    STUB = Path(tempfile.mkdtemp(prefix="torchstub_"))
    (STUB / "torch" / "nn").mkdir(parents=True)
    (STUB / "torch" / "__init__.py").write_text(
        "def initial_seed():\n    return 12345\nclass Generator: pass\n")
    (STUB / "torch" / "nn" / "__init__.py").write_text("")
    (STUB / "torch" / "nn" / "functional.py").write_text("")
    sys.path.insert(0, str(STUB))
    os.environ["PYTHONPATH"] = str(STUB) + os.pathsep + os.environ.get("PYTHONPATH", "")

from run_cpu_checks import build_fake, fake_image, run  # noqa: E402

from mpoxgen import external, metrics  # noqa: E402
from mpoxgen.ita import skin_mask  # noqa: E402
from mpoxgen.labels import task_target  # noqa: E402
from mpoxgen.methods import CONFIGS, balanced_weights, group_ids, shift_tone  # noqa: E402
from skimage.color import rgb2lab  # noqa: E402

RNG = np.random.default_rng(1)


def check_tone_shift():
    img = fake_image(7, 0)  # light synthetic skin with red lesions
    rgb = np.asarray(img)
    lab = rgb2lab(rgb)
    skin = skin_mask(rgb)
    from scipy.ndimage import binary_erosion
    red = (rgb[..., 0].astype(int) - rgb[..., 1].astype(int)) > 80  # the planted lesion blobs
    lesion = binary_erosion(red, iterations=4)
    out = shift_tone(img, target_ita=-10.0)
    assert out is not None, "tone shift skipped on a skin image"
    lab2 = rgb2lab(np.asarray(out))
    healthy = skin & ~red
    dl_skin = np.median(lab2[..., 0][healthy] - lab[..., 0][healthy])
    dl_lesion = np.median(np.abs(lab2[..., 0][lesion] - lab[..., 0][lesion]))
    print(f"tone shift: healthy-skin dL={dl_skin:.1f}, lesion |dL|={dl_lesion:.1f}")
    assert dl_skin < -8, "healthy skin should get darker for a dark target"
    assert dl_lesion < abs(dl_skin) / 2, "lesion should change much less than healthy skin"
    assert shift_tone(Image.new("RGB", (80, 80), (20, 120, 30)), 30.0) is None  # no skin -> skipped


def check_sampling():
    y = np.array([1] * 30 + [0] * 70)
    bands = ["dark"] * 20 + ["light"] * 10 + ["dark"] * 10 + ["light"] * 50 + ["intermediate"] * 10
    g = group_ids(y, bands)
    w = balanced_weights(g)
    totals = pd.Series(w).groupby(g).sum()
    assert np.allclose(totals, totals.iloc[0]), "every group should carry the same total weight"
    assert set(CONFIGS) == {"erm", "hsv", "mixup", "gdro", "la", "tone", "tale"}


def build_fake_fitz(base: Path, msld_img: Path):
    d = base / "fitzpatrick17k-original" / "finalfitz17k"
    d.mkdir(parents=True)
    rows = []
    conds = ["scabies", "folliculitis", "milia", "psoriasis"]  # psoriasis is not a look-alike
    for k in range(48):
        h = f"{k:032x}"
        cond = conds[k % 4]
        fake_image(int(RNG.integers(1e9)), k % 3).save(d / f"{h}.jpg", quality=92)
        qc = "3 Wrongly labelled" if k == 4 else ""
        rows.append({"md5hash": h, "fitzpatrick_scale": 1 + k % 6, "fitzpatrick_centaur": 1,
                     "label": cond, "nine_partition_label": "inflammatory",
                     "three_partition_label": "non-neoplastic", "qc": qc, "url": "", "url_alphanum": ""})
    # Planted duplicate of a study image inside the pool.
    h = "f" * 32
    with Image.open(msld_img) as im:
        im.save(d / f"{h}.jpg", quality=75)
    rows.append({"md5hash": h, "fitzpatrick_scale": 2, "fitzpatrick_centaur": 2, "label": "scabies",
                 "nine_partition_label": "inflammatory", "three_partition_label": "non-neoplastic",
                 "qc": "", "url": "", "url_alphanum": ""})
    csv = base / "fitzpatrick17k.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return d.parent, csv


def main():
    check_tone_shift()
    check_sampling()
    try:
        external.build_pool(".", ".", conditions=["mpox"])
        raise AssertionError("an evaluation class was accepted as a pool condition")
    except ValueError:
        pass

    tmp = Path(tempfile.mkdtemp(prefix="mpoxgen_method_check_"))
    try:
        msld, msid = build_fake(tmp / "data")
        mpox_img = sorted((msld / "Original Images").rglob("Mpox/*.jpg"))[0]
        fitz_root, csv = build_fake_fitz(tmp / "data", mpox_img)
        work = tmp / "work"
        shutil.copytree(ROOT / "scripts", work / "scripts")
        shutil.copytree(ROOT / "src", work / "src")
        run(["scripts/01_build_index.py", "--data", f"msld_v2={msld}", "--data", f"msid={msid}",
             "--out", "data/index"], work)
        run(["scripts/03_duplicate_audit.py", "--index", "data/index"], work)
        run(["scripts/04_make_splits.py", "--index", "data/index", "--out", "splits",
             "--reps", "2", "--cross-seeds", "2"], work)
        out = run(["scripts/08_lookalike_pool.py", "--fitz", str(fitz_root), "--csv", str(csv),
                   "--index", "data/index", "--out", "data/pool", "--per-condition", "8"], work)
        pool = pd.read_csv(work / "data/pool/pool.csv")
        assert set(pool["condition"]) <= set(external.LOOKALIKE_CONDITIONS), "non-look-alike condition in pool"
        assert "psoriasis" not in set(pool["condition"])
        assert pool.groupby("condition").size().max() <= 8
        assert not any(pool["path"].str.contains("f" * 32)), "planted duplicate of a study image kept"
        assert not any(pool["path"].str.contains(f"{4:032x}")), "QC 'wrongly labelled' image kept"
        assert pool["ita_band"].notna().all()
        assert (work / "data/pool/tone_shift_examples.png").exists()
        print(f"pool: {len(pool)} images; planted duplicate removed")

        # Fake runs for every method to exercise 10_summarize_methods.py.
        images = pd.read_csv(work / "data/index/images.csv")
        man = pd.read_csv(work / "splits/manifest.csv")
        man = man[(man.task == "binary") & man.protocol.isin(["P2", "P3"])]
        for r in man.itertuples():
            spec = json.loads((work / "splits" / f"{r.name}.json").read_text())
            lab = images.set_index("image_id").loc[spec["test"], "label"]
            y = np.array([task_target(l, "binary") for l in lab])
            meths = list(CONFIGS) if r.protocol == "P3" else ["erm", "tale"]
            for model in ("modelA", "modelB"):
                for meth in meths:
                    prob = RNG.dirichlet(np.ones(2), len(y))
                    d = work / "results_methods" / r.name / model / meth
                    d.mkdir(parents=True, exist_ok=True)
                    pr = pd.DataFrame({"image_id": spec["test"], "y_true": y, "y_pred": prob.argmax(1),
                                       "p0": prob[:, 0], "p1": prob[:, 1]})
                    pr.to_csv(d / "predictions.csv", index=False)
                    rec = {"protocol": r.protocol, "task": "binary", "rep": int(r.rep),
                           "train_dataset": r.train_dataset, "test_dataset": r.test_dataset,
                           "model": model, "method": meth, **metrics.compute(y, prob, 1)}
                    (d / "metrics.json").write_text(json.dumps(rec))
        run(["scripts/10_summarize_methods.py", "--results", "results_methods", "--index", "data/index"], work)
        for f in ("methods_tables.md", "table_p3_methods.csv", "table_tone_methods.csv",
                  "table_called_mpox_methods.csv", "table_p2_methods.csv", "table_tests.csv", "fig_methods.png"):
            assert (work / "results_methods" / f).exists(), f
        shutil.copy(work / "data/pool/tone_shift_examples.png", ROOT / "tests" / "_tone_shift_FAKE_DATA.png")
        print("\nALL METHOD CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if STUB:
            shutil.rmtree(STUB, ignore_errors=True)


if __name__ == "__main__":
    main()
