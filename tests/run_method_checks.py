"""CPU checks of the method-study code on synthetic images (no training).

Checks the tone-shift augmentation, tone-balanced sampling weights, the
look-alike, control and atlas pool builders (including removal of a planted
duplicate of a study image), the FACTOR draw weights and sampler, the external
MCSI indexer (including a planted crop and copy of a study image) and the
summary scripts on fake predictions. If PyTorch is not
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
    (STUB / "torch" / "utils" / "data").mkdir(parents=True)
    (STUB / "torch" / "__init__.py").write_text(
        "def initial_seed():\n    return 12345\nclass Generator: pass\n")
    (STUB / "torch" / "nn" / "__init__.py").write_text("")
    (STUB / "torch" / "nn" / "functional.py").write_text("")
    (STUB / "torch" / "utils" / "__init__.py").write_text("")
    (STUB / "torch" / "utils" / "data" / "__init__.py").write_text("class Sampler: pass\nclass Dataset: pass\n")
    sys.path.insert(0, str(STUB))
    os.environ["PYTHONPATH"] = str(STUB) + os.pathsep + os.environ.get("PYTHONPATH", "")

from run_cpu_checks import build_fake, fake_image, run  # noqa: E402

from mpoxgen import external, metrics  # noqa: E402
from mpoxgen.ita import skin_mask  # noqa: E402
from mpoxgen.labels import task_target  # noqa: E402
from mpoxgen.factor import EpochSampler, atlas_weights  # noqa: E402
from mpoxgen.methods import CONFIGS, balanced_weights, config, group_ids, shift_tone  # noqa: E402
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
    assert set(CONFIGS) == {"erm", "hsv", "mixup", "gdro", "la", "rand", "tone", "tale", "atlas", "hod",
                            "factor", "factor_notone", "factor_static", "div5", "div20"}
    assert config("erm")["head"] == "binary" and config("factor")["draw"] == "cond_tone"
    assert config("factor_static")["extra"] == "control" and config("div5")["n_conditions"] == 5
    assert CONFIGS["rand"]["extra"] == "control" and CONFIGS["la"]["extra"] == "lookalike"


def check_factor():
    """Atlas draw weights and the per-epoch sampler (the losses need real PyTorch)."""
    cond = np.array(["a"] * 60 + ["b"] * 30 + ["c"] * 10)
    bands = np.array((["light"] * 50 + ["dark"] * 10) + (["light"] * 20 + ["dark"] * 10) + ["intermediate"] * 10)
    w = atlas_weights(cond, bands, mode="cond")
    per = pd.Series(w).groupby(cond).sum()
    assert np.allclose(per, 1 / 3), "condition-uniform draw should give every condition the same weight"
    target = ["dark"] * 6 + ["light"] * 3 + ["intermediate"] * 1
    wt = atlas_weights(cond, bands, mode="cond_tone", mpox_bands=target)
    mix = pd.Series(wt).groupby(bands).sum()
    assert np.isclose(mix["dark"], 0.6) and np.isclose(mix["light"], 0.3) and np.isclose(mix["intermediate"], 0.1), \
        f"tone matching should reproduce the mpox band mix, got {mix.to_dict()}"
    w2 = atlas_weights(cond, bands, mode="cond", n_conditions=2, seed=3)
    assert (pd.Series(w2).groupby(cond).sum() > 0).sum() == 2, "n_conditions should restrict the draw"
    try:
        atlas_weights(cond, bands, mode="cond_tone")
        raise AssertionError("cond_tone without mpox bands was accepted")
    except ValueError:
        pass
    s = EpochSampler(n_study=40, atlas_offset=40, atlas_w=w, k=25, seed=0)
    e1, e2 = list(iter(s)), list(iter(s))
    assert len(e1) == len(s) == 65
    assert sorted(i for i in e1 if i < 40) == list(range(40)), "every study image once per epoch"
    a1, a2 = [i for i in e1 if i >= 40], [i for i in e2 if i >= 40]
    assert len(a1) == 25 and len(set(a1)) == 25 and max(a1) < 140, "atlas draw without replacement, in range"
    assert set(a1) != set(a2), "a new atlas subset should be drawn every epoch"
    print("factor: draw weights, tone matching and epoch sampler OK")


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


def build_fake_mcsi(base: Path, study_img: Path):
    """Fake MCSI folder: 4 classes x 6 images, plus a crop+zoom of a study image and an exact copy."""
    root = base / "MCSI"
    (root / "images").mkdir(parents=True)
    rows, tone = [], []
    for cls, n in [("monkeypox", 6), ("chickenpox", 6), ("normal", 6), ("acne_l1", 6)]:
        for k in range(n):
            f = f"{cls}_{k}.png"
            fake_image(int(RNG.integers(1e9)), k % 3).save(root / "images" / f)
            rows.append({"diagnostic": cls, "img_id": f, "source": "Fake"})
            tone.append({"Filename": f, "ITA": 20.0, "CLS": "tan1", "Type": "light"})
    with Image.open(study_img) as im:
        w, h = im.size
        im.crop((w // 6, h // 6, w - w // 5, h - h // 5)).resize((300, 260)).save(root / "images" / "monkeypox_crop.png")
        im.save(root / "images" / "monkeypox_copy.jpg", quality=80)
    rows += [{"diagnostic": "monkeypox", "img_id": "monkeypox_crop.png", "source": "Fake"},
             {"diagnostic": "monkeypox", "img_id": "monkeypox_copy.jpg", "source": "Fake"}]
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
    pd.DataFrame(tone).to_csv(root / "skin_tone_labels.csv", index=False)
    return root


def main():
    check_tone_shift()
    check_sampling()
    check_factor()
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
                   "--index", "data/index", "--out", "data/pool", "--per-condition", "8",
                   "--control-out", "data/pool_control"], work)
        pool = pd.read_csv(work / "data/pool/pool.csv")
        assert set(pool["condition"]) <= set(external.LOOKALIKE_CONDITIONS), "non-look-alike condition in pool"
        assert "psoriasis" not in set(pool["condition"])
        assert pool.groupby("condition").size().max() <= 8
        assert not any(pool["path"].str.contains("f" * 32)), "planted duplicate of a study image kept"
        assert not any(pool["path"].str.contains(f"{4:032x}")), "QC 'wrongly labelled' image kept"
        assert pool["ita_band"].notna().all()
        assert (work / "data/pool/tone_shift_examples.png").exists()
        print(f"pool: {len(pool)} images; planted duplicate removed")
        ctl = pd.read_csv(work / "data/pool_control/pool.csv")
        assert len(ctl) > 0 and len(ctl) <= len(pool)
        assert not set(ctl["condition"]) & set(external.LOOKALIKE_CONDITIONS), "look-alike condition in control pool"
        assert not set(ctl["md5"]) & set(pool["md5"]), "control pool overlaps the look-alike pool"
        lim = pool["fitzpatrick"].value_counts()
        assert all(n <= lim.get(f, 0) for f, n in ctl["fitzpatrick"].value_counts().items()), "Fitzpatrick mix exceeds target"
        print(f"control pool: {len(ctl)} images from {ctl['condition'].nunique()} non-look-alike condition(s)")
        run(["scripts/15_atlas_pool.py", "--fitz", str(fitz_root), "--csv", str(csv),
             "--index", "data/index", "--out", "data/pool_atlas", "--workers", "2"], work)
        atl = pd.read_csv(work / "data/pool_atlas/pool.csv")
        assert set(atl["condition"]) == {"scabies", "folliculitis", "milia", "psoriasis"}, "atlas should keep all conditions"
        assert not any(atl["path"].str.contains("f" * 32)), "planted duplicate of a study image kept in atlas"
        assert not any(atl["path"].str.contains(f"{4:032x}")), "QC 'wrongly labelled' image kept in atlas"
        assert atl["ita_band"].notna().all() and len(atl) == 47
        print(f"atlas pool: {len(atl)} images; planted duplicate removed")
        run(["scripts/16_ita_validation.py", "--pool", "data/pool_atlas/pool.csv", "--index", "data/index",
             "--out", "results_methods", "--n-boot", "50"], work)
        mix = pd.read_csv(work / "results_methods/atlas_tone_mix.csv")
        for ds, g in mix.groupby("dataset"):
            tgt = g[g["draw"] == "mpox images (target)"].iloc[0]
            got = g[g["draw"] == "atlas, cond_tone"].iloc[0]
            for b in ["light", "intermediate", "dark"]:
                if tgt[b] > 0 and (atl["ita_band"] == b).any():
                    assert abs(tgt[b] - got[b]) < 0.002, f"tone matching off for {ds}/{b}"
        mcsi_root = build_fake_mcsi(tmp / "data", mpox_img)
        run(["scripts/18_external_mcsi.py", "--root", str(mcsi_root), "--index", "data/index",
             "--out", "data/external_mcsi"], work)
        ext = pd.read_csv(work / "data/external_mcsi/images.csv")
        clean = pd.read_csv(work / "data/external_mcsi/images_clean.csv")
        dups = set(ext.loc[ext["duplicate_any"], "file"])
        assert {"monkeypox_crop.png", "monkeypox_copy.jpg"} <= dups, f"planted MCSI duplicates not found: {dups}"
        assert len(dups) <= 3, f"too many MCSI images flagged on fake data: {dups}"
        assert not set(clean["file"]) & dups and set(clean["label"]) == {"mpox", "chickenpox", "healthy", "acne"}
        print(f"MCSI: {len(ext)} images, flagged {sorted(dups)}, clean {len(clean)}")
        run(["scripts/12_tone_shift_check.py", "--index", "data/index", "--out", "results_methods",
             "--per-group", "4"], work)
        assert (work / "results_methods/tone_shift_check.md").exists()

        # Fake runs for every method to exercise 10_summarize_methods.py.
        images = pd.read_csv(work / "data/index/images.csv")
        man = pd.read_csv(work / "splits/manifest.csv")
        man = man[(man.task == "binary") & man.protocol.isin(["P2", "P3"])]
        for r in man.itertuples():
            spec = json.loads((work / "splits" / f"{r.name}.json").read_text())
            lab = images.set_index("image_id").loc[spec["test"], "label"]
            y = np.array([task_target(l, "binary") for l in lab])
            meths = list(CONFIGS) if r.protocol == "P3" else ["erm", "tale"]
            vlab = images.set_index("image_id").loc[spec["val"], "label"]
            yv = np.array([task_target(l, "binary") for l in vlab])
            for model in ("modelA", "modelB"):
                for meth in meths:
                    prob = RNG.dirichlet(np.ones(2), len(y))
                    d = work / "results_methods" / r.name / model / meth
                    d.mkdir(parents=True, exist_ok=True)
                    pr = pd.DataFrame({"image_id": spec["test"], "y_true": y, "y_pred": prob.argmax(1),
                                       "p0": prob[:, 0], "p1": prob[:, 1]})
                    if config(meth)["head"] == "factor":
                        pr["pd"], pr["pm"] = RNG.random(len(y)), RNG.random(len(y))
                    pr.to_csv(d / "predictions.csv", index=False)
                    if r.protocol == "P3":
                        pe = RNG.dirichlet(np.ones(2), len(clean))
                        pm_ = pd.DataFrame({"image_id": clean["image_id"], "y_true": (clean["label"] == "mpox").astype(int),
                                            "y_pred": pe.argmax(1), "p0": pe[:, 0], "p1": pe[:, 1]})
                        if config(meth)["head"] == "factor":
                            pm_["pd"], pm_["pm"] = RNG.random(len(clean)), RNG.random(len(clean))
                        pm_.to_csv(d / "predictions_mcsi.csv", index=False)
                    pv = RNG.dirichlet(np.ones(2), len(yv))
                    pd.DataFrame({"image_id": spec["val"], "y_true": yv, "y_pred": pv.argmax(1),
                                  "p0": pv[:, 0], "p1": pv[:, 1]}).to_csv(d / "val_predictions.csv", index=False)
                    rec = {"protocol": r.protocol, "task": "binary", "rep": int(r.rep),
                           "train_dataset": r.train_dataset, "test_dataset": r.test_dataset,
                           "model": model, "method": meth, **metrics.compute(y, prob, 1)}
                    (d / "metrics.json").write_text(json.dumps(rec))
        run(["scripts/10_summarize_methods.py", "--results", "results_methods", "--index", "data/index"], work)
        run(["scripts/11_operating_points.py", "--results", "results_methods"], work)
        op = pd.read_csv(work / "results_methods/operating_points_runs.csv")
        assert (op["val_sens_val90"] >= 0.9 - 1e-9).all(), "validation threshold misses the 90% sensitivity target"
        assert set(op["method"]) == set(CONFIGS)
        run(["scripts/13_paired_stats.py", "--runs", "results_methods/all_runs.csv",
             "--metrics", "balanced_accuracy", "mpox_specificity"], work)
        ps = pd.read_csv(work / "results_methods/paired_stats.csv")
        assert len(ps) and (ps["p_holm"] >= ps["p"] - 1e-12).all() and (ps["p_holm"] <= 1).all()
        assert ((ps["ci_low"] <= ps["mean_diff"] + 1e-12) & (ps["mean_diff"] <= ps["ci_high"] + 1e-12)).all()
        run(["scripts/17_open_set.py", "--results", "results_methods", "--index", "data/index"], work)
        osr = pd.read_csv(work / "results_methods/open_set_runs.csv")
        assert set(osr["method"]) == set(CONFIGS)
        assert osr.loc[osr["method"] == "factor", "factor_mpox_auroc"].notna().all()
        assert osr.loc[osr["method"] == "erm", "factor_mpox_auroc"].isna().all()
        un = osr[osr["direction"] == "msid->msld_v2"]["n_unseen"]
        assert (un > 0).all(), "cowpox/HFMD should be unseen classes for MSID-trained runs"
        assert (osr[osr["direction"] == "msld_v2->msid"]["n_unseen"] == 0).all()
        run(["scripts/13_paired_stats.py", "--runs", "results_methods/open_set_runs.csv", "--pairs",
             "factor:erm", "factor:atlas", "--out", "results_methods/open_set_paired",
             "--metrics", "closed_bal_acc", "unseen_spec"], work)
        assert (work / "results_methods/open_set_paired.md").exists()
        run(["scripts/19_external_eval.py", "--results", "results_methods",
             "--external", "data/external_mcsi/images_clean.csv"], work)
        er = pd.read_csv(work / "results_methods/external_runs.csv")
        assert set(er["direction"]) == {"msid->mcsi", "msld_v2->mcsi"} and set(er["method"]) == set(CONFIGS)
        assert er["called_acne"].notna().all() and er["shortcut_gap"].notna().all()
        assert er.loc[er["method"] == "factor", "factor_mpox_auroc"].notna().all()
        run(["scripts/13_paired_stats.py", "--runs", "results_methods/external_runs.csv", "--pairs",
             "factor:erm", "atlas:erm", "--out", "results_methods/external_paired",
             "--metrics", "balanced_accuracy", "called_acne"], work)
        eps = pd.read_csv(work / "results_methods/external_paired.csv")
        assert set(eps["direction"]) == {"msid->mcsi", "msld_v2->mcsi"}, "paired stats should cover the external directions"
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
