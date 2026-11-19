# Released splits and per-image predictions

Everything needed to recompute the paper's numbers without retraining: the split files,
the image index with duplicate groups and skin-tone estimates, the auxiliary negative pools,
and the per-image predictions of every training run. Files were extracted from the Kaggle
outputs of the runs listed below. No images are included; image paths point to the public
Kaggle copies of the datasets (see the main README).

| Folder | Study | Kaggle run |
| --- | --- | --- |
| `v2_evaluation/` | evaluation study: 6 backbones × P1/P2/P3 | Version 2, run 358186751 |
| `v3_tale/` | TALE method study | Version 3, run 359635265 |
| `v4_control/` | random-negative control, thresholds, tone-shift check | Version 4, run 360058236 |
| `v5_outlier_exposure/` | atlas exposure, HOD, FACTOR, condition diversity | Version 5, run 360301520 |
| `v7_external/` | pre-registered external test on MCSI | Version 7, run 360404890 |

## Contents

- `v2_evaluation/splits/`: every P1, P2 and P3 split (binary and 4-class; 52 JSON files) with
  training, validation and test image ids, the run's `manifest.csv`, and `verification.csv`.
  All later sessions used these same split files. `scripts/21_verify_splits.py` checks every
  split against the released predictions: the test ids of all 714 `predictions.csv` files and the
  validation ids of all 372 `val_predictions.csv` files match their split, and no P2/P3 split
  shares an image or duplicate group between training(+validation) and test.
  `scripts/04_make_splits.py` regenerates 42 of the 52 files identically from the released index;
  the other 10 (MSID P2) depend on the scikit-learn version, because they use StratifiedGroupKFold
  with shuffling, so these released files are the reference.
- `v2_evaluation/results/`: per-image predictions of the 156 evaluation-study trainings, plus the
  run's `all_runs.csv`, `summary_by_model.csv`, `inflation_gap.csv` and `ita_breakdown.csv`.
  These files and the split files were missing from the first release because `.gitignore`
  excluded every folder named `splits/` or `results/`; they were added from the Kaggle Version 2
  output unchanged.
- `v2_evaluation/index/`: `images.csv` (one row per unique image: dataset, label, md5,
  patient key from the file name, duplicate group, ITA and ITA band), `duplicate_pairs.csv`,
  `embedding_pairs.csv`, `duplicate_summary.csv`, `label_conflicts.csv` and
  `official_folds.csv` (the MSLD v2.0 patient-independent folds as shipped).
- `*/results*/<split>/<model or method>/predictions.csv`: one row per test image with
  `image_id`, `y_true`, `y_pred` and class probabilities. Runs of the method studies also
  have `val_predictions.csv` (validation set, used to choose thresholds); the external-test
  runs add predictions for the MCSI images.
- `*/results*/all_runs.csv` and summary tables: per-run metrics as written by the run.
- `v4_control/pools/`: the look-alike pool and the matched random-negative control pool
  (Fitzpatrick17k image ids, conditions, Fitzpatrick types and ITA).
- `v5_outlier_exposure/pools/atlas_pool.csv`: the whole-atlas pool used by the atlas, HOD
  and FACTOR methods.

Join predictions to `index/images.csv` on `image_id` to get the class, dataset and ITA band
of each test image.
