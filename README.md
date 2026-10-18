# How well do mpox skin-lesion classifiers generalize?

Code for a strict evaluation of deep-learning mpox classifiers: the same
models are tested under three protocols, after a duplicate audit of the data.

| Protocol | Test set | Purpose |
| --- | --- | --- |
| **P1 naive** | random image-level 70/20/10 split | the setup many papers use |
| **P2 patient-level** | official patient-independent folds (MSLD v2.0) or duplicate-group splits; no duplicate group crosses train and test | honest within-dataset score |
| **P3 cross-dataset** | train on one dataset, test on the other, minus duplicates of training images | closest proxy for new patients and cameras |

Also included: a perceptual-hash + embedding **duplicate audit** within and
across datasets, an approximate **skin-tone (ITA) breakdown**, and
**Grad-CAM** sheets for checking what the models look at.

> **Status:** experiments complete (Kaggle, Tesla T4, 156 trainings in 155 min,
> October 2026). Summary tables and the Grad-CAM ratings are in
> `paper_results/`. Per-image predictions and split files are in the Kaggle
> notebook output and will be published here. The paper is in preparation.

## Results at a glance

Binary task (mpox vs. non-mpox), balanced accuracy, mean over models.

| Finding | Number |
| --- | --- |
| Duplicate pairs shared between MSID and MSLD v2.0 | 67 (plus 21 within the datasets) |
| Inflation from a naive split (P1 − P2) | +1.9 points (MSID), +2.6 points (MSLD v2.0) |
| Trained on MSID: own test (P2) → MSLD v2.0 (P3) | 0.919 → 0.688 |
| Trained on MSLD v2.0: own test (P2) → MSID (P3) | 0.805 → 0.853 |
| MSLD v2.0 images called mpox by MSID-trained models | cowpox 95.5%, chickenpox 76.1%, HFMD 51.4% |
| Cross-dataset mpox sensitivity, dark vs. light skin (ITA) | 0.874 vs. 0.579 |
| Grad-CAM peak on the lesion (83 rated maps) | 55 lesion, 19 mixed, 9 off-lesion |

![Balanced accuracy by protocol](paper_results/fig1_protocols.png)

## Data

Both datasets are public and on Kaggle. Cite them if you use this code.

| Dataset | Kaggle | Original source |
| --- | --- | --- |
| MSLD v2.0 (755 images, 541 patients, 6 classes) | `joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20` | Ali et al., Biomed. Signal Process. Control 98:106742 (2024); arXiv:2306.14169 |
| MSID (4 classes) | `dipuiucse/monkeypoxskinimagedataset` | Bala & Hossain, Mendeley Data, doi:10.17632/r9bfpnvyxr.6 |

Only original images are used; folders whose path contains `aug` are skipped
and augmentation is applied inside training folds only.

## Run it on Kaggle (free GPU)

1. Create a new Kaggle notebook. **Settings:** Accelerator = GPU, Internet = On.
2. **Add Input:** the two datasets above.
3. **One-cell run:** put this in a single cell, then use
   **Save Version → Save & Run All** so it runs in the background:

   ```
   !git clone -q https://github.com/MirRahibHussain/mpox-generalization.git && bash mpox-generalization/run_all.sh
   ```

   `run_all.sh` finds the datasets, runs every step (smoke test first), and
   writes `key_results/` and `outputs.zip` to `/kaggle/working`.
4. **Step-by-step instead:** import `notebooks/kaggle_run.ipynb`
   (File → Import Notebook), set `REPO_URL`, and run the cells in order.

## Method study: TALE (in progress)

TALE (Tone-Aware Look-alike Exposure) targets the two failures found above:

- look-alike exposure: about 1,200 images of 24 papulovesicular, pustular, bullous or crusted
  conditions from Fitzpatrick17k are added as extra non-mpox training images (never used for
  validation or testing; duplicates of study images removed);
- tone-balanced sampling: every (class, skin-tone band) group is drawn equally often;
- tone-shift augmentation: the healthy skin around a lesion is moved to a random ITA band
  while the lesion is left unchanged.

It is compared with standard fine-tuning, hue/saturation jitter, mixup and Group DRO, plus
ablations, on the same splits (`src/mpoxgen/methods.py`, `scripts/08`-`10`). On Kaggle, add
the Fitzpatrick17k images (e.g. `mobaswiralfarabi/fitzpatrick17k-original`) as a third input
and run:

```
!git clone -q https://github.com/MirRahibHussain/mpox-generalization.git && bash mpox-generalization/run_methods.sh
```

`python tests/run_method_checks.py` checks the method code on synthetic images.

## Pipeline

```bash
pip install -r requirements.txt

# 1. index (one row per unique file; official folds kept)
python scripts/01_build_index.py --data msld_v2=PATH_TO_MSLD --data msid=PATH_TO_MSID
# 2. (GPU, optional) embedding near-duplicates -> review embedding_review_sheet.png
python scripts/02_embedding_pairs.py
# 3. duplicate audit (hash + accepted embedding pairs) and ITA skin tone
python scripts/03_duplicate_audit.py
# 4. P1/P2/P3 splits + leakage check (fails loudly if P2/P3 leak)
python scripts/04_make_splits.py
# 5. (GPU) train/evaluate; resumable; --smoke for a 1-epoch test
python scripts/05_run_experiments.py --tasks binary --save-ckpt
# 6. tables + figure for the paper
python scripts/06_summarize.py
# 7. Grad-CAM sheets (CNNs), then fill the 'attention' column by eye
python scripts/07_gradcam.py --run results/binary_P3_msld_v2_to_msid_rep0/resnet50
```

Models (timm, ImageNet-pretrained): ResNet-50, DenseNet-121, EfficientNet-B0,
MobileNetV3-Large, ConvNeXt-Tiny, ViT-Small/16. Training: 224 px, AdamW,
cosine schedule, up to 30 epochs, early stopping on validation balanced
accuracy (patience 7), class-weighted cross-entropy, mild brightness/contrast
jitter only (no hue change, so skin tone is not altered).

## Outputs

| File | Content |
| --- | --- |
| `data/index/duplicate_summary.csv` | duplicate counts within and across datasets |
| `data/index/cross_dataset_duplicates.png` | contact sheet of cross-dataset duplicates |
| `splits/*.json`, `splits/manifest.csv` | every split, reusable by others |
| `results/<split>/<model>/predictions.csv` | per-image predictions |
| `results/summary_by_model.csv` | mean ± sd per model and protocol |
| `results/inflation_gap.csv` | P1 − P2 (inflation) and P2 − P3 (generalization gap) |
| `results/ita_breakdown.csv` | mpox sensitivity/specificity per skin-tone band |
| `results/fig_protocols.png` | balanced accuracy by protocol |

## Checks

`python tests/run_cpu_checks.py` builds two synthetic datasets with planted
duplicates and checks indexing, the duplicate audit, ITA, split leakage and
the summary step on CPU. It does not train models and produces no research
results.

## Limitations to keep in mind

- Both datasets are web-collected, not clinical cohorts.
- MSID has no patient IDs; its P2 splits group duplicates only.
- ITA from web photos without lesion masks is an approximate skin-tone proxy.

## Citation

If you use this code, please cite the paper (details to be added) and the two
datasets above.

## Acknowledgment of AI assistance

An AI assistant (Claude, Anthropic) drafted the code, set up and launched the
Kaggle runs, rated the Grad-CAM maps in `paper_results/gradcam_ratings.csv`,
and drafted the manuscript. The author is responsible for the study and its
results.

## License

MIT - see `LICENSE`.
