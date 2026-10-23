# Pre-registered external test on MCSI (Kaggle Version 7, run 360404890, 20 October 2026)

Unedited output of `run_external.sh` (Tesla T4, 326 min, exit 0; deterministic cuDNN). The analysis plan
(`PREREGISTRATION_MCSI.md`, copied here as the run used it) was committed before any model saw MCSI.
108 trainings: 6 methods × 3 backbones × 3 seeds × 2 training datasets (P3). Every model predicted its usual
cross-dataset test set and every MCSI image.

| File | Content |
| --- | --- |
| `external.md`, `external_runs.csv` | results on the pre-registered clean MCSI set (293 images, 6 mpox) |
| `external_direction.md`, `external_runs_direction.csv` | secondary results on the direction-specific clean sets (amendment 1) |
| `external_paired_vs_erm.*` | H1–H3: each method vs ERM (balanced accuracy, acne called mpox, shortcut gap, AUROC), Holm within family |
| `external_paired_factor.*` | H4: FACTOR vs the atlas head, HOD, FACTOR without tone matching and the control |
| `external_direction_paired_vs_erm.*` | the same tests on the direction-specific sets |
| `mcsi_report.txt`, `mcsi_pairs.csv`, `mcsi_images.csv` | crop-aware duplicate audit: per-image flags, every flagged and borderline pair with hash distance, cosine and ORB/RANSAC inliers |
| `all_runs.csv`, `methods_tables.md`, `paired_stats.*`, `open_set*`, `table_*`, `tone_*` | the same models on MSLD v2.0 and MSID |
| `run_log_external.txt` | full console log of the run |

ERM's one-sample test for H3 (gap > 0) is not in the files above: all 9 gaps are positive in both training
directions, so the exact two-sided Wilcoxon p = 0.0039 for each.

Note on amendment 1: it describes most removed mpox images as cropped copies found only by ORB. That holds for
the flagged *pairs* (88 of 204 flagged pairs were found only by ORB); per *image*, 95 of the 106 removed images
also had a hash or embedding match to some study image, and 11 (6 chickenpox, 5 mpox) were found only by ORB.
All 204 flagged pairs have the same class on both sides, against 25 of 45 borderline pairs (10–24 inliers).
The review sheet (`duplicates_review.png` in the run output, not included here because it shows third-party
images) covers the 80 pairs with the most inliers; every one shows the same photograph.
