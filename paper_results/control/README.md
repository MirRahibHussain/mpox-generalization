# Control study: results

Source: Kaggle notebook version 4 "Control study (random negatives)" (run 360058236),
19 October 2026, Tesla T4, 72 trainings, 182 minutes (169 minutes of training). Produced by
`run_control.sh`: ERM, look-alike exposure only (`la`), the random-negative control (`rand`)
and TALE, under P3 in both directions, ResNet-50 / EfficientNet-B0 / ViT-Small/16, seeds 0-2.

| File | Content |
| --- | --- |
| `methods_tables.md` | Kaggle summary from `scripts/10_summarize_methods.py`: P3 by method and direction, by backbone, skin tone, share of MSLD v2.0 images called mpox, pooled Wilcoxon tests |
| `operating_points.md` | Kaggle output of `scripts/11_operating_points.py`: AUROC, ROC operating points, validation-threshold results, Holm-adjusted Wilcoxon tests |
| `paired_stats.md` | `scripts/13_paired_stats.py` on `runs.csv`: per-direction mean difference, 95% bootstrap CI, wins/losses, exact Wilcoxon p, Holm-adjusted p (family: the 6 pairs of the 4 methods) |
| `runs.csv` | one row per run, same columns as `../tale/runs.csv` plus `valba` (validation balanced accuracy); `me` = erm, la, rand, tale; `ext` = extra Fitzpatrick17k images |
| `operating_points_runs.csv` | per run: AUROC, specificity at 90% sensitivity and sensitivity at 90% specificity (test ROC), validation thresholds (90% sensitivity, Youden) and the test results they give |
| `tone_sensitivity_runs.csv`, `tone_specificity_runs.csv` | per run, sensitivity (trained on MSLD v2.0) or specificity (trained on MSID) by ITA band; `gap` = dark - light |
| `tone_shift_check.md` | Kaggle output of `scripts/12_tone_shift_check.py` on 360 study images (1,080 shifts) |
| `control_pool_report.txt` | how the control pool was drawn: 1,258 candidates, 6 hash duplicates removed, 1,195 images from 90 conditions, Fitzpatrick types matched to the look-alike pool |

Values in the CSV files are rounded to 4 decimals; the Kaggle markdown reports were computed from
full-precision values, so the last digit can differ by 1.
