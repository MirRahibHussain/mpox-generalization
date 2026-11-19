# FACTOR study (Kaggle Version 5, run 360301520, 20 October 2026)

Unedited output of `run_factor.sh` (Tesla T4, 418 min, exit 0; deterministic cuDNN). 192 trainings:
7 methods × 3 backbones (ResNet-50, EfficientNet-B0, ViT-Small/16) × 3 seeds × 2 directions under P3,
the condition-diversity test (ResNet-50, MSID-trained, 5 and 20 conditions, 3 seeds each), and ERM and
FACTOR under P2 (5 folds per dataset).

| File | Content |
| --- | --- |
| `all_runs.csv` | one row per training: metrics with bootstrap CIs, settings, minutes |
| `methods_tables.md`, `table_*.csv` | summary tables (P3 by method and backbone, skin tone, share called mpox, P2) |
| `paired_stats.md/.csv` | pre-specified paired comparisons per direction: bootstrap CI, exact Wilcoxon, Holm |
| `open_set.md`, `open_set_runs.csv`, `open_set_paired.*` | seen vs unseen negative classes; AUROC vs healthy and vs other diseases (shortcut gap = difference) |
| `operating_points.*` | AUROC, test-ROC operating points and validation-chosen thresholds |
| `ita_validation.md`, `ita_by_fst.csv`, `ita_band_vs_fst_group.csv`, `fig_ita_vs_fst.png` | ITA estimate vs annotated Fitzpatrick type on Fitzpatrick17k |
| `atlas_tone_mix.csv`, `atlas_pool_report.txt`, `control_pool_report.txt` | auxiliary negative pools and the achieved tone mix |
| `tone_*_runs.csv` | sensitivity and specificity per ITA band per run |
| `manifest.csv` | split files used |
| `run_log_factor.txt` | full console log of the run |

The atlas example montage of the run is not included because it shows third-party images.
