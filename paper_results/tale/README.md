# TALE method study: results

Source: Kaggle notebook version 3 "TALE method study" (run 359635265), 18 October 2026,
Tesla T4, 186 trainings, 331 minutes. Produced by `run_methods.sh`.

| File | Content |
| --- | --- |
| `methods_tables.md` | summary tables from `scripts/10_summarize_methods.py` (P3 by method and direction, by backbone, skin tone, share of images called mpox, P2, pooled Wilcoxon tests) and the look-alike pool report |
| `runs.csv` | one row per training run. `p` protocol; `tr` training set (m = MSID, l = MSLD v2.0; P3 tests on the other set); `mo` backbone (r = ResNet-50, e = EfficientNet-B0, v = ViT-Small/16); `me` method (erm, hsv = colour jitter, mixup, gdro = Group DRO, la = look-alike only, tone = tone only, tale); `rep` seed or fold; `ba` balanced accuracy; `se` mpox sensitivity; `sp` specificity; `au` AUROC; `f1` macro-F1; `min` minutes; `ntr` training images; `ext` look-alike images added; `be` best epoch |
| `tone_sensitivity_runs.csv` | per P3 run trained on MSLD v2.0: mpox sensitivity on MSID by ITA band |
| `tone_specificity_runs.csv` | per P3 run trained on MSID: specificity on MSLD v2.0 by ITA band |
| `paired_stats.md` | per-direction paired comparisons (`scripts/13_paired_stats.py`): mean difference with 95% bootstrap CI, wins, exact Wilcoxon p and Holm-adjusted p (family: TALE vs each other method, look-alike only vs ERM, tone only vs ERM) |

Values in `runs.csv` are rounded to 4 decimals. Within-direction comparisons in the paper use
the exact paired Wilcoxon signed-rank test over the 9 matched (backbone, seed) pairs.
