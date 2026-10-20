## P3: cross-dataset, mean ± SD over models and seeds
| Method | MSID → MSLD v2.0: bal. acc. | MSID → MSLD v2.0: spec. | MSID → MSLD v2.0: sens. | MSLD v2.0 → MSID: bal. acc. | MSLD v2.0 → MSID: spec. | MSLD v2.0 → MSID: sens. | Mean bal. acc. (both) | n runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ERM (standard fine-tuning) | 0.701 ± 0.045 | 0.476 ± 0.109 | 0.926 ± 0.032 | 0.849 ± 0.028 | 0.968 ± 0.015 | 0.731 ± 0.048 | 0.775 ± 0.085 | 18 |
| Look-alike exposure only | 0.744 ± 0.030 | 0.633 ± 0.084 | 0.855 ± 0.059 | 0.852 ± 0.020 | 0.955 ± 0.032 | 0.749 ± 0.026 | 0.798 ± 0.061 | 18 |
| Random-negative control | 0.757 ± 0.020 | 0.673 ± 0.088 | 0.842 ± 0.076 | 0.849 ± 0.020 | 0.965 ± 0.019 | 0.733 ± 0.044 | 0.803 ± 0.051 | 18 |
| TALE (proposed) | 0.752 ± 0.026 | 0.653 ± 0.096 | 0.852 ± 0.070 | 0.842 ± 0.020 | 0.967 ± 0.023 | 0.717 ± 0.048 | 0.797 ± 0.052 | 18 |

## P3 balanced accuracy by backbone (mean over directions and seeds)
| method | efficientnet_b0 | resnet50 | vit_small_patch16_224 |
| --- | --- | --- | --- |
| erm | 0.75 | 0.763 | 0.812 |
| la | 0.796 | 0.781 | 0.816 |
| rand | 0.787 | 0.808 | 0.815 |
| tale | 0.78 | 0.792 | 0.82 |

## Skin tone (sensitivity: MSLD v2.0 → MSID; specificity: MSID → MSLD v2.0)
| Method | Sens. dark | Sens. light | Sens. gap (dark-light) | Worst-band sens. | Spec. dark | Spec. light | Spec. gap (dark-light) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ERM (standard fine-tuning) | 0.859 | 0.553 | +0.306 | 0.553 | 0.405 | 0.512 | -0.107 |
| Look-alike exposure only | 0.868 | 0.592 | +0.276 | 0.592 | 0.551 | 0.669 | -0.118 |
| Random-negative control | 0.857 | 0.544 | +0.313 | 0.544 | 0.602 | 0.708 | -0.106 |
| TALE (proposed) | 0.851 | 0.542 | +0.309 | 0.542 | 0.621 | 0.676 | -0.054 |

## % of MSLD v2.0 images called mpox by MSID-trained models
| method | chickenpox | cowpox | healthy | hfmd | measles | mpox |
| --- | --- | --- | --- | --- | --- | --- |
| erm | 75.8 | 94.6 | 24.6 | 45.5 | 42.5 | 92.6 |
| la | 64.9 | 78.3 | 23.8 | 18.5 | 25.9 | 85.5 |
| rand | 61.6 | 68.2 | 18.8 | 17.5 | 21.8 | 84.2 |
| tale | 64.6 | 75.8 | 20.9 | 16.6 | 24.0 | 85.2 |

## Paired Wilcoxon signed-rank tests (P3 balanced accuracy)
| Comparison | pairs | mean diff | wins | Wilcoxon p |
| --- | --- | --- | --- | --- |
| TALE (proposed) vs ERM (standard fine-tuning) | 18 | +0.022 | 11 | 0.0495 |
| TALE (proposed) vs Look-alike exposure only | 18 | -0.001 | 7 | 0.5226 |
| TALE (proposed) vs Random-negative control | 18 | -0.006 | 6 | 0.2121 |
| Look-alike exposure only vs ERM (standard fine-tuning) | 18 | +0.023 | 11 | 0.0483 |
| Random-negative control vs ERM (standard fine-tuning) | 18 | +0.028 | 11 | 0.0483 |
| Look-alike exposure only vs Random-negative control | 18 | -0.005 | 6 | 0.5226 |
