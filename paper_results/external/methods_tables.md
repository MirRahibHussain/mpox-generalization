## P3: cross-dataset, mean ± SD over models and seeds
| Method | MSID → MSLD v2.0: bal. acc. | MSID → MSLD v2.0: spec. | MSID → MSLD v2.0: sens. | MSLD v2.0 → MSID: bal. acc. | MSLD v2.0 → MSID: spec. | MSLD v2.0 → MSID: sens. | Mean bal. acc. (both) | n runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ERM (standard fine-tuning) | 0.703 ± 0.047 | 0.475 ± 0.107 | 0.931 ± 0.029 | 0.852 ± 0.034 | 0.967 ± 0.010 | 0.738 ± 0.060 | 0.777 ± 0.087 | 18 |
| Random-negative control | 0.744 ± 0.033 | 0.616 ± 0.100 | 0.872 ± 0.055 | 0.847 ± 0.018 | 0.966 ± 0.015 | 0.728 ± 0.035 | 0.796 ± 0.059 | 18 |
| Atlas negatives, flat head | 0.749 ± 0.029 | 0.652 ± 0.121 | 0.846 ± 0.079 | 0.842 ± 0.018 | 0.971 ± 0.012 | 0.714 ± 0.034 | 0.795 ± 0.053 | 18 |
| HOD (Roy et al. 2022) | 0.772 ± 0.028 | 0.682 ± 0.088 | 0.862 ± 0.047 | 0.839 ± 0.021 | 0.968 ± 0.008 | 0.710 ± 0.041 | 0.806 ± 0.042 | 18 |
| FACTOR without tone matching | 0.755 ± 0.029 | 0.621 ± 0.131 | 0.889 ± 0.085 | 0.846 ± 0.035 | 0.958 ± 0.010 | 0.735 ± 0.072 | 0.801 ± 0.056 | 18 |
| FACTOR (proposed) | 0.745 ± 0.030 | 0.601 ± 0.123 | 0.889 ± 0.087 | 0.846 ± 0.022 | 0.957 ± 0.021 | 0.736 ± 0.044 | 0.796 ± 0.058 | 18 |

## P3 balanced accuracy by backbone (mean over directions and seeds)
| method | efficientnet_b0 | resnet50 | vit_small_patch16_224 |
| --- | --- | --- | --- |
| erm | 0.748 | 0.762 | 0.822 |
| rand | 0.774 | 0.798 | 0.814 |
| atlas | 0.78 | 0.795 | 0.811 |
| hod | 0.817 | 0.779 | 0.82 |
| factor_notone | 0.802 | 0.769 | 0.831 |
| factor | 0.79 | 0.778 | 0.819 |

## Skin tone (sensitivity: MSLD v2.0 → MSID; specificity: MSID → MSLD v2.0)
| Method | Sens. dark | Sens. light | Sens. gap (dark-light) | Worst-band sens. | Spec. dark | Spec. light | Spec. gap (dark-light) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ERM (standard fine-tuning) | 0.859 | 0.581 | +0.278 | 0.581 | 0.400 | 0.514 | -0.114 |
| Random-negative control | 0.849 | 0.563 | +0.286 | 0.563 | 0.553 | 0.651 | -0.097 |
| Atlas negatives, flat head | 0.841 | 0.529 | +0.313 | 0.529 | 0.584 | 0.679 | -0.095 |
| HOD (Roy et al. 2022) | 0.823 | 0.551 | +0.272 | 0.551 | 0.637 | 0.712 | -0.075 |
| FACTOR without tone matching | 0.850 | 0.581 | +0.269 | 0.580 | 0.560 | 0.647 | -0.087 |
| FACTOR (proposed) | 0.851 | 0.583 | +0.269 | 0.583 | 0.536 | 0.622 | -0.086 |

## % of MSLD v2.0 images called mpox by MSID-trained models
| method | chickenpox | cowpox | healthy | hfmd | measles | mpox |
| --- | --- | --- | --- | --- | --- | --- |
| erm | 74.3 | 94.4 | 24.5 | 47.1 | 41.0 | 93.1 |
| rand | 71.9 | 73.7 | 22.1 | 21.8 | 29.2 | 87.2 |
| atlas | 72.5 | 68.5 | 19.0 | 15.7 | 29.4 | 84.6 |
| hod | 64.6 | 70.9 | 15.8 | 14.9 | 19.8 | 86.2 |
| factor_notone | 73.3 | 73.6 | 21.2 | 19.9 | 30.9 | 88.9 |
| factor | 73.0 | 73.1 | 20.8 | 23.5 | 39.7 | 88.9 |

## Paired Wilcoxon signed-rank tests (P3 balanced accuracy)
| Comparison | pairs | mean diff | wins | Wilcoxon p |
| --- | --- | --- | --- | --- |
| FACTOR (proposed) vs ERM (standard fine-tuning) | 18 | +0.018 | 10 | 0.1540 |
| FACTOR without tone matching vs ERM (standard fine-tuning) | 18 | +0.024 | 13 | 0.0432 |
| Atlas negatives, flat head vs ERM (standard fine-tuning) | 18 | +0.018 | 9 | 0.2288 |
| HOD (Roy et al. 2022) vs ERM (standard fine-tuning) | 18 | +0.028 | 12 | 0.0539 |
| Random-negative control vs ERM (standard fine-tuning) | 18 | +0.018 | 11 | 0.1084 |
| FACTOR (proposed) vs Atlas negatives, flat head | 18 | +0.000 | 10 | 1.0000 |
| FACTOR (proposed) vs HOD (Roy et al. 2022) | 18 | -0.010 | 9 | 0.2837 |
| FACTOR (proposed) vs FACTOR without tone matching | 18 | -0.005 | 6 | 0.2121 |
| Atlas negatives, flat head vs Random-negative control | 18 | -0.000 | 7 | 0.7660 |
