## P3: cross-dataset, mean ± SD over models and seeds
| Method | MSID → MSLD v2.0: bal. acc. | MSID → MSLD v2.0: spec. | MSID → MSLD v2.0: sens. | MSLD v2.0 → MSID: bal. acc. | MSLD v2.0 → MSID: spec. | MSLD v2.0 → MSID: sens. | Mean bal. acc. (both) | n runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ERM (standard fine-tuning) | 0.702 ± 0.047 | 0.476 ± 0.110 | 0.928 ± 0.030 | 0.847 ± 0.028 | 0.971 ± 0.015 | 0.723 ± 0.044 | 0.774 ± 0.084 | 18 |
| Random-negative control | 0.738 ± 0.029 | 0.601 ± 0.111 | 0.874 ± 0.067 | 0.849 ± 0.020 | 0.962 ± 0.015 | 0.736 ± 0.045 | 0.793 ± 0.062 | 18 |
| Atlas negatives, flat head | 0.750 ± 0.030 | 0.624 ± 0.093 | 0.876 ± 0.047 | 0.843 ± 0.017 | 0.972 ± 0.015 | 0.713 ± 0.031 | 0.796 ± 0.053 | 18 |
| Atlas, 5 conditions | 0.748 ± 0.004 | 0.592 ± 0.058 | 0.904 ± 0.055 |  |  |  | 0.748 ± 0.004 | 3 |
| Atlas, 20 conditions | 0.729 ± 0.023 | 0.548 ± 0.104 | 0.910 ± 0.058 |  |  |  | 0.729 ± 0.023 | 3 |
| HOD (Roy et al. 2022) | 0.770 ± 0.027 | 0.684 ± 0.084 | 0.857 ± 0.042 | 0.848 ± 0.024 | 0.965 ± 0.014 | 0.732 ± 0.052 | 0.809 ± 0.047 | 18 |
| FACTOR head, static control pool | 0.740 ± 0.032 | 0.566 ± 0.102 | 0.913 ± 0.048 | 0.850 ± 0.023 | 0.952 ± 0.019 | 0.747 ± 0.053 | 0.795 ± 0.063 | 18 |
| FACTOR without tone matching | 0.761 ± 0.034 | 0.616 ± 0.116 | 0.906 ± 0.053 | 0.843 ± 0.032 | 0.965 ± 0.015 | 0.721 ± 0.062 | 0.802 ± 0.053 | 18 |
| FACTOR (proposed) | 0.750 ± 0.035 | 0.607 ± 0.135 | 0.893 ± 0.076 | 0.847 ± 0.024 | 0.958 ± 0.023 | 0.737 ± 0.048 | 0.799 ± 0.058 | 18 |

## P3 balanced accuracy by backbone (mean over directions and seeds)
| method | efficientnet_b0 | resnet50 | vit_small_patch16_224 |
| --- | --- | --- | --- |
| erm | 0.748 | 0.762 | 0.813 |
| rand | 0.774 | 0.798 | 0.808 |
| atlas | 0.78 | 0.795 | 0.813 |
| div5 | nan | 0.748 | nan |
| div20 | nan | 0.729 | nan |
| hod | 0.817 | 0.779 | 0.831 |
| factor_static | 0.8 | 0.781 | 0.803 |
| factor_notone | 0.802 | 0.769 | 0.833 |
| factor | 0.79 | 0.778 | 0.827 |

## Skin tone (sensitivity: MSLD v2.0 → MSID; specificity: MSID → MSLD v2.0)
| Method | Sens. dark | Sens. light | Sens. gap (dark-light) | Worst-band sens. | Spec. dark | Spec. light | Spec. gap (dark-light) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ERM (standard fine-tuning) | 0.852 | 0.547 | +0.306 | 0.547 | 0.401 | 0.515 | -0.114 |
| Random-negative control | 0.849 | 0.577 | +0.272 | 0.577 | 0.546 | 0.629 | -0.083 |
| Atlas negatives, flat head | 0.839 | 0.536 | +0.303 | 0.536 | 0.556 | 0.656 | -0.101 |
| Atlas, 5 conditions | nan | nan | +nan | nan | 0.486 | 0.623 | -0.136 |
| Atlas, 20 conditions | nan | nan | +nan | nan | 0.476 | 0.567 | -0.091 |
| HOD (Roy et al. 2022) | 0.844 | 0.577 | +0.268 | 0.577 | 0.644 | 0.712 | -0.068 |
| FACTOR head, static control pool | 0.867 | 0.589 | +0.278 | 0.589 | 0.524 | 0.587 | -0.063 |
| FACTOR without tone matching | 0.840 | 0.563 | +0.277 | 0.562 | 0.559 | 0.647 | -0.088 |
| FACTOR (proposed) | 0.859 | 0.572 | +0.287 | 0.572 | 0.533 | 0.635 | -0.102 |

## % of MSLD v2.0 images called mpox by MSID-trained models
| method | chickenpox | cowpox | healthy | hfmd | measles | mpox |
| --- | --- | --- | --- | --- | --- | --- |
| erm | 73.4 | 94.8 | 24.4 | 47.2 | 41.0 | 92.8 |
| rand | 69.7 | 76.9 | 22.9 | 24.4 | 32.2 | 87.4 |
| atlas | 76.0 | 75.8 | 21.5 | 16.9 | 31.2 | 87.6 |
| div5 | 80.6 | 83.8 | 30.1 | 16.8 | 24.8 | 90.4 |
| div20 | 82.4 | 84.3 | 35.9 | 22.2 | 32.0 | 91.0 |
| hod | 64.6 | 69.7 | 15.6 | 14.8 | 20.5 | 85.7 |
| factor_static | 76.3 | 78.1 | 22.8 | 29.9 | 35.3 | 91.3 |
| factor_notone | 75.8 | 75.9 | 20.6 | 19.5 | 31.8 | 90.6 |
| factor | 71.6 | 72.2 | 20.5 | 23.5 | 38.1 | 89.3 |

## P2: patient-level, mean ± SD over models and folds
| Method | msid P2 bal. acc. | msld_v2 P2 bal. acc. |
| --- | --- | --- |
| ERM (standard fine-tuning) | 0.926 ± 0.018 | 0.798 ± 0.062 |
| FACTOR (proposed) | 0.915 ± 0.035 | 0.785 ± 0.061 |

## Paired Wilcoxon signed-rank tests (P3 balanced accuracy)
| Comparison | pairs | mean diff | wins | Wilcoxon p |
| --- | --- | --- | --- | --- |
| FACTOR (proposed) vs ERM (standard fine-tuning) | 18 | +0.024 | 13 | 0.0665 |
| FACTOR (proposed) vs Random-negative control | 18 | +0.005 | 9 | 0.9323 |
| FACTOR (proposed) vs Atlas negatives, flat head | 18 | +0.003 | 10 | 0.5798 |
| FACTOR (proposed) vs HOD (Roy et al. 2022) | 18 | -0.011 | 10 | 0.3038 |
| FACTOR (proposed) vs FACTOR without tone matching | 18 | -0.003 | 7 | 0.5798 |
| FACTOR (proposed) vs FACTOR head, static control pool | 18 | +0.004 | 11 | 0.8650 |
| Atlas negatives, flat head vs Random-negative control | 18 | +0.003 | 7 | 0.9323 |
| HOD (Roy et al. 2022) vs Atlas negatives, flat head | 18 | +0.013 | 11 | 0.1084 |
| Random-negative control vs ERM (standard fine-tuning) | 18 | +0.019 | 12 | 0.0665 |
