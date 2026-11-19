## P3: cross-dataset, mean ± SD over models and seeds
| Method | MSID → MSLD v2.0: bal. acc. | MSID → MSLD v2.0: spec. | MSID → MSLD v2.0: sens. | MSLD v2.0 → MSID: bal. acc. | MSLD v2.0 → MSID: spec. | MSLD v2.0 → MSID: sens. | Mean bal. acc. (both) | n runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ERM (standard fine-tuning) | 0.703 ± 0.039 | 0.481 ± 0.086 | 0.924 ± 0.026 | 0.859 ± 0.023 | 0.961 ± 0.023 | 0.758 ± 0.043 | 0.781 ± 0.086 | 18 |
| Colour jitter (hue/saturation) | 0.706 ± 0.041 | 0.493 ± 0.110 | 0.918 ± 0.040 | 0.863 ± 0.023 | 0.964 ± 0.015 | 0.762 ± 0.042 | 0.784 ± 0.087 | 18 |
| Mixup | 0.714 ± 0.043 | 0.516 ± 0.109 | 0.912 ± 0.042 | 0.837 ± 0.048 | 0.935 ± 0.054 | 0.738 ± 0.043 | 0.775 ± 0.077 | 18 |
| Group DRO | 0.691 ± 0.046 | 0.451 ± 0.112 | 0.932 ± 0.047 | 0.852 ± 0.030 | 0.951 ± 0.024 | 0.754 ± 0.051 | 0.772 ± 0.091 | 18 |
| Look-alike exposure only | 0.755 ± 0.023 | 0.632 ± 0.082 | 0.878 ± 0.044 | 0.849 ± 0.028 | 0.963 ± 0.024 | 0.735 ± 0.051 | 0.802 ± 0.054 | 18 |
| Tone balancing + tone shift only | 0.695 ± 0.043 | 0.512 ± 0.106 | 0.879 ± 0.064 | 0.846 ± 0.023 | 0.965 ± 0.014 | 0.728 ± 0.043 | 0.771 ± 0.085 | 18 |
| TALE (proposed) | 0.749 ± 0.030 | 0.641 ± 0.109 | 0.857 ± 0.070 | 0.837 ± 0.027 | 0.969 ± 0.020 | 0.704 ± 0.048 | 0.793 ± 0.053 | 18 |
## P3 balanced accuracy by backbone (mean over directions and seeds)
| method | efficientnet_b0 | resnet50 | vit_small_patch16_224 |
| --- | --- | --- | --- |
| erm | 0.758 | 0.768 | 0.817 |
| hsv | 0.761 | 0.781 | 0.81 |
| mixup | 0.727 | 0.778 | 0.821 |
| gdro | 0.754 | 0.751 | 0.81 |
| la | 0.789 | 0.793 | 0.824 |
| tone | 0.739 | 0.773 | 0.801 |
| tale | 0.768 | 0.794 | 0.817 |
## Skin tone (sensitivity: MSLD v2.0 → MSID; specificity: MSID → MSLD v2.0)
| Method | Sens. dark | Sens. light | Sens. gap (dark-light) | Worst-band sens. | Spec. dark | Spec. light | Spec. gap (dark-light) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ERM (standard fine-tuning) | 0.882 | 0.590 | +0.292 | 0.590 | 0.392 | 0.525 | -0.132 |
| Colour jitter (hue/saturation) | 0.890 | 0.589 | +0.301 | 0.589 | 0.397 | 0.543 | -0.146 |
| Mixup | 0.864 | 0.557 | +0.307 | 0.557 | 0.426 | 0.564 | -0.138 |
| Group DRO | 0.853 | 0.619 | +0.235 | 0.619 | 0.418 | 0.470 | -0.052 |
| Look-alike exposure only | 0.864 | 0.557 | +0.307 | 0.557 | 0.561 | 0.659 | -0.097 |
| Tone balancing + tone shift only | 0.849 | 0.563 | +0.286 | 0.563 | 0.458 | 0.541 | -0.083 |
| TALE (proposed) | 0.836 | 0.518 | +0.318 | 0.518 | 0.613 | 0.656 | -0.043 |
## % of MSLD v2.0 images called mpox by MSID-trained models
| method | chickenpox | cowpox | healthy | hfmd | measles | mpox |
| --- | --- | --- | --- | --- | --- | --- |
| erm | 73.4 | 95.6 | 25.2 | 45.3 | 39.0 | 92.4 |
| hsv | 71.3 | 92.9 | 26.3 | 43.7 | 37.7 | 91.8 |
| mixup | 69.4 | 91.6 | 24.5 | 40.2 | 37.3 | 91.2 |
| gdro | 75.7 | 93.1 | 28.1 | 50.9 | 42.7 | 93.2 |
| la | 68.8 | 79.3 | 23.8 | 16.1 | 27.0 | 87.8 |
| tone | 66.8 | 89.9 | 20.5 | 45.8 | 36.8 | 87.9 |
| tale | 65.9 | 76.1 | 19.6 | 18.6 | 28.8 | 85.7 |
## P2: patient-level, mean ± SD over models and folds
| Method | msid P2 bal. acc. | msld_v2 P2 bal. acc. |
| --- | --- | --- |
| ERM (standard fine-tuning) | 0.929 ± 0.025 | 0.800 ± 0.071 |
| TALE (proposed) | 0.904 ± 0.033 | 0.820 ± 0.042 |
## Paired Wilcoxon signed-rank tests (P3 balanced accuracy, both directions pooled)
| Comparison | pairs | mean diff | wins | Wilcoxon p |
| --- | --- | --- | --- | --- |
| TALE (proposed) vs ERM (standard fine-tuning) | 18 | +0.012 | 8 | 0.4951 |
| TALE (proposed) vs Colour jitter (hue/saturation) | 18 | +0.009 | 9 | 0.4171 |
| TALE (proposed) vs Mixup | 18 | +0.018 | 11 | 0.0304 |
| TALE (proposed) vs Group DRO | 18 | +0.021 | 11 | 0.1187 |
| TALE (proposed) vs Look-alike exposure only | 18 | -0.009 | 5 | 0.0385 |
| TALE (proposed) vs Tone balancing + tone shift only | 18 | +0.022 | 13 | 0.0483 |
| Look-alike exposure only vs ERM (standard fine-tuning) | 18 | +0.021 | 12 | 0.0665 |
| Tone balancing + tone shift only vs ERM (standard fine-tuning) | 18 | -0.010 | 7 | 0.2462 |

Source: Kaggle Version 3 "TALE method study", run 359635265, 18 Oct 2026, 186 runs, 08:43-14:15 UTC (331 min; 321 min training). Pool: 1,200 sampled, 5 hash duplicates removed, 0 embedding matches, 1,195 final (393 dark, 437 intermediate, 355 light, 10 unknown ITA; Fitzpatrick 1-6: 207/255/230/218/143/72, 70 unlabelled).
