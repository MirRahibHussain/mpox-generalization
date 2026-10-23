## Operating points (P3), mean ± SD over backbones and seeds

### MSID → MSLD v2.0

| method | AUROC | spec. @ 90% sens. (test ROC) | sens. @ 90% spec. (test ROC) | test sens. (val. thr. for 90% sens.) | test spec. (same thr.) | test bal. acc. (same thr.) | test bal. acc. (val. Youden thr.) | runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| atlas | 0.841 ± 0.031 | 0.597 ± 0.055 | 0.506 ± 0.072 | 0.870 ± 0.074 | 0.617 ± 0.123 | 0.744 ± 0.039 | 0.755 ± 0.029 | 9 |
| div20 | 0.826 ± 0.012 | 0.574 ± 0.008 | 0.450 ± 0.080 | 0.878 ± 0.080 | 0.587 ± 0.143 | 0.732 ± 0.032 | 0.741 ± 0.014 | 3 |
| div5 | 0.840 ± 0.007 | 0.619 ± 0.043 | 0.489 ± 0.049 | 0.899 ± 0.082 | 0.604 ± 0.148 | 0.751 ± 0.037 | 0.766 ± 0.010 | 3 |
| erm | 0.813 ± 0.039 | 0.549 ± 0.082 | 0.402 ± 0.167 | 0.873 ± 0.050 | 0.583 ± 0.106 | 0.728 ± 0.041 | 0.692 ± 0.041 | 9 |
| factor | 0.846 ± 0.027 | 0.612 ± 0.053 | 0.519 ± 0.084 | 0.842 ± 0.097 | 0.682 ± 0.107 | 0.762 ± 0.025 | 0.753 ± 0.033 | 9 |
| factor_notone | 0.852 ± 0.021 | 0.640 ± 0.053 | 0.516 ± 0.075 | 0.853 ± 0.098 | 0.661 ± 0.131 | 0.757 ± 0.027 | 0.750 ± 0.030 | 9 |
| factor_static | 0.838 ± 0.024 | 0.606 ± 0.060 | 0.495 ± 0.065 | 0.866 ± 0.060 | 0.644 ± 0.105 | 0.755 ± 0.031 | 0.749 ± 0.031 | 9 |
| hod | 0.859 ± 0.024 | 0.638 ± 0.053 | 0.558 ± 0.071 | 0.851 ± 0.099 | 0.651 ± 0.200 | 0.751 ± 0.056 | 0.757 ± 0.041 | 9 |
| rand | 0.831 ± 0.035 | 0.594 ± 0.078 | 0.490 ± 0.093 | 0.803 ± 0.089 | 0.677 ± 0.118 | 0.740 ± 0.032 | 0.738 ± 0.037 | 9 |

### MSLD v2.0 → MSID

| method | AUROC | spec. @ 90% sens. (test ROC) | sens. @ 90% spec. (test ROC) | test sens. (val. thr. for 90% sens.) | test spec. (same thr.) | test bal. acc. (same thr.) | test bal. acc. (val. Youden thr.) | runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| atlas | 0.933 ± 0.012 | 0.795 ± 0.070 | 0.850 ± 0.023 | 0.697 ± 0.064 | 0.968 ± 0.031 | 0.833 ± 0.022 | 0.833 ± 0.020 | 9 |
| erm | 0.938 ± 0.016 | 0.817 ± 0.067 | 0.856 ± 0.034 | 0.771 ± 0.076 | 0.942 ± 0.050 | 0.857 ± 0.023 | 0.849 ± 0.028 | 9 |
| factor | 0.920 ± 0.018 | 0.742 ± 0.086 | 0.831 ± 0.030 | 0.715 ± 0.071 | 0.960 ± 0.036 | 0.837 ± 0.029 | 0.837 ± 0.031 | 9 |
| factor_notone | 0.924 ± 0.016 | 0.769 ± 0.073 | 0.837 ± 0.027 | 0.707 ± 0.091 | 0.962 ± 0.036 | 0.834 ± 0.035 | 0.843 ± 0.024 | 9 |
| factor_static | 0.924 ± 0.015 | 0.751 ± 0.088 | 0.839 ± 0.032 | 0.740 ± 0.058 | 0.954 ± 0.029 | 0.847 ± 0.020 | 0.847 ± 0.029 | 9 |
| hod | 0.944 ± 0.015 | 0.833 ± 0.050 | 0.850 ± 0.034 | 0.689 ± 0.074 | 0.974 ± 0.019 | 0.831 ± 0.032 | 0.837 ± 0.027 | 9 |
| rand | 0.933 ± 0.012 | 0.798 ± 0.039 | 0.825 ± 0.023 | 0.698 ± 0.078 | 0.969 ± 0.025 | 0.833 ± 0.031 | 0.837 ± 0.028 | 9 |

## Paired Wilcoxon tests within each direction (Holm-adjusted within direction and metric)

| direction | metric | comparison | pairs | mean diff | wins | p | p (Holm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MSID → MSLD v2.0 | auroc | factor vs erm | 9 | +0.033 | 8 | 0.0078 | 0.0703 |
| MSID → MSLD v2.0 | auroc | factor vs rand | 9 | +0.016 | 4 | 0.7344 | 1.0000 |
| MSID → MSLD v2.0 | auroc | factor vs atlas | 9 | +0.005 | 4 | 0.6523 | 1.0000 |
| MSID → MSLD v2.0 | auroc | factor vs hod | 9 | -0.012 | 1 | 0.0391 | 0.3125 |
| MSID → MSLD v2.0 | auroc | factor vs factor_notone | 9 | -0.006 | 4 | 0.3594 | 1.0000 |
| MSID → MSLD v2.0 | auroc | factor vs factor_static | 9 | +0.008 | 6 | 0.6523 | 1.0000 |
| MSID → MSLD v2.0 | auroc | atlas vs rand | 9 | +0.010 | 6 | 0.4258 | 1.0000 |
| MSID → MSLD v2.0 | auroc | hod vs atlas | 9 | +0.018 | 6 | 0.1641 | 0.9844 |
| MSID → MSLD v2.0 | auroc | rand vs erm | 9 | +0.018 | 6 | 0.1289 | 0.9023 |
| MSID → MSLD v2.0 | test_bal_acc_val90 | factor vs erm | 9 | +0.034 | 7 | 0.0273 | 0.2461 |
| MSID → MSLD v2.0 | test_bal_acc_val90 | factor vs rand | 9 | +0.022 | 7 | 0.2500 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_val90 | factor vs atlas | 9 | +0.018 | 6 | 0.1641 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_val90 | factor vs hod | 9 | +0.011 | 4 | 0.8203 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_val90 | factor vs factor_notone | 9 | +0.005 | 4 | 0.6523 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_val90 | factor vs factor_static | 9 | +0.007 | 3 | 0.9102 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_val90 | atlas vs rand | 9 | +0.003 | 4 | 0.9102 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_val90 | hod vs atlas | 9 | +0.007 | 6 | 0.6523 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_val90 | rand vs erm | 9 | +0.012 | 5 | 0.5703 | 1.0000 |
| MSID → MSLD v2.0 | test_spec_val90 | factor vs erm | 9 | +0.099 | 8 | 0.0273 | 0.2461 |
| MSID → MSLD v2.0 | test_spec_val90 | factor vs rand | 9 | +0.005 | 4 | 1.0000 | 1.0000 |
| MSID → MSLD v2.0 | test_spec_val90 | factor vs atlas | 9 | +0.065 | 5 | 0.1719 | 1.0000 |
| MSID → MSLD v2.0 | test_spec_val90 | factor vs hod | 9 | +0.031 | 5 | 0.4766 | 1.0000 |
| MSID → MSLD v2.0 | test_spec_val90 | factor vs factor_notone | 9 | +0.021 | 5 | 0.4609 | 1.0000 |
| MSID → MSLD v2.0 | test_spec_val90 | factor vs factor_static | 9 | +0.038 | 5 | 0.4453 | 1.0000 |
| MSID → MSLD v2.0 | test_spec_val90 | atlas vs rand | 9 | -0.060 | 2 | 0.1289 | 0.9023 |
| MSID → MSLD v2.0 | test_spec_val90 | hod vs atlas | 9 | +0.034 | 7 | 0.4258 | 1.0000 |
| MSID → MSLD v2.0 | test_spec_val90 | rand vs erm | 9 | +0.094 | 6 | 0.0742 | 0.5938 |
| MSID → MSLD v2.0 | test_bal_acc_youden | factor vs erm | 9 | +0.061 | 8 | 0.0078 | 0.0703 |
| MSID → MSLD v2.0 | test_bal_acc_youden | factor vs rand | 9 | +0.016 | 6 | 0.4258 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_youden | factor vs atlas | 9 | -0.002 | 5 | 1.0000 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_youden | factor vs hod | 9 | -0.004 | 4 | 0.8203 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_youden | factor vs factor_notone | 9 | +0.004 | 4 | 0.9102 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_youden | factor vs factor_static | 9 | +0.004 | 4 | 0.9102 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_youden | atlas vs rand | 9 | +0.017 | 6 | 0.3594 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_youden | hod vs atlas | 9 | +0.002 | 6 | 0.4258 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_youden | rand vs erm | 9 | +0.045 | 7 | 0.0273 | 0.2188 |
| MSLD v2.0 → MSID | auroc | factor vs erm | 9 | -0.017 | 2 | 0.0195 | 0.1562 |
| MSLD v2.0 → MSID | auroc | factor vs rand | 9 | -0.013 | 3 | 0.0977 | 0.6836 |
| MSLD v2.0 → MSID | auroc | factor vs atlas | 9 | -0.013 | 2 | 0.1641 | 0.8203 |
| MSLD v2.0 → MSID | auroc | factor vs hod | 9 | -0.024 | 0 | 0.0039 | 0.0352 |
| MSLD v2.0 → MSID | auroc | factor vs factor_notone | 9 | -0.003 | 5 | 0.7344 | 1.0000 |
| MSLD v2.0 → MSID | auroc | factor vs factor_static | 9 | -0.004 | 3 | 0.4961 | 1.0000 |
| MSLD v2.0 → MSID | auroc | atlas vs rand | 9 | +0.000 | 5 | 0.9102 | 1.0000 |
| MSLD v2.0 → MSID | auroc | hod vs atlas | 9 | +0.011 | 7 | 0.1289 | 0.7734 |
| MSLD v2.0 → MSID | auroc | rand vs erm | 9 | -0.005 | 3 | 0.2500 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_val90 | factor vs erm | 9 | -0.020 | 2 | 0.0977 | 0.7812 |
| MSLD v2.0 → MSID | test_bal_acc_val90 | factor vs rand | 9 | +0.004 | 6 | 0.6523 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_val90 | factor vs atlas | 9 | +0.005 | 6 | 0.5703 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_val90 | factor vs hod | 9 | +0.006 | 5 | 0.8203 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_val90 | factor vs factor_notone | 9 | +0.003 | 4 | 0.8203 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_val90 | factor vs factor_static | 9 | -0.010 | 3 | 0.4258 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_val90 | atlas vs rand | 9 | -0.001 | 5 | 0.9102 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_val90 | hod vs atlas | 9 | -0.001 | 5 | 0.9102 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_val90 | rand vs erm | 9 | -0.024 | 2 | 0.0547 | 0.4922 |
| MSLD v2.0 → MSID | test_spec_val90 | factor vs erm | 9 | +0.017 | 4 | 0.4766 | 1.0000 |
| MSLD v2.0 → MSID | test_spec_val90 | factor vs rand | 9 | -0.009 | 3 | 0.2500 | 1.0000 |
| MSLD v2.0 → MSID | test_spec_val90 | factor vs atlas | 9 | -0.009 | 3 | 0.3281 | 1.0000 |
| MSLD v2.0 → MSID | test_spec_val90 | factor vs hod | 9 | -0.014 | 3 | 0.1562 | 1.0000 |
| MSLD v2.0 → MSID | test_spec_val90 | factor vs factor_notone | 9 | -0.002 | 5 | 0.9453 | 1.0000 |
| MSLD v2.0 → MSID | test_spec_val90 | factor vs factor_static | 9 | +0.006 | 6 | 0.4961 | 1.0000 |
| MSLD v2.0 → MSID | test_spec_val90 | atlas vs rand | 9 | -0.000 | 5 | 0.9453 | 1.0000 |
| MSLD v2.0 → MSID | test_spec_val90 | hod vs atlas | 9 | +0.005 | 5 | 0.7148 | 1.0000 |
| MSLD v2.0 → MSID | test_spec_val90 | rand vs erm | 9 | +0.026 | 5 | 0.1484 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_youden | factor vs erm | 9 | -0.012 | 2 | 0.2500 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_youden | factor vs rand | 9 | +0.000 | 5 | 0.9102 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_youden | factor vs atlas | 9 | +0.004 | 5 | 0.6523 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_youden | factor vs hod | 9 | -0.000 | 5 | 1.0000 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_youden | factor vs factor_notone | 9 | -0.006 | 4 | 0.7344 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_youden | factor vs factor_static | 9 | -0.010 | 3 | 0.4258 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_youden | atlas vs rand | 9 | -0.004 | 3 | 0.5703 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_youden | hod vs atlas | 9 | +0.005 | 5 | 0.8203 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_youden | rand vs erm | 9 | -0.012 | 2 | 0.1641 | 1.0000 |
