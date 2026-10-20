## Operating points (P3), mean ± SD over backbones and seeds

### MSID → MSLD v2.0

| method | AUROC | spec. @ 90% sens. (test ROC) | sens. @ 90% spec. (test ROC) | test sens. (val. thr. for 90% sens.) | test spec. (same thr.) | test bal. acc. (same thr.) | test bal. acc. (val. Youden thr.) | runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| erm | 0.811 ± 0.036 | 0.549 ± 0.077 | 0.398 ± 0.164 | 0.890 ± 0.054 | 0.540 ± 0.125 | 0.715 ± 0.044 | 0.695 ± 0.042 | 9 |
| la | 0.836 ± 0.030 | 0.587 ± 0.084 | 0.511 ± 0.065 | 0.853 ± 0.071 | 0.637 ± 0.137 | 0.745 ± 0.046 | 0.749 ± 0.031 | 9 |
| rand | 0.845 ± 0.023 | 0.598 ± 0.059 | 0.529 ± 0.060 | 0.824 ± 0.089 | 0.677 ± 0.110 | 0.750 ± 0.024 | 0.750 ± 0.023 | 9 |
| tale | 0.842 ± 0.025 | 0.596 ± 0.051 | 0.520 ± 0.085 | 0.866 ± 0.060 | 0.623 ± 0.125 | 0.744 ± 0.034 | 0.754 ± 0.019 | 9 |

### MSLD v2.0 → MSID

| method | AUROC | spec. @ 90% sens. (test ROC) | sens. @ 90% spec. (test ROC) | test sens. (val. thr. for 90% sens.) | test spec. (same thr.) | test bal. acc. (same thr.) | test bal. acc. (val. Youden thr.) | runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| erm | 0.937 ± 0.013 | 0.824 ± 0.061 | 0.846 ± 0.048 | 0.767 ± 0.085 | 0.942 ± 0.061 | 0.855 ± 0.029 | 0.851 ± 0.032 | 9 |
| la | 0.925 ± 0.023 | 0.762 ± 0.090 | 0.829 ± 0.049 | 0.733 ± 0.059 | 0.954 ± 0.045 | 0.844 ± 0.024 | 0.847 ± 0.021 | 9 |
| rand | 0.936 ± 0.013 | 0.809 ± 0.062 | 0.852 ± 0.024 | 0.727 ± 0.063 | 0.961 ± 0.024 | 0.844 ± 0.024 | 0.838 ± 0.024 | 9 |
| tale | 0.923 ± 0.023 | 0.740 ± 0.115 | 0.816 ± 0.036 | 0.714 ± 0.056 | 0.963 ± 0.026 | 0.838 ± 0.024 | 0.821 ± 0.034 | 9 |

## Paired Wilcoxon tests within each direction (Holm-adjusted within direction and metric)

| direction | metric | comparison | pairs | mean diff | wins | p | p (Holm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MSID → MSLD v2.0 | auroc | la vs erm | 9 | +0.024 | 7 | 0.0547 | 0.2188 |
| MSID → MSLD v2.0 | auroc | rand vs erm | 9 | +0.034 | 8 | 0.0195 | 0.0977 |
| MSID → MSLD v2.0 | auroc | tale vs erm | 9 | +0.031 | 9 | 0.0039 | 0.0234 |
| MSID → MSLD v2.0 | auroc | la vs rand | 9 | -0.010 | 4 | 0.3008 | 0.9023 |
| MSID → MSLD v2.0 | auroc | tale vs rand | 9 | -0.003 | 3 | 0.6523 | 1.0000 |
| MSID → MSLD v2.0 | auroc | tale vs la | 9 | +0.007 | 5 | 0.8203 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_val90 | la vs erm | 9 | +0.030 | 8 | 0.0977 | 0.4883 |
| MSID → MSLD v2.0 | test_bal_acc_val90 | rand vs erm | 9 | +0.035 | 8 | 0.0391 | 0.2344 |
| MSID → MSLD v2.0 | test_bal_acc_val90 | tale vs erm | 9 | +0.030 | 6 | 0.1641 | 0.6562 |
| MSID → MSLD v2.0 | test_bal_acc_val90 | la vs rand | 9 | -0.005 | 5 | 1.0000 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_val90 | tale vs rand | 9 | -0.006 | 4 | 0.7344 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_val90 | tale vs la | 9 | -0.001 | 3 | 0.6523 | 1.0000 |
| MSID → MSLD v2.0 | test_spec_val90 | la vs erm | 9 | +0.098 | 7 | 0.0547 | 0.2734 |
| MSID → MSLD v2.0 | test_spec_val90 | rand vs erm | 9 | +0.137 | 9 | 0.0039 | 0.0234 |
| MSID → MSLD v2.0 | test_spec_val90 | tale vs erm | 9 | +0.083 | 7 | 0.1641 | 0.6562 |
| MSID → MSLD v2.0 | test_spec_val90 | la vs rand | 9 | -0.039 | 3 | 0.2500 | 0.7500 |
| MSID → MSLD v2.0 | test_spec_val90 | tale vs rand | 9 | -0.054 | 3 | 0.3711 | 0.7500 |
| MSID → MSLD v2.0 | test_spec_val90 | tale vs la | 9 | -0.014 | 3 | 0.5703 | 0.7500 |
| MSID → MSLD v2.0 | test_bal_acc_youden | la vs erm | 9 | +0.054 | 8 | 0.0117 | 0.0586 |
| MSID → MSLD v2.0 | test_bal_acc_youden | rand vs erm | 9 | +0.055 | 8 | 0.0273 | 0.1094 |
| MSID → MSLD v2.0 | test_bal_acc_youden | tale vs erm | 9 | +0.059 | 8 | 0.0078 | 0.0469 |
| MSID → MSLD v2.0 | test_bal_acc_youden | la vs rand | 9 | -0.002 | 6 | 0.8203 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_youden | tale vs rand | 9 | +0.004 | 4 | 0.9102 | 1.0000 |
| MSID → MSLD v2.0 | test_bal_acc_youden | tale vs la | 9 | +0.006 | 4 | 1.0000 | 1.0000 |
| MSLD v2.0 → MSID | auroc | la vs erm | 9 | -0.012 | 1 | 0.0547 | 0.2188 |
| MSLD v2.0 → MSID | auroc | rand vs erm | 9 | -0.001 | 4 | 0.8203 | 1.0000 |
| MSLD v2.0 → MSID | auroc | tale vs erm | 9 | -0.014 | 2 | 0.0195 | 0.1172 |
| MSLD v2.0 → MSID | auroc | la vs rand | 9 | -0.011 | 2 | 0.0742 | 0.2227 |
| MSLD v2.0 → MSID | auroc | tale vs rand | 9 | -0.013 | 1 | 0.0273 | 0.1367 |
| MSLD v2.0 → MSID | auroc | tale vs la | 9 | -0.002 | 5 | 1.0000 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_val90 | la vs erm | 9 | -0.011 | 2 | 0.2500 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_val90 | rand vs erm | 9 | -0.010 | 4 | 0.4258 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_val90 | tale vs erm | 9 | -0.016 | 2 | 0.0391 | 0.2344 |
| MSLD v2.0 → MSID | test_bal_acc_val90 | la vs rand | 9 | -0.001 | 3 | 0.9102 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_val90 | tale vs rand | 9 | -0.006 | 4 | 0.6523 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_val90 | tale vs la | 9 | -0.006 | 3 | 0.3594 | 1.0000 |
| MSLD v2.0 → MSID | test_spec_val90 | la vs erm | 9 | +0.012 | 3 | 0.9453 | 1.0000 |
| MSLD v2.0 → MSID | test_spec_val90 | rand vs erm | 9 | +0.019 | 5 | 0.7344 | 1.0000 |
| MSLD v2.0 → MSID | test_spec_val90 | tale vs erm | 9 | +0.021 | 5 | 0.6406 | 1.0000 |
| MSLD v2.0 → MSID | test_spec_val90 | la vs rand | 9 | -0.007 | 4 | 0.5156 | 1.0000 |
| MSLD v2.0 → MSID | test_spec_val90 | tale vs rand | 9 | +0.001 | 3 | 0.7969 | 1.0000 |
| MSLD v2.0 → MSID | test_spec_val90 | tale vs la | 9 | +0.008 | 5 | 0.4336 | 1.0000 |
| MSLD v2.0 → MSID | test_bal_acc_youden | la vs erm | 9 | -0.005 | 3 | 0.6328 | 0.8516 |
| MSLD v2.0 → MSID | test_bal_acc_youden | rand vs erm | 9 | -0.013 | 2 | 0.2500 | 0.8125 |
| MSLD v2.0 → MSID | test_bal_acc_youden | tale vs erm | 9 | -0.030 | 2 | 0.0195 | 0.0977 |
| MSLD v2.0 → MSID | test_bal_acc_youden | la vs rand | 9 | +0.009 | 4 | 0.4258 | 0.8516 |
| MSLD v2.0 → MSID | test_bal_acc_youden | tale vs rand | 9 | -0.017 | 3 | 0.2031 | 0.8125 |
| MSLD v2.0 → MSID | test_bal_acc_youden | tale vs la | 9 | -0.026 | 1 | 0.0078 | 0.0469 |
