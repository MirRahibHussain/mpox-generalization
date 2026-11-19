# Paper results

| Folder | Study | Kaggle run |
| --- | --- | --- |
| this folder | evaluation study: 6 backbones × P1/P2/P3 (156 trainings) | Version 2, run 358186751, 14 Oct 2026 |
| `tale/` | TALE method study (186 trainings) | Version 3, 18 Oct 2026 |
| `control/` | random-negative control, thresholds, tone-shift check (72 trainings) | Version 4, 19 Oct 2026 |
| `factor/` | FACTOR study: atlas exposure, factorized head, HOD, tone matching, condition diversity, ITA check, open-set analysis (192 trainings) | Version 5, run 360301520, 20 Oct 2026 |
| `external/` | pre-registered external test on MCSI (108 trainings) | Version 7, run 360404890, 20 Oct 2026 |

Tables of the evaluation study, taken from Kaggle Version 2 (run 358186751, 14 Oct 2026,
Tesla T4, 155 min, exit 0). They were computed from the notebook output
(`results/*/predictions.csv`, `data/index/images.csv`, `splits/manifest.csv`, `results/all_runs.csv`).

| File | Content |
| --- | --- |
| `summary_binary.csv` | balanced accuracy (mean and sd), AUROC, mpox sensitivity and specificity per model, protocol and direction |
| `ita_counts.csv` | images per skin-tone band (ITA), dataset and class |
| `p3_skin_tone.csv` | cross-dataset mpox sensitivity and specificity per skin-tone band |
| `run_log_evaluation.txt` | full run log of Kaggle Version 2 (run 358186751) |
| `revision_stats.txt` | output of `scripts/20_revision_stats.py`: statistics added in the journal revision, recomputed from the released files |
| `fig1_protocols.png` | balanced accuracy by protocol, mean ± sd over repeats (the data of Figure 2 of the paper, redrawn for the web) |
