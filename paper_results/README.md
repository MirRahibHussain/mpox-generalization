# Paper results

Tables used in the paper, taken from Kaggle Version 2 (run 358186751, 14 Oct 2026,
Tesla T4, 155 min, exit 0). They were computed from the notebook output
(`results/*/predictions.csv`, `data/index/images.csv`, `splits/manifest.csv`, `results/all_runs.csv`).

| File | Content |
| --- | --- |
| `summary_binary.csv` | balanced accuracy (mean and sd), AUROC, mpox sensitivity and specificity per model, protocol and direction |
| `ita_counts.csv` | images per skin-tone band (ITA), dataset and class |
| `p3_skin_tone.csv` | cross-dataset mpox sensitivity and specificity per skin-tone band |
| `gradcam_ratings.csv` | where the Grad-CAM peak falls for 93 cross-dataset test images |
| `fig1_protocols.png` | balanced accuracy by protocol, mean ± sd over repeats (the data of Figure 1, redrawn for the web) |

## Grad-CAM ratings

Maps come from `scripts/07_gradcam.py` (ResNet-50 and ConvNeXt-Tiny, P3, seed 0,
up to six each of TP, FN, FP and TN per model and direction). For each map the
hottest (red) region was rated:

- `lesion`: on a visible lesion or on lesion-bearing skin
- `off-lesion`: on other content (background, hands, eyes, hair, censor bar)
- `mixed`: split between the two
- `n/a (healthy)`: healthy image, no lesion to compare against

The ratings were made by an AI assistant (Claude, Anthropic) from the overlay
images alone, without lesion masks or dermatologist review. Treat them as
indicative. The `note` column says what the peak covered.
