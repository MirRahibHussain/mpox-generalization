# Pre-registered analysis plan: external test on MCSI

Written 20 October 2026, 10:46 UTC, and committed to this repository before any model was trained
or evaluated on MCSI. The commit time on GitHub is the record of when the plan was fixed.

## What had been seen when this plan was written

- All results of the evaluation study, the TALE method study and the control study (MSLD v2.0 and
  MSID only).
- Part of the FACTOR study (Kaggle Version 5, still running): all cross-dataset runs from MSID to
  MSLD v2.0 and most runs from MSLD v2.0 to MSID. In the MSID to MSLD v2.0 direction, mean balanced
  accuracy over 9 runs was ERM 0.702, random-negative control 0.738, atlas with a flat head 0.749,
  FACTOR 0.750, FACTOR without tone matching 0.761 and HOD 0.770. FACTOR was therefore not better
  than the simpler atlas head, and HOD was best.
- No MCSI image had been loaded by any code. Only MCSI's file list, README, metadata.csv and
  skin_tone_labels.csv were read, to write the indexer.

## Data

MCSI (Mpox Close Skin Images; Campana et al., Pervasive and Mobile Computing 98:101874, 2024;
Zenodo record 8360076, version 2, CC BY 4.0, md5 e8e490856d0a7e4871ae9e2dc15e81fe): 400 images,
100 each of mpox, chickenpox, acne (three severity levels, from ACNE04) and healthy skin. Its mpox
and healthy images come from the first MSLD release, which MSLD v2.0 extends; the images were
cropped and zoomed onto the lesion.

Cleaning rule (scripts/18_external_mcsi.py), fixed before seeing any result: an MCSI image is
removed if it duplicates any MSLD v2.0 or MSID image by pHash and dHash within 6 bits (mirrored
copies included), ResNet-50 embedding cosine >= 0.92, or >= 25 RANSAC inliers of ORB matches against
its 10 nearest study images. The same clean set is used for models trained on either dataset. We
report how many images each rule removed, and the review sheet of flagged pairs. The thresholds will
not be changed after seeing model results. If visual review shows a flagged pair that is clearly not
the same photograph, or an unflagged pair that clearly is, we will report it and report results both
with the pre-specified set and with the corrected set.

## Models and methods (fixed)

Backbones ResNet-50, EfficientNet-B0, ViT-Small/16; seeds 0, 1, 2; the P3 training and validation
splits of MSID and of MSLD v2.0 (the same as in the FACTOR study); training exactly as in the FACTOR
study (deterministic cuDNN). Methods: ERM, random-negative control (rand), atlas with a flat head
(atlas), HOD, FACTOR without tone matching (factor_notone) and FACTOR. Decision threshold 0.5
(argmax). No method, threshold or hyper-parameter is tuned on MCSI.

## Hypotheses

Primary analyses use models trained on MSID, whose training data share no source with MCSI.
Models trained on MSLD v2.0 are secondary, because MCSI's mpox and healthy images come from the
earlier MSLD release (duplicates are removed, but the source is shared).

- **H1.** Each method that adds diseased-skin negatives (rand, atlas, HOD, factor_notone, FACTOR)
  reaches higher balanced accuracy on clean MCSI than ERM.
- **H2.** ERM calls a larger share of acne images mpox than each of these methods. (Acne is not a
  class of MSLD v2.0 or MSID. Note that the Fitzpatrick17k atlas and the control pool contain acne
  conditions, from another source, so for the atlas methods acne is a condition they have seen.)
- **H3 (disease-presence shortcut).** For ERM, the AUROC of the mpox score for mpox against healthy
  skin exceeds its AUROC for mpox against other diseases (chickenpox, acne), and each
  diseased-negative method has a smaller gap than ERM.
- **H4 (no directional prediction).** FACTOR compared with atlas, HOD, factor_notone and rand.
  Reported whatever the result.

## Statistics

Within each training dataset, methods are compared on the 9 matched (backbone, seed) pairs:
mean paired difference with a 95% percentile-bootstrap interval (10,000 resamples), wins/losses and
the exact two-sided Wilcoxon signed-rank test (smallest possible p = 0.0039). H1, H2 and H3 each
form one family of 5 comparisons against ERM, and H4 one family of 4, corrected with Holm's method
within training dataset and metric (scripts/13_paired_stats.py). The first part of H3 (ERM's gap
> 0) is tested with a one-sample exact Wilcoxon signed-rank test on the 9 ERM runs. A hypothesis counts as supported
for a method if its Holm-adjusted p < 0.05 in the predicted direction. Secondary, unadjusted:
AUROC, sensitivity, specificity, per-class share called mpox, sensitivity and specificity per ITA
band, and the two factor AUROCs of FACTOR.

## Reporting

Every number above is reported for every method, in both training directions, in the paper and in
paper_results/, whether or not a hypothesis is supported.

## Amendment 1 (20 October 2026, 11:36 UTC, before any model was trained on or scored against MCSI)

The first external run (Kaggle Version 6, started 10:57 UTC) was stopped during data preparation,
before any training began, for two reasons found in its log:

1. **Bug: the 100 acne images were not loaded.** metadata.csv lists their file names without the
   ".png" extension, so the indexer skipped them. The indexer now adds the missing extension and
   stops with an error if any listed image is not found. No analysis choice changes.
2. **The duplicate audit removed 94 of the 100 mpox images** (copies of MSLD v2.0 images: 62;
   of MSID images: 66; most found only by the ORB check, i.e. cropped copies), 6 of 100 chickenpox
   and 6 of 99 healthy images. The pre-registered clean set therefore holds only 6 mpox images,
   which makes sensitivity, balanced accuracy and the AUROCs on it very imprecise. The primary
   analysis stays exactly as registered. We add one **secondary** analysis, fixed now: each model is
   also scored on a direction-specific clean set that removes only the copies of images in its own
   training dataset (MSID-trained models keep the MCSI mpox images that copy MSLD v2.0 images, and
   the other way round). These models never saw those images, but the images are not independent
   of the other study dataset, so these results are reported as secondary and labelled as such.
   Models therefore predict every MCSI image, and scripts/19_external_eval.py selects the set.
   Hypotheses H1-H4, methods, thresholds and statistics are unchanged and are applied to both sets.

The size of the overlap is itself reported as a result of the duplicate audit: the mpox class of a
third public mpox dataset consists mostly of images also present, often cropped, in MSLD v2.0 and
MSID.
