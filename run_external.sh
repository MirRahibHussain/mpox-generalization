#!/usr/bin/env bash
# External validation on a third public dataset (MCSI) in one command (Kaggle GPU notebook, Internet on):
#   !git clone -q https://github.com/MirRahibHussain/mpox-generalization.git && bash mpox-generalization/run_external.sh
# The analysis plan was fixed before any model saw MCSI: PREREGISTRATION_MCSI.md.
# MCSI (Campana et al., Zenodo 8360076, CC BY 4.0) is downloaded here; a crop-aware duplicate
# audit (hash, embedding, ORB + RANSAC) removes every MCSI image that copies an MSLD v2.0 or
# MSID image. The six pre-specified methods are trained on the same P3 training splits as before
# and every model predicts its usual cross-dataset test set and the clean MCSI set.
set -euo pipefail
cd "$(dirname "$0")"

INPUT=${INPUT:-/kaggle/input}
OUT=${OUT:-/kaggle/working}
WORKERS=${WORKERS:-4}
MAX_HOURS=${MAX_HOURS:-10.5}
METHODS=${METHODS:-erm factor atlas hod rand factor_notone}
CACHE=${CACHE:-/tmp/mpoxgen_img288}
RES=results_external
FITZ_CSV_URL=${FITZ_CSV_URL:-https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv}
PAIRS="factor:erm factor_notone:erm atlas:erm hod:erm rand:erm factor:atlas factor:hod factor:factor_notone atlas:rand"
EXTPAIRS="rand:erm atlas:erm hod:erm factor_notone:erm factor:erm"
FPAIRS="factor:atlas factor:hod factor:factor_notone factor:rand"
export PIPELINE_START=$(date +%s)
export MPOXGEN_DETERMINISTIC=1
mkdir -p "$OUT"
exec > >(tee "$OUT/run_log_external.txt") 2>&1
stamp() { echo "== [$(date -u +%H:%M:%S) UTC, +$(( ($(date +%s) - PIPELINE_START) / 60 )) min] $*"; }

stamp "environment"
python -c "import torch, timm, cv2; print('torch', torch.__version__, '| timm', timm.__version__, '| cv2', cv2.__version__, '| cuda', torch.cuda.is_available())"
git log -1 --format="code version: %h %s" || true

stamp "0. locate datasets"
eval "$(python scripts/00_find_datasets.py --input "$INPUT")"
echo "MSLD=$MSLD"; echo "MSID=$MSID"; echo "FITZ=$FITZ"
[ -n "$FITZ" ] || { echo "Fitzpatrick17k not found under $INPUT - add it as notebook input"; exit 1; }

stamp "1-4. index, embeddings, duplicate audit + ITA, splits (same as before)"
python scripts/01_build_index.py --data "msld_v2=$MSLD" --data "msid=$MSID" --out data/index
python scripts/02_embedding_pairs.py --index data/index
python scripts/03_duplicate_audit.py --index data/index
python scripts/04_make_splits.py --index data/index --out splits

stamp "18. MCSI: download, index, crop-aware duplicate audit against MSLD v2.0 and MSID"
python scripts/18_external_mcsi.py --index data/index --out data/external_mcsi

stamp "8. control pool (for 'rand')"
wget -q -O fitzpatrick17k.csv "$FITZ_CSV_URL"
python scripts/08_lookalike_pool.py --fitz "$FITZ" --csv fitzpatrick17k.csv --index data/index --out data/pool \
    --control-out data/pool_control

stamp "15. whole-atlas pool"
python scripts/15_atlas_pool.py --fitz "$FITZ" --csv fitzpatrick17k.csv --index data/index --out data/pool_atlas \
    --workers "$WORKERS"

stamp "9a. smoke test (1 epoch, with the external test set)"
python scripts/09_run_methods.py --smoke --methods erm factor hod --p2-methods \
    --external mcsi=data/external_mcsi/images_clean.csv --workers "$WORKERS" --cache-dir "$CACHE"

stamp "9b. pre-specified methods on the P3 training splits; every model also predicts clean MCSI"
python scripts/09_run_methods.py --methods $METHODS --p2-methods --order split --out "$RES" \
    --external mcsi=data/external_mcsi/images_clean.csv --workers "$WORKERS" --cache-dir "$CACHE" \
    --max-hours "$MAX_HOURS"

stamp "10-13, 17, 19. summaries and statistics"
python scripts/10_summarize_methods.py --results "$RES" --index data/index --pairs $PAIRS || echo "summary failed"
python scripts/13_paired_stats.py --runs "$RES/all_runs.csv" --pairs $PAIRS \
    --metrics balanced_accuracy auroc mpox_specificity || echo "paired stats failed"
python scripts/17_open_set.py --results "$RES" --index data/index || echo "open-set analysis failed"
python scripts/19_external_eval.py --results "$RES" --external data/external_mcsi/images_clean.csv || echo "external eval failed"
# Pre-registered families (PREREGISTRATION_MCSI.md): H1-H3 each method vs ERM; H4 FACTOR vs the others.
python scripts/13_paired_stats.py --runs "$RES/external_runs.csv" --pairs $EXTPAIRS --out "$RES/external_paired_vs_erm" \
    --metrics balanced_accuracy called_acne shortcut_gap auroc || echo "external stats (H1-H3) failed"
python scripts/13_paired_stats.py --runs "$RES/external_runs.csv" --pairs $FPAIRS --out "$RES/external_paired_factor" \
    --metrics balanced_accuracy auroc called_acne || echo "external stats (H4) failed"

stamp "package"
KEY="$OUT/key_results_external"
mkdir -p "$KEY"
cp "$RES"/*.csv "$RES"/*.md "$RES"/*.png "$KEY/" 2>/dev/null || true
cp data/external_mcsi/report.txt "$KEY/mcsi_report.txt" 2>/dev/null || true
cp data/external_mcsi/pairs.csv "$KEY/mcsi_pairs.csv" 2>/dev/null || true
cp data/external_mcsi/images.csv "$KEY/mcsi_images.csv" 2>/dev/null || true
cp data/external_mcsi/duplicates_review.png "$KEY/mcsi_duplicates_review.png" 2>/dev/null || true
cp PREREGISTRATION_MCSI.md "$KEY/" 2>/dev/null || true
zip -qr "$OUT/outputs_external.zip" "$RES" data/external_mcsi/images.csv data/external_mcsi/pairs.csv \
    data/external_mcsi/report.txt splits -x "*.pt" "*.npy" || true
stamp "DONE. Key files in $KEY"
