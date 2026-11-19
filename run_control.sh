#!/usr/bin/env bash
# Control study in one command (Kaggle GPU notebook with Internet on):
#   !git clone -q https://github.com/MirRahibHussain/mpox-generalization.git && bash mpox-generalization/run_control.sh
# Re-runs ERM, look-alike exposure and TALE next to a random-negative control
# (same number of Fitzpatrick17k images, same Fitzpatrick-type mix, conditions
# that are not look-alikes) on the P3 splits, saves validation predictions for
# threshold analysis, and checks the tone-shift augmentation on study images.
# Inputs: MSLD v2.0, MSID and a Fitzpatrick17k copy with <md5>.jpg images
# (e.g. kaggle.com/datasets/mobaswiralfarabi/fitzpatrick17k-original).
# Steps 1-4 are the same as run_all.sh, so splits match the evaluation study.
set -euo pipefail
cd "$(dirname "$0")"

INPUT=${INPUT:-/kaggle/input}
OUT=${OUT:-/kaggle/working}
WORKERS=${WORKERS:-4}
MAX_HOURS=${MAX_HOURS:-10.5}
METHODS=${METHODS:-erm la rand tale}
CACHE=${CACHE:-/tmp/mpoxgen_img288}
FITZ_CSV_URL=${FITZ_CSV_URL:-https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv}
export PIPELINE_START=$(date +%s)
mkdir -p "$OUT"
exec > >(tee "$OUT/run_log_control.txt") 2>&1
stamp() { echo "== [$(date -u +%H:%M:%S) UTC, +$(( ($(date +%s) - PIPELINE_START) / 60 )) min] $*"; }

stamp "environment"
python -c "import torch, timm; print('torch', torch.__version__, '| timm', timm.__version__, '| cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

stamp "0. locate datasets"
eval "$(python scripts/00_find_datasets.py --input "$INPUT")"
echo "MSLD=$MSLD"; echo "MSID=$MSID"; echo "FITZ=$FITZ"
[ -n "$FITZ" ] || { echo "Fitzpatrick17k not found under $INPUT - add it as notebook input"; exit 1; }

stamp "1-4. index, embeddings, duplicate audit + ITA, splits (same as the evaluation study)"
python scripts/01_build_index.py --data "msld_v2=$MSLD" --data "msid=$MSID" --out data/index
python scripts/02_embedding_pairs.py --index data/index
python scripts/03_duplicate_audit.py --index data/index
python scripts/04_make_splits.py --index data/index --out splits

stamp "8. look-alike pool and matched control pool from Fitzpatrick17k"
wget -q -O fitzpatrick17k.csv "$FITZ_CSV_URL"
python scripts/08_lookalike_pool.py --fitz "$FITZ" --csv fitzpatrick17k.csv --index data/index --out data/pool \
    --control-out data/pool_control

stamp "12. tone-shift check on study images (CPU)"
python scripts/12_tone_shift_check.py --index data/index --out results_control || echo "tone-shift check failed"

stamp "9a. smoke test (1 epoch)"
python scripts/09_run_methods.py --smoke --methods $METHODS --p2-methods --workers "$WORKERS" --cache-dir "$CACHE"

stamp "9b. control study (time budget ${MAX_HOURS} h)"
python scripts/09_run_methods.py --methods $METHODS --p2-methods --out results_control \
    --workers "$WORKERS" --cache-dir "$CACHE" --max-hours "$MAX_HOURS"

stamp "10. summary"
python scripts/10_summarize_methods.py --results results_control --index data/index || echo "summary failed"
python scripts/11_operating_points.py --results results_control || echo "operating points failed"

stamp "11. package"
KEY="$OUT/key_results_control"
mkdir -p "$KEY"
cp results_control/*.csv results_control/*.md results_control/*.png "$KEY/" 2>/dev/null || true
cp data/pool_control/pool.csv "$KEY/control_pool.csv" 2>/dev/null || true
cp data/pool_control/pool_report.txt "$KEY/control_pool_report.txt" 2>/dev/null || true
cp data/pool_control/pool_examples.png "$KEY/control_pool_examples.png" 2>/dev/null || true
cp data/pool/pool_report.txt "$KEY/lookalike_pool_report.txt" 2>/dev/null || true
cp data/index/images.csv splits/manifest.csv "$KEY/" 2>/dev/null || true
zip -qr "$OUT/outputs_control.zip" results_control data/pool data/pool_control data/index/images.csv splits -x "*.pt" "*.npy" || true
stamp "DONE. Key files in $KEY"
