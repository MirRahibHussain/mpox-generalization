#!/usr/bin/env bash
# FACTOR study in one command (Kaggle GPU notebook with Internet on):
#   !git clone -q https://github.com/MirRahibHussain/mpox-generalization.git && bash mpox-generalization/run_factor.sh
# FACTOR = factorized disease x mpox-given-disease head + condition-diverse,
# tone-matched exposure to the whole Fitzpatrick17k atlas (src/mpoxgen/factor.py).
# Compared on the P3 cross-dataset splits with ERM, the random-negative control,
# a flat head on the same atlas draws, HOD (Roy et al. 2022) and two ablations;
# plus a condition-diversity test, an ITA-vs-Fitzpatrick check of the skin-tone
# proxy, an open-set analysis and P2 (in-dataset) runs for ERM and FACTOR.
# Inputs: MSLD v2.0, MSID and a Fitzpatrick17k copy with <md5>.jpg images.
# Steps 1-4 are the same as run_all.sh, so splits match the evaluation study.
set -euo pipefail
cd "$(dirname "$0")"

INPUT=${INPUT:-/kaggle/input}
OUT=${OUT:-/kaggle/working}
WORKERS=${WORKERS:-4}
MAX_HOURS=${MAX_HOURS:-10.5}
MAIN=${MAIN:-erm factor atlas hod rand factor_notone factor_static}
CACHE=${CACHE:-/tmp/mpoxgen_img288}
RES=results_factor
FITZ_CSV_URL=${FITZ_CSV_URL:-https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv}
PAIRS="factor:erm factor:rand factor:atlas factor:hod factor:factor_notone factor:factor_static atlas:rand hod:atlas rand:erm"
export PIPELINE_START=$(date +%s)
export MPOXGEN_DETERMINISTIC=1
mkdir -p "$OUT"
exec > >(tee "$OUT/run_log_factor.txt") 2>&1
stamp() { echo "== [$(date -u +%H:%M:%S) UTC, +$(( ($(date +%s) - PIPELINE_START) / 60 )) min] $*"; }

stamp "environment"
python -c "import torch, timm; print('torch', torch.__version__, '| timm', timm.__version__, '| cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
git log -1 --format="code version: %h %s" || true

stamp "0. locate datasets"
eval "$(python scripts/00_find_datasets.py --input "$INPUT")"
echo "MSLD=$MSLD"; echo "MSID=$MSID"; echo "FITZ=$FITZ"
[ -n "$FITZ" ] || { echo "Fitzpatrick17k not found under $INPUT - add it as notebook input"; exit 1; }

stamp "1-4. index, embeddings, duplicate audit + ITA, splits (same as the evaluation study)"
python scripts/01_build_index.py --data "msld_v2=$MSLD" --data "msid=$MSID" --out data/index
python scripts/02_embedding_pairs.py --index data/index
python scripts/03_duplicate_audit.py --index data/index
python scripts/04_make_splits.py --index data/index --out splits

stamp "8. look-alike pool and matched control pool (for 'rand' and 'factor_static')"
wget -q -O fitzpatrick17k.csv "$FITZ_CSV_URL"
python scripts/08_lookalike_pool.py --fitz "$FITZ" --csv fitzpatrick17k.csv --index data/index --out data/pool \
    --control-out data/pool_control

stamp "15. whole-atlas pool (duplicates of study images removed, ITA per image)"
python scripts/15_atlas_pool.py --fitz "$FITZ" --csv fitzpatrick17k.csv --index data/index --out data/pool_atlas \
    --workers "$WORKERS"

stamp "16. ITA proxy vs annotated Fitzpatrick type; tone-matching check (CPU)"
mkdir -p "$RES"
python scripts/16_ita_validation.py --pool data/pool_atlas/pool.csv --index data/index --out "$RES" \
    || echo "ITA validation failed"

stamp "9a. smoke test (1 epoch, every new method and both heads)"
python scripts/09_run_methods.py --smoke --methods $MAIN div5 --p2-methods factor \
    --workers "$WORKERS" --cache-dir "$CACHE"

stamp "9b. main comparison on P3 (time budget ${MAX_HOURS} h; all methods of a split before the next split)"
python scripts/09_run_methods.py --methods $MAIN --p2-methods --order split --out "$RES" \
    --workers "$WORKERS" --cache-dir "$CACHE" --max-hours "$MAX_HOURS"

stamp "9c. condition-diversity test (ResNet-50, MSID-trained, 3 seeds; 'atlas' = all conditions)"
python scripts/09_run_methods.py --methods div5 div20 --p2-methods --models resnet50 --train-dataset msid \
    --order split --out "$RES" --workers "$WORKERS" --cache-dir "$CACHE" --max-hours "$MAX_HOURS"

stamp "9d. in-dataset (P2) check for ERM and FACTOR"
python scripts/09_run_methods.py --methods --p2-methods erm factor --order split --out "$RES" \
    --workers "$WORKERS" --cache-dir "$CACHE" --max-hours "$MAX_HOURS"

stamp "10-13, 17. summary, operating points, paired statistics, open-set analysis"
python scripts/10_summarize_methods.py --results "$RES" --index data/index --pairs $PAIRS || echo "summary failed"
python scripts/11_operating_points.py --results "$RES" --pairs $PAIRS || echo "operating points failed"
python scripts/13_paired_stats.py --runs "$RES/all_runs.csv" --pairs $PAIRS \
    --metrics balanced_accuracy auroc mpox_specificity mpox_sensitivity || echo "paired stats failed"
python scripts/17_open_set.py --results "$RES" --index data/index || echo "open-set analysis failed"
python scripts/13_paired_stats.py --runs "$RES/open_set_runs.csv" --pairs $PAIRS --out "$RES/open_set_paired" \
    --metrics closed_bal_acc unseen_spec disease_spec healthy_spec auroc_vs_disease || echo "open-set stats failed"

stamp "package"
KEY="$OUT/key_results_factor"
mkdir -p "$KEY"
cp "$RES"/*.csv "$RES"/*.md "$RES"/*.png "$KEY/" 2>/dev/null || true
cp data/pool_atlas/pool_report.txt "$KEY/atlas_pool_report.txt" 2>/dev/null || true
cp data/pool_atlas/pool_examples.png "$KEY/atlas_pool_examples.png" 2>/dev/null || true
cp data/pool_atlas/pool.csv "$KEY/atlas_pool.csv" 2>/dev/null || true
cp data/pool_control/pool_report.txt "$KEY/control_pool_report.txt" 2>/dev/null || true
cp data/index/images.csv splits/manifest.csv "$KEY/" 2>/dev/null || true
zip -qr "$OUT/outputs_factor.zip" "$RES" data/pool_atlas/pool.csv data/pool_control/pool.csv data/index/images.csv splits \
    -x "*.pt" "*.npy" || true
stamp "DONE. Key files in $KEY"
