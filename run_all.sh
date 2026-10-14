#!/usr/bin/env bash
# Full pipeline in one command (made for a Kaggle GPU notebook):
#   !git clone -q https://github.com/<user>/mpox-generalization.git && bash mpox-generalization/run_all.sh
# Environment overrides: INPUT (datasets dir), OUT (outputs dir), TASKS, WORKERS,
# MAX_HOURS (stop starting new training runs after this many hours; default 10.5,
# safely inside Kaggle's 12-hour limit so finished results are always saved).
# To finish an interrupted study, add the earlier notebook version's output as an
# input: any results/<split>/<model>/metrics.json found under INPUT is reused.
set -euo pipefail
cd "$(dirname "$0")"

INPUT=${INPUT:-/kaggle/input}
OUT=${OUT:-/kaggle/working}
TASKS=${TASKS:-binary}
WORKERS=${WORKERS:-4}
MAX_HOURS=${MAX_HOURS:-10.5}
CACHE=${CACHE:-/tmp/mpoxgen_img288}
export PIPELINE_START=$(date +%s)
mkdir -p "$OUT"
exec > >(tee "$OUT/run_log.txt") 2>&1
stamp() { echo "== [$(date -u +%H:%M:%S) UTC, +$(( ($(date +%s) - PIPELINE_START) / 60 )) min] $*"; }

stamp "environment"
python -c "import torch, timm; print('torch', torch.__version__, '| timm', timm.__version__, '| cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
pip install -q grad-cam || echo "grad-cam not installed; Grad-CAM step will be skipped"

stamp "0. locate datasets"
eval "$(python scripts/00_find_datasets.py --input "$INPUT")"
echo "MSLD=$MSLD"
echo "MSID=$MSID"
RESUME=""
prev=$(find "$INPUT" -maxdepth 7 -path "*/results/*/*/metrics.json" 2>/dev/null | head -1 || true)
if [ -n "$prev" ]; then
  RESUME=$(dirname "$(dirname "$(dirname "$prev")")")
  echo "Earlier results found, will reuse: $RESUME"
fi

stamp "1. index"
python scripts/01_build_index.py --data "msld_v2=$MSLD" --data "msid=$MSID" --out data/index

stamp "2. embedding near-duplicates (auto-accept cos >= 0.97; rest saved for review)"
python scripts/02_embedding_pairs.py --index data/index

stamp "3. duplicate audit + ITA"
python scripts/03_duplicate_audit.py --index data/index

stamp "4. splits + leakage check"
python scripts/04_make_splits.py --index data/index --out splits

stamp "5a. smoke test"
python scripts/05_run_experiments.py --smoke --workers "$WORKERS" --cache-dir "$CACHE"

stamp "5b. full study: $TASKS (time budget ${MAX_HOURS} h)"
python scripts/05_run_experiments.py --tasks $TASKS --save-ckpt --workers "$WORKERS" \
  --cache-dir "$CACHE" --max-hours "$MAX_HOURS" ${RESUME:+--resume-from "$RESUME"}

stamp "6. summary"
python scripts/06_summarize.py --results results --index data/index || echo "summary step failed"

stamp "7. Grad-CAM"
if python -c "import pytorch_grad_cam" 2>/dev/null; then
  for run in results/binary_P3_*_rep0/resnet50 results/binary_P3_*_rep0/convnext_tiny; do
    [ -f "$run/model.pt" ] && python scripts/07_gradcam.py --run "$run" --index data/index || true
  done
fi
find results -name model.pt -delete || true   # keep the saved output small

stamp "8. package outputs"
KEY="$OUT/key_results"
mkdir -p "$KEY"
for f in results/paper_tables.md results/summary_by_model.csv results/inflation_gap.csv \
         results/ita_breakdown.csv results/all_runs.csv results/fig_protocols.png \
         data/index/duplicate_summary.csv data/index/index_report.txt splits/manifest.csv \
         data/index/cross_dataset_duplicates.png data/index/label_mismatch_duplicates.png \
         data/index/embedding_review_sheet.png; do
  [ -f "$f" ] && cp "$f" "$KEY/" || true
done
zip -qr "$OUT/outputs.zip" data/index splits results results_smoke -x "*.pt" "*.npy" || true
stamp "DONE. Key files in $KEY ; everything in $OUT/outputs.zip"
