#!/usr/bin/env bash
# Full pipeline in one command (made for a Kaggle GPU notebook):
#   !git clone -q https://github.com/<user>/mpox-generalization.git && bash mpox-generalization/run_all.sh
# Environment overrides: INPUT (datasets dir), OUT (outputs dir), TASKS, WORKERS.
set -euo pipefail
cd "$(dirname "$0")"

INPUT=${INPUT:-/kaggle/input}
OUT=${OUT:-/kaggle/working}
TASKS=${TASKS:-binary}
WORKERS=${WORKERS:-4}
mkdir -p "$OUT"
exec > >(tee "$OUT/run_log.txt") 2>&1

echo "== environment"
python -c "import torch, timm; print('torch', torch.__version__, '| timm', timm.__version__, '| cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
pip install -q grad-cam || echo "grad-cam not installed; Grad-CAM step will be skipped"

echo "== 0. locate datasets"
eval "$(python scripts/00_find_datasets.py --input "$INPUT")"
echo "MSLD=$MSLD"
echo "MSID=$MSID"

echo "== 1. index"
python scripts/01_build_index.py --data "msld_v2=$MSLD" --data "msid=$MSID" --out data/index

echo "== 2. embedding near-duplicates (auto-accept cos >= 0.97; rest saved for review)"
python scripts/02_embedding_pairs.py --index data/index

echo "== 3. duplicate audit + ITA"
python scripts/03_duplicate_audit.py --index data/index

echo "== 4. splits + leakage check"
python scripts/04_make_splits.py --index data/index --out splits

echo "== 5a. smoke test"
python scripts/05_run_experiments.py --smoke --workers "$WORKERS"

echo "== 5b. full study: $TASKS"
python scripts/05_run_experiments.py --tasks $TASKS --save-ckpt --workers "$WORKERS"

echo "== 6. summary"
python scripts/06_summarize.py --results results --index data/index

echo "== 7. Grad-CAM"
if python -c "import pytorch_grad_cam" 2>/dev/null; then
  for run in results/binary_P3_*_rep0/resnet50 results/binary_P3_*_rep0/convnext_tiny; do
    [ -f "$run/model.pt" ] && python scripts/07_gradcam.py --run "$run" --index data/index || true
  done
fi

echo "== 8. package outputs"
KEY="$OUT/key_results"
mkdir -p "$KEY"
for f in results/paper_tables.md results/summary_by_model.csv results/inflation_gap.csv \
         results/ita_breakdown.csv results/all_runs.csv results/fig_protocols.png \
         data/index/duplicate_summary.csv data/index/index_report.txt splits/manifest.csv \
         data/index/cross_dataset_duplicates.png data/index/label_mismatch_duplicates.png \
         data/index/embedding_review_sheet.png; do
  [ -f "$f" ] && cp "$f" "$KEY/"
done
zip -qr "$OUT/outputs.zip" data/index splits results results_smoke -x "*.pt" "*.npy"
echo "== DONE. Key files in $KEY ; everything in $OUT/outputs.zip"
