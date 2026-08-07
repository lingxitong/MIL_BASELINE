#!/usr/bin/env bash
# Extract missing UNI2 feats (2 GPUs) then train AB_MIL on internal thyroid & breast.
set -euo pipefail

ROOT="/home/oymx/work/frozen/project/MIL_BASELINE"
OUT="/mnt/net_sda/oymx/frozen_distill/mil_uni2"
FEAT="/mnt/net_sda/oymx/frozen_distill/eval_uni2/pt_files"
LOGDIR="$OUT/logs"
WEIGHTS="/mnt/sdb/chenwm/PFM_Segmentation/weight/uni_v2"
PY="${PY:-/home/oymx/miniconda3/bin/python3}"
MAX_PATCHES="${MAX_PATCHES:-4096}"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

cd "$ROOT"

extract_shard() {
  local gpu="$1"
  local shard="$2"
  local log="$LOGDIR/extract_shard${shard}_${TS}.log"
  echo "[extract] gpu=$gpu shard=$shard -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts_frozen/extract_uni2_from_patch_dirs.py \
    --manifest "$OUT/manifests/missing_uni2_shard${shard}.json" \
    --feat_dir "$FEAT" \
    --pretrained_weights_dir "$WEIGHTS" \
    --batch_size 64 \
    --num_workers 8 \
    --max_patches "$MAX_PATCHES" \
    2>&1 | tee "$log"
}

echo "[$(date -Is)] start extract MAX_PATCHES=$MAX_PATCHES" | tee "$LOGDIR/pipeline_${TS}.log"
# GPU3 free ~32GB; GPU4 free ~40GB (share with Jason ~8GB)
extract_shard 3 0 &
PID0=$!
extract_shard 4 1 &
PID1=$!
wait $PID0
EC0=$?
wait $PID1
EC1=$?
echo "[$(date -Is)] extract done ec0=$EC0 ec1=$EC1" | tee -a "$LOGDIR/pipeline_${TS}.log"
if [[ $EC0 -ne 0 || $EC1 -ne 0 ]]; then
  echo "EXTRACT FAILED" | tee -a "$LOGDIR/pipeline_${TS}.log"
  exit 1
fi

# verify coverage
"$PY" - <<'PY'
import pandas as pd
from pathlib import Path
pt=Path("/mnt/net_sda/oymx/frozen_distill/eval_uni2/pt_files")
for organ in ("thyroid","breast"):
    meta=pd.read_csv(f"/mnt/net_sda/oymx/frozen_distill/mil_uni2/datasets/internal_{organ}_uni2_meta.csv")
    miss=[sid for sid in meta.slide_id if not (pt/f"{sid}.pt").exists()]
    print(organ, "missing_after_extract", len(miss))
    if miss:
        raise SystemExit(f"still missing {organ}: {miss[:5]}")
print("coverage OK")
PY

train_one() {
  local organ="$1"
  local gpu="$2"
  local yaml="$OUT/configs/AB_MIL_internal_${organ}_uni2.yaml"
  local log="$LOGDIR/train_${organ}_${TS}.log"
  echo "[$(date -Is)] train $organ on gpu $gpu -> $log" | tee -a "$LOGDIR/pipeline_${TS}.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" train_mil.py --yaml_path "$yaml" 2>&1 | tee "$log"
  echo "[$(date -Is)] train $organ DONE" | tee -a "$LOGDIR/pipeline_${TS}.log"
}

# Train sequentially on GPU3 (ABMIL is light; avoid contention with extract leftovers)
train_one thyroid 3
train_one breast 3

echo "[$(date -Is)] ALL DONE" | tee -a "$LOGDIR/pipeline_${TS}.log"
echo "$LOGDIR/pipeline_${TS}.log" > "$LOGDIR/latest_pipeline.path"
