#!/usr/bin/env bash
# Train only (features already extracted).
set -euo pipefail
ROOT="/home/oymx/work/frozen/project/MIL_BASELINE"
OUT="/mnt/net_sda/oymx/frozen_distill/mil_uni2"
LOGDIR="$OUT/logs"
PY="${PY:-/home/oymx/miniconda3/envs/torch/bin/python}"
GPU="${GPU:-3}"
TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
cd "$ROOT"

# coverage check
"$PY" - <<'PY'
import pandas as pd
from pathlib import Path
pt=Path("/mnt/net_sda/oymx/frozen_distill/eval_uni2/pt_files")
for organ in ("thyroid","breast"):
    meta=pd.read_csv(f"/mnt/net_sda/oymx/frozen_distill/mil_uni2/datasets/internal_{organ}_uni2_meta.csv")
    miss=[sid for sid in meta.slide_id if not (pt/f"{sid}.pt").exists()]
    print(organ, "missing", len(miss))
    if miss:
        raise SystemExit(f"still missing {organ}: {miss[:5]}")
print("coverage OK")
PY

PIPE="$LOGDIR/train_pipeline_${TS}.log"
echo "$PIPE" > "$LOGDIR/latest_pipeline.path"
echo "[$(date -Is)] start train-only GPU=$GPU" | tee "$PIPE"

for organ in thyroid breast; do
  yaml="$OUT/configs/AB_MIL_internal_${organ}_uni2.yaml"
  log="$LOGDIR/train_${organ}_${TS}.log"
  echo "[$(date -Is)] train $organ -> $log" | tee -a "$PIPE"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" train_mil.py --yaml_path "$yaml" 2>&1 | tee "$log"
  echo "[$(date -Is)] train $organ DONE" | tee -a "$PIPE"
done

echo "[$(date -Is)] ALL TRAIN DONE" | tee -a "$PIPE"
