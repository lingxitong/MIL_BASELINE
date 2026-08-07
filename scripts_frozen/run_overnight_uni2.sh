#!/usr/bin/env bash
# Overnight: UNI2 feature extract (3 eval sets + internal LP train) -> ABMIL linear probe
set -uo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PYTHONUNBUFFERED=1
ROOT="/home/oymx/work/frozen/project/MIL_BASELINE"
SCRIPTS="${ROOT}/scripts_frozen"
OUT="/mnt/net_sda/oymx/frozen_distill/eval_uni2"
LOG_DIR="${OUT}/logs"
PY="${PY:-/home/oymx/miniconda3/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/overnight_${STAMP}.log"
mkdir -p "${LOG_DIR}"
# CIFS/net_sda may not support symlinks — keep a plain pointer file
echo "${LOG}" > "${LOG_DIR}/latest.path"
export PYTHONUNBUFFERED=1
# line-buffered tee so watch scripts see progress on CIFS
exec > >(stdbuf -oL -eL tee -a "${LOG}") 2>&1

echo "======== overnight UNI2 start $(date) CUDA=${CUDA_VISIBLE_DEVICES} ========"
echo "python=${PY}"
"${PY}" -c "import torch,timm,sklearn; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'timm',timm.__version__)"

cd "${ROOT}"
echo "---- build manifest ----"
"${PY}" "${SCRIPTS}/build_uni2_manifest.py"

# Cap patches/slide so overnight can finish. Typical bags are 10k–30k;
# 4096 is a common MIL subsample and still covers the slide well.
MAX_PATCHES="${MAX_PATCHES:-4096}"
echo "---- MAX_PATCHES=${MAX_PATCHES} ----"

# Phase A: eval sets first (user-facing 3 datasets)
echo "---- extract EVAL sets ----"
"${PY}" "${SCRIPTS}/extract_uni2_from_patch_dirs.py" \
  --roles eval \
  --batch_size 64 \
  --num_workers 8 \
  --max_patches "${MAX_PATCHES}"

# Phase B: internal train/valid for linear-probe heads
echo "---- extract TRAIN_LP sets ----"
"${PY}" "${SCRIPTS}/extract_uni2_from_patch_dirs.py" \
  --roles train_lp \
  --batch_size 64 \
  --num_workers 8 \
  --max_patches "${MAX_PATCHES}"

# Phase C: evaluate
echo "---- linear probe eval ----"
"${PY}" "${SCRIPTS}/linear_probe_uni2_eval.py" --device cuda:0 --epochs 15

# Sync short summary into workspace
"${PY}" - <<'PY'
import json
from pathlib import Path
src = Path('/mnt/net_sda/oymx/frozen_distill/eval_uni2/results/summary_uni2.json')
dst_dir = Path('/home/oymx/work/frozen/data/eval_ready/uni2_results')
dst_dir.mkdir(parents=True, exist_ok=True)
if src.exists():
    data = json.loads(src.read_text())
    (dst_dir/'summary_uni2.json').write_text(json.dumps(data, indent=2, ensure_ascii=False))
    lines = ['# UNI2 ABMIL linear-probe on hard eval sets\n', f'Generated from {src}\n\n']
    lines.append('| dataset | UNI2 bacc | UNI2 wF1 | FrozenPath paper site bacc |\n')
    lines.append('|---|---:|---:|---:|\n')
    for k,v in data.items():
        base = v.get('baselines',{}).get('FrozenPath_paper_site_bacc', float('nan'))
        lines.append(
            f"| {k} | {v['balanced_accuracy_mean']:.4f}±{v['balanced_accuracy_std']:.4f} | "
            f"{v['weighted_f1_mean']:.4f}±{v['weighted_f1_std']:.4f} | {base} |\n"
        )
    (dst_dir/'RESULTS.md').write_text(''.join(lines))
    print(''.join(lines))
else:
    print('NO summary yet')
PY

echo "======== overnight UNI2 done $(date) ========"
touch "${LOG_DIR}/DONE"
