#!/usr/bin/env bash
set -uo pipefail
OUT=/mnt/net_sda/oymx/frozen_distill/eval_uni2
echo "======== UNI2 overnight $(date '+%F %T') ========"
if [[ -f "${OUT}/logs/DONE" ]]; then echo "[状态] 已完成"; else echo "[状态] 进行中/未完成"; fi
NPROC=$(pgrep -fc 'extract_uni2_from_patch_dirs|linear_probe_uni2_eval|run_overnight_uni2' || true)
echo "[进程] ${NPROC}"
NPT=$(ls -1 "${OUT}/pt_files"/*.pt 2>/dev/null | wc -l)
echo "[特征] ${NPT} 个 .pt"
if [[ -f "${OUT}/manifests/uni2_extract_manifest.json" ]]; then
  python3 - <<'PY'
import json
from pathlib import Path
m=json.loads(Path('/mnt/net_sda/oymx/frozen_distill/eval_uni2/manifests/uni2_extract_manifest.json').read_text())
print('[清单]', m['n_slides'], 'slides', m['by_dataset'])
PY
fi
echo "[日志尾]"
LOGF="${OUT}/logs/latest.log"
if [[ -f "${OUT}/logs/latest.path" ]]; then LOGF="$(cat "${OUT}/logs/latest.path")"; fi
# fallback: newest overnight_*.log
[[ -f "${LOGF}" ]] || LOGF="$(ls -t "${OUT}"/logs/overnight_*.log 2>/dev/null | head -1 || true)"
tail -n 15 "${LOGF}" 2>/dev/null || echo "(no log)"
if [[ -f "${OUT}/results/summary_uni2.json" ]]; then
  echo "[结果]"
  python3 - <<'PY'
import json
from pathlib import Path
d=json.loads(Path('/mnt/net_sda/oymx/frozen_distill/eval_uni2/results/summary_uni2.json').read_text())
for k,v in d.items():
    print(f"  {k}: bacc={v['balanced_accuracy_mean']:.4f}±{v['balanced_accuracy_std']:.4f} wf1={v['weighted_f1_mean']:.4f}")
PY
fi
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-4}" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
