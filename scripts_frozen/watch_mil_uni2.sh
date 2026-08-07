#!/usr/bin/env bash
OUT="/mnt/net_sda/oymx/frozen_distill/mil_uni2/logs"
FEAT="/mnt/net_sda/oymx/frozen_distill/eval_uni2/pt_files"
echo "=== $(date -Is) ==="
if [[ -f "$OUT/latest_pipeline.path" ]]; then
  PIPE=$(cat "$OUT/latest_pipeline.path")
  echo "pipeline: $PIPE"
  tail -n 20 "$PIPE" 2>/dev/null || true
fi
echo "--- extract progress ---"
for f in "$OUT"/extract_shard*_*.log; do
  [[ -f "$f" ]] || continue
  echo "## $(basename "$f")"
  rg -c '\] OK |\] skip |\] FAIL |SUMMARY' "$f" 2>/dev/null || true
  tail -n 2 "$f"
done
echo "--- feat count ---"
ls "$FEAT"/*.pt 2>/dev/null | wc -l
echo "--- train tails ---"
for f in "$OUT"/train_*_*.log; do
  [[ -f "$f" ]] || continue
  echo "## $(basename "$f")"
  tail -n 15 "$f"
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '4,5p'
