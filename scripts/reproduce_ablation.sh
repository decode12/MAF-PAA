#!/usr/bin/env bash
# Full MAF is the BBC row of the main experiment, not a new fourth model.
set -euo pipefail
mode="${1:-evaluate}"
device="${2:-cpu}"
if [[ "$mode" != train && "$mode" != evaluate ]]; then
  echo 'Usage: bash scripts/reproduce_ablation.sh train|evaluate cpu|cuda' >&2
  exit 2
fi
for variant in without_mgr_complete without_haf_complete without_dafo_complete; do
  if [[ "$mode" == train ]]; then
    python scripts/train.py --config "configs/ablation/${variant}.json" \
      --output-dir "runs/bbc_${variant}" --device "$device"
    checkpoint="runs/bbc_${variant}/best.pt"
  else
    checkpoint="checkpoints/bbc_${variant}.pt"
  fi
  python scripts/evaluate.py --config "configs/ablation/${variant}.json" \
    --checkpoint "$checkpoint" --split test --output-dir "outputs/bbc_${variant}" --device "$device"
done
