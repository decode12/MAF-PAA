#!/usr/bin/env bash
# Run from the repository root. CUDA_VISIBLE_DEVICES selects an available GPU.
set -euo pipefail
mode="${1:-evaluate}"
device="${2:-cpu}"
if [[ "$mode" != train && "$mode" != evaluate ]]; then
  echo 'Usage: bash scripts/reproduce_main.sh train|evaluate cpu|cuda' >&2
  exit 2
fi
for source in bbc guardian usa_today washington_post; do
  if [[ "$mode" == train ]]; then
    python scripts/train.py --config configs/main.json --source "$source" \
      --output-dir "runs/main_${source}" --device "$device"
    checkpoint="runs/main_${source}/best.pt"
  else
    checkpoint="checkpoints/main_${source}.pt"
  fi
  python scripts/evaluate.py --config configs/main.json --source "$source" \
    --checkpoint "$checkpoint" --split test --output-dir "outputs/main_${source}" --device "$device"
done
