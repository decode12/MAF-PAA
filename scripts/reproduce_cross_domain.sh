#!/usr/bin/env bash
# Optional extension: no cross-domain table is reported in the current paper.
set -euo pipefail
trained_source="${1:-bbc}"
device="${2:-cpu}"
case "$trained_source" in bbc|guardian|usa_today|washington_post) ;; *) echo 'Unknown source' >&2; exit 2;; esac
for target in bbc guardian usa_today washington_post; do
  python scripts/evaluate.py --config configs/main.json --source "$trained_source" \
    --target-source "$target" --checkpoint "checkpoints/main_${trained_source}.pt" \
    --split test --output-dir "outputs/cross_domain/${trained_source}_to_${target}" --device "$device"
done
