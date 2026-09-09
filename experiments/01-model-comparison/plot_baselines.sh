#!/bin/bash
# Figures comparing the flow model with the 2D baselines on nii_test.
# Uses the merged flow CSV (eval_nii_test_fg5.csv, built from the fg5 shards
# if missing) plus every merged baseline CSV that exists
# (eval_nii_test_{swinir,realesrgan}_{raw,denoised}.csv). Runs as the last
# job of submit_baselines.sh, or by hand once the CSVs are there.
set -euo pipefail
EXP_DIR=/vast/tibrahim/jil202/autoflow/experiments/01-model-comparison
cd "$EXP_DIR"
source activate vsr 2>/dev/null || true
# ENSEMBLE=3plane (env): use the three-plane ensemble baseline CSVs (*_3plane)
# and write *_3plane figures.
ENSEMBLE=${ENSEMBLE:-axial}
case "$ENSEMBLE" in axial) SUFFIX="" ;; 3plane) SUFFIX="_3plane" ;;
    *) echo "unknown ENSEMBLE '$ENSEMBLE'" >&2; exit 1 ;; esac

FLOW=eval_nii_test_fg5.csv
if [ ! -f "$FLOW" ]; then
    awk 'FNR==1 && NR!=1{next}1' eval_nii_test_fg5_shard*.csv > "$FLOW"
    echo "merged flow shards -> $FLOW ($(wc -l < "$FLOW") lines)"
fi

ALL=""
for METHOD in swinir realesrgan; do
    CSVS=$(ls eval_nii_test_${METHOD}_{raw,denoised}${SUFFIX}.csv 2>/dev/null || true)
    if [ -z "$CSVS" ]; then
        echo "no merged CSVs for $METHOD yet; skipping"; continue
    fi
    ALL="$ALL $CSVS"
    echo "=== $METHOD: $CSVS ==="
    python plot_eval_nii_test.py "$FLOW" $CSVS \
        --out_prefix figures/eval_nii_test_${METHOD}${SUFFIX} --stats
done
if [ -n "$ALL" ]; then
    echo "=== all methods ==="
    python plot_eval_nii_test.py "$FLOW" $ALL \
        --out_prefix figures/eval_nii_test_methods${SUFFIX} --stats
fi
