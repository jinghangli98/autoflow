#!/bin/bash
# Submit the 2D baseline pipeline for every (method, target) pair:
#   enhance_<method>.sh <target>            (GPU job array, restores volumes)
#     -> evaluate_baseline_nii_test.sh      (GPU job array, per-shard CSVs; afterok)
#        -> merge job                       (one CSV per method/target; afterok)
#           -> plot job (after ALL merges)  (plot_eval_nii_test.py figures)
#
# Final CSVs: eval_nii_test_{swinir,realesrgan}_{raw,denoised}.csv in EXP_DIR.
# Figures (flow fg5 CSV + every baseline CSV present, both tasks):
#   figures/eval_nii_test_methods.{pdf,png} + _summary.csv   (all methods)
#   figures/eval_nii_test_<method>.{pdf,png} + _summary.csv  (flow + one baseline)
# The plot job runs plot_baselines.sh, which can also be run by hand once the
# CSVs exist:  bash experiments/01-model-comparison/plot_baselines.sh
# Every step is resumable, so rerunning this script after a failure only
# redoes what is missing.
#
# Usage:
#   bash experiments/01-model-comparison/submit_baselines.sh                    (all four)
#   bash experiments/01-model-comparison/submit_baselines.sh swinir             (both targets)
#   bash experiments/01-model-comparison/submit_baselines.sh swinir raw         (one pair)
#   ENSEMBLE=3plane bash experiments/01-model-comparison/submit_baselines.sh   (3-plane ensemble)
set -euo pipefail
EXP_DIR=/vast/tibrahim/jil202/autoflow/experiments/01-model-comparison
METHODS=${1:-"swinir realesrgan"}
TARGETS=${2:-"raw denoised"}
# ENSEMBLE=3plane (env) runs/scores the three-plane ensemble variant; files get
# a _3plane suffix so both variants coexist. Inherited by every sbatch below.
export ENSEMBLE=${ENSEMBLE:-axial}
case "$ENSEMBLE" in axial) SUFFIX="" ;; 3plane) SUFFIX="_3plane" ;;
    *) echo "unknown ENSEMBLE '$ENSEMBLE'" >&2; exit 1 ;; esac

cd "$EXP_DIR"
MERGES=""
for METHOD in $METHODS; do
    for TARGET in $TARGETS; do
        ENH=$(sbatch --parsable "enhance_${METHOD}.sh" "$TARGET" | cut -d';' -f1)
        EVAL=$(sbatch --parsable --dependency=afterok:"$ENH" \
                      evaluate_baseline_nii_test.sh "$METHOD" "$TARGET" | cut -d';' -f1)
        MERGE=$(sbatch --parsable --dependency=afterok:"$EVAL" \
                      --cluster=gpu --partition=a100_nvlink --gres=gpu:1 --time=0-00:10:00 \
                      --job-name="merge-$METHOD-$TARGET$SUFFIX" \
                      --output="$EXP_DIR/slurm-merge-$METHOD-$TARGET-%j.out" \
                      --wrap="cd $EXP_DIR && awk 'FNR==1 && NR!=1{next}1' \
                              eval_nii_test_${METHOD}_${TARGET}${SUFFIX}_shard*.csv \
                              > eval_nii_test_${METHOD}_${TARGET}${SUFFIX}.csv \
                              && wc -l eval_nii_test_${METHOD}_${TARGET}${SUFFIX}.csv" \
                | cut -d';' -f1)
        echo "$METHOD/$TARGET: enhance $ENH -> eval $EVAL -> merge $MERGE"
        MERGES="$MERGES:$MERGE"
    done
done
PLOT=$(sbatch --parsable --dependency=afterok${MERGES} \
              --cluster=gpu --partition=a100_nvlink --gres=gpu:1 --time=0-00:30:00 \
              --job-name="plot-baselines$SUFFIX" \
              --output="$EXP_DIR/slurm-plot-baselines-%j.out" \
              "$EXP_DIR/plot_baselines.sh" | cut -d';' -f1)
echo "plot (after merges${MERGES}): $PLOT"
