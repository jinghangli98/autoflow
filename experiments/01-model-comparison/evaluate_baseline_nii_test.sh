#!/bin/bash
#SBATCH --job-name=eval-baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --mail-user=jil202@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-8

# Slice-wise metrics (evaluate_baseline_nii_test.py) for a 2D baseline's
# nii_test outputs written by enhance_<METHOD>.sh <TARGET>. Scores the
# degraded input and the baseline output against the target's ground truth
# (raw -> <S>.nii.gz, denoised -> md<S>.nii.gz) with the same metrics, slice
# filter and CSV schema as the flow evaluation (evaluate_nii_test.sh), so the
# CSVs plot together with plot_eval_nii_test.py.
#
# Job array: subjects split round-robin across shards, one CSV per task.
# Merge afterwards (keeps one header):
#   awk 'FNR==1 && NR!=1{next}1' eval_nii_test_swinir_raw_shard*.csv > eval_nii_test_swinir_raw.csv
# submit_baselines.sh submits this with --dependency on the enhance array and
# a merge job after it.
#
# Usage:
#   sbatch experiments/01-model-comparison/evaluate_baseline_nii_test.sh swinir raw
#   sbatch experiments/01-model-comparison/evaluate_baseline_nii_test.sh realesrgan denoised
EXP_DIR=/vast/tibrahim/jil202/autoflow/experiments/01-model-comparison
DATA_ROOT=/vast/tibrahim/jil202/nii_test
METHOD=${1:?usage: $0 METHOD(swinir|realesrgan) TARGET(raw|denoised)}
TARGET=${2:-raw}
case "$TARGET" in
    raw) TASK=raw ;;
    denoised|md) TARGET=denoised; TASK=md ;;
    *) echo "unknown TARGET '$TARGET' (raw | denoised)" >&2; exit 1 ;;
esac
# ENSEMBLE=3plane (env) scores the three-plane ensemble outputs/CSVs (*_3plane).
ENSEMBLE=${ENSEMBLE:-axial}
case "$ENSEMBLE" in axial) SUFFIX="" ;; 3plane) SUFFIX="_3plane" ;;
    *) echo "unknown ENSEMBLE '$ENSEMBLE'" >&2; exit 1 ;; esac
ENHANCED_ROOT=$EXP_DIR/enhanced_nii_test_${METHOD}_${TARGET}${SUFFIX}

NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD_ID=$((${SLURM_ARRAY_TASK_ID:-1} - 1))
OUT_CSV=$EXP_DIR/eval_nii_test_${METHOD}_${TARGET}${SUFFIX}_shard${SLURM_ARRAY_TASK_ID:-1}.csv

source activate vsr
nvidia-smi
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "shard $SHARD_ID / $NUM_SHARDS: $METHOD/$TARGET (task $TASK) -> $OUT_CSV"
echo "enhanced root: $ENHANCED_ROOT"

python "$EXP_DIR/evaluate_baseline_nii_test.py" \
        --enhanced_root "$ENHANCED_ROOT" \
        --condition "$METHOD" --task "$TASK" \
        --data_root "$DATA_ROOT" --anatomy brain \
        --out_csv "$OUT_CSV" \
        --num_shards "$NUM_SHARDS" --shard_id "$SHARD_ID"
