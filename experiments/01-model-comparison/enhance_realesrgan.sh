#!/bin/bash
#SBATCH --job-name=enh-realesrgan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --mail-user=jil202@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-12

# Whole-volume 2D Real-ESRGAN inference over every degraded input in the
# held-out nii_test split, writing the restored volumes into a mirror tree at
# $OUT_ROOT (<anatomy>/<acq>/<subject>/<file>.nii.gz).
#
# Inputs come from list_test_inputs.py -- the same classify_file discovery
# evaluate_nii_test.py uses -- so the baseline processes exactly the volumes
# the flow model is scored on (artifact files only; never the clean GT, the
# md* target, or d* denoised volumes). Existing outputs are skipped so the
# job is resumable.
#
# Job array: the --array range sets the number of shards; the flat list of input
# files is split round-robin across shards. Edit --array=1-N to change parallelism.
#
# Usage (from anywhere; paths are anchored to EXP_DIR):
#   sbatch experiments/01-model-comparison/enhance_realesrgan.sh raw       (-> fully sampled GT)
#   sbatch experiments/01-model-comparison/enhance_realesrgan.sh denoised  (-> md denoised+biascorr GT)
#
# Arg 1 TARGET (raw | denoised, default raw) picks the checkpoint trained for
# that ground truth and a target-specific output tree, so the two models
# never share (or skip on) each other's outputs. Override the checkpoint
# with CKPT=... in the environment. Score the outputs afterwards with
# evaluate_baseline_nii_test.sh realesrgan <TARGET> (submit_baselines.sh chains
# both with a SLURM dependency).
EXP_DIR=/vast/tibrahim/jil202/autoflow/experiments/01-model-comparison
DATA_ROOT=/vast/tibrahim/jil202/nii_test
TARGET=${1:-raw}
case "$TARGET" in
    raw)       DEFAULT_CKPT=/vast/tibrahim/jil202/autoflow/checkpoints_realesrgan/realesrgan_2d_raw_brain_all_090726_best.pt ;;
    denoised|md) TARGET=denoised
               DEFAULT_CKPT=/vast/tibrahim/jil202/autoflow/checkpoints_realesrgan/realesrgan_2d_denoised_brain_all_090726_best.pt ;;
    *) echo "unknown TARGET '$TARGET' (raw | denoised)" >&2; exit 1 ;;
esac
# ENSEMBLE=3plane (env): mean-ensemble the three array-axis passes per volume
# (TSE: through-plane only), like the flow evaluation; outputs go to a
# *_3plane tree so the axial-only results are kept. Default: axial only.
ENSEMBLE=${ENSEMBLE:-axial}
case "$ENSEMBLE" in
    axial)  SUFFIX="";        ENS_FLAG="" ;;
    3plane) SUFFIX="_3plane"; ENS_FLAG="--ensemble" ;;
    *) echo "unknown ENSEMBLE '$ENSEMBLE' (axial | 3plane)" >&2; exit 1 ;;
esac
OUT_ROOT=$EXP_DIR/enhanced_nii_test_realesrgan_${TARGET}${SUFFIX}
CKPT=${CKPT:-$DEFAULT_CKPT}
# percentile matches checkpoints retrained on the whole-volume dataset_2d
# pipeline; use NORM=max for the old max-normalized checkpoints.
NORM=${NORM:-percentile}
PLANE=axial
BATCH_SIZE=16

NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD_ID=$((${SLURM_ARRAY_TASK_ID:-1} - 1))

source activate vsr
nvidia-smi
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "shard $SHARD_ID / $NUM_SHARDS -> $OUT_ROOT"
echo "target: $TARGET  ensemble: $ENSEMBLE"
echo "checkpoint: $CKPT"
ls -l --time-style=long-iso "$CKPT"

# Deterministic list of artifact inputs (classify_file discovery), then take
# this shard's slice.
mapfile -t INPUTS < <(python "$EXP_DIR/list_test_inputs.py" \
        --data_root "$DATA_ROOT" --anatomy brain | sort)

echo "total inputs: ${#INPUTS[@]}"

idx=0
for in_path in "${INPUTS[@]}"; do
    if [ $((idx % NUM_SHARDS)) -ne "$SHARD_ID" ]; then
        idx=$((idx + 1)); continue
    fi
    idx=$((idx + 1))

    rel=${in_path#"$DATA_ROOT"/}                # <anatomy>/<acq>/<subject>/<file>.nii.gz
    out_path="$OUT_ROOT/$rel"
    if [ -f "$out_path" ]; then
        echo "skip (exists): $out_path"; continue
    fi

    echo "=== $rel ==="
    python "$EXP_DIR/enhance_realesrgan.py" \
        --input_path "$in_path" \
        --output_path "$out_path" \
        --checkpoint_path "$CKPT" \
        --plane "$PLANE" --batch_size "$BATCH_SIZE" --norm "$NORM" --fp16 $ENS_FLAG
done

echo "shard $SHARD_ID done."
