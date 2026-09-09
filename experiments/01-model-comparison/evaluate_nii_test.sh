#!/bin/bash
#SBATCH --job-name=eval-niitest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --mail-user=jil202@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-28

# Slice-wise evaluation of the flow model over the held-out nii_test split
# (evaluate_nii_test.py). Every degraded input per subject is scored per
# last-dim slice against both ground truths (raw / md) in three conditions:
# the degraded input itself (baseline), conditioned cfg1 (guidance 1.0), and
# unconditioned cfg0 (guidance 0.0). Metrics: PSNR/SSIM/MAE/LPIPS/FSIM -> CSV,
# one row per slice; near-empty slices (GT foreground below --min_slice_fg,
# default 5%) are skipped; reruns resume. The _fg5 CSV prefix keeps this
# campaign separate from pre-filter CSVs (resume would otherwise keep old
# unfiltered rows).
#
# Job array: --array sets the shard count; subjects split round-robin, each
# task writes its own CSV. After all tasks finish, merge (keeps one header):
#   awk 'FNR==1 && NR!=1{next}1' eval_nii_test_fg5_shard*.csv > eval_nii_test_fg5.csv
# then:
#   python plot_eval_nii_test.py eval_nii_test_fg5.csv --out_prefix figures/eval_nii_test --stats
#
# Usage (from anywhere; all paths are anchored to EXP_DIR):
#   sbatch experiments/01-model-comparison/evaluate_nii_test.sh     (all brain acquisitions)
#   sbatch experiments/01-model-comparison/evaluate_nii_test.sh mp2rage_ax_7T
EXP_DIR=/vast/tibrahim/jil202/autoflow/experiments/01-model-comparison
DATA_ROOT=/vast/tibrahim/jil202/nii_test
CKPT=/vast/tibrahim/jil202/autoflow/checkpoints_uncertainty/flow_matching_3d_brain_all_128ch_090526_ft_best.pt
ACQ=${1:-}
SAVE_DIR=$EXP_DIR/enhanced_nii_test

NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD_ID=$((SLURM_ARRAY_TASK_ID - 1))
OUT_CSV=$EXP_DIR/eval_nii_test_fg5${ACQ:+_$ACQ}_shard${SLURM_ARRAY_TASK_ID}.csv

source activate vsr
nvidia-smi
# RadBERT weights load from $HF_HOME cache; compute nodes may be offline.
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "shard $SHARD_ID / $NUM_SHARDS -> $OUT_CSV"

python "$EXP_DIR/evaluate_nii_test.py" \
        --checkpoint_path "$CKPT" \
        --data_root "$DATA_ROOT" \
        --anatomy brain \
        --out_csv "$OUT_CSV" \
        --save_dir "$SAVE_DIR" \
        --num_shards "$NUM_SHARDS" --shard_id "$SHARD_ID" \
        --num_sampling_steps 1 --euler --fp16 --non_overlap 3 \
        --planes axial sagittal coronal --compile \
        --guidance_scale 1.0 0.0 \
        ${ACQ:+--acquisition $ACQ}
