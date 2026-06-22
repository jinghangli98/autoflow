#!/bin/bash
#SBATCH --job-name=ddp-1000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --partition=h200
#SBATCH --mail-user=jil202@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:4

# Usage:
#   sbatch train_flow.sh           (default: all anatomies)
#   sbatch train_flow.sh "brain"   (single anatomy)
#
# Data: /vast multi-anatomy dataset (brain, knee, prostate). Each anatomy spans
# all its acquisitions and the artifact->raw / artifact->denoised /
# raw->denoised restoration tasks.
DATA_ROOT=/vast/tibrahim/jil202/data
CONTRASTS=${1:-brain knee prostate}
echo "Data: root=$DATA_ROOT anatomies=[$CONTRASTS]"

# Per-(anatomy x artifact) per-epoch sample quota. For the default
# brain/knee/prostate run we set explicit counts per cell: aniso is weighted
# heaviest (weakest artifact), spike next, undersample (R*/GRAPPA) baseline; and
# knee/prostate above brain (anatomy imbalance). Each anatomy's three cells sum
# to its old per-anatomy budget (brain 2000, knee/prostate 20000). raw->denoised
# is kept separately at a fixed per-anatomy count via --denoise_samples_per_contrast.
# For any other --contrast set, fall back to anatomy auto-balance (0).
if [ "$CONTRASTS" = "brain knee prostate" ]; then
    CELL_SAMPLES="\
brain:undersample=5000 brain:spike=5000 brain:aniso=5000 \
knee:undersample=5000 knee:spike=5000 knee:aniso=5000 \
prostate:undersample=5000 prostate:spike=5000 prostate:aniso=5000"
    DENOISE_PER_CONTRAST=5000
    QUOTA_ARGS="--cell_samples $CELL_SAMPLES --denoise_samples_per_contrast $DENOISE_PER_CONTRAST"
    echo "cell_samples=[$CELL_SAMPLES] denoise_per_contrast=$DENOISE_PER_CONTRAST"
else
    QUOTA_ARGS="--samples_per_contrast 0"
    echo "samples_per_contrast=[0] (anatomy auto-balance)"
fi

# Properly activate conda environment (`source activate` isn't on PATH on the
# compute nodes; source conda.sh first, then `conda activate`).
source /ihome/tibrahim/jil202/miniconda3/etc/profile.d/conda.sh
conda activate vsr
nvidia-smi

# RadBERT weights are pre-cached under $HF_HOME (/ix1/.../.cache/huggingface);
# load from cache without contacting huggingface.co (compute nodes may be offline).
export HF_HUB_OFFLINE=1

# Multi-node rendezvous: first node in the allocation is the c10d master.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

# GPUs on THIS node (one torchrun launcher per node spawns this many workers).
NGPU=$(nvidia-smi -L | wc -l)
echo "Launching $SLURM_NNODES node(s) x $NGPU GPU(s); master=$MASTER_ADDR:$MASTER_PORT"

# Single node: run torchrun directly (no srun). srun would inherit multiple
# SLURM_MEM_PER_* vars from the allocation and abort ("mutually exclusive"); it
# buys nothing here since torchrun already spawns one worker per local GPU. For a
# true multi-node run, wrap this in `srun` and first
# `unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU` to leave a single mem spec.
python -m torch.distributed.run \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=$NGPU \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train_flow.py \
        --contrast $CONTRASTS \
        --data_root $DATA_ROOT \
        --distributed --fp16 --save_model --compile \
        --batch_size 4 --max_epochs 100 --sample 100 \
        --num_sampling_steps 2 $QUOTA_ARGS \
        --cfg_dropout_prob 0.1 --size 192 --val_balanced_per_cell 100 --checkpoint_path /vast/tibrahim/jil202/autoflow/checkpoints_l/flow_matching_3d_brain_knee_prostate_best_epoch_47.pt

# To warm-start from a prior model instead, point --checkpoint_path at an
# existing .pt (architecture is unchanged); the relaxed loader handles mismatches.
