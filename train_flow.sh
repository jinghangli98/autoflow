#!/bin/bash
#SBATCH --job-name=ddp-1000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --mail-user=jil202@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:8
#SBATCH --constraint=80g

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

# # Properly activate conda environment
source activate vsr
nvidia-smi

# RadBERT weights are pre-cached under $HF_HOME (/ix1/.../.cache/huggingface);
# load from cache without contacting huggingface.co (compute nodes may be offline).
export HF_HUB_OFFLINE=1

# One process per allocated GPU.
NGPU=$(nvidia-smi -L | wc -l)
echo "Launching on $NGPU GPUs"

python -m torch.distributed.run --nproc_per_node=$NGPU train_flow.py \
        --contrast $CONTRASTS \
        --data_root $DATA_ROOT \
        --distributed --fp16 --save_model --compile \
        --batch_size 4 --max_epochs 100 --sample 100 \
        --num_sampling_steps 2 --samples_per_contrast 0 \
        --cfg_dropout_prob 0.1 --size 192 --checkpoint_path /vast/tibrahim/jil202/autoflow/checkpoints_s/flow_matching_3d_brain_knee_prostate_best_epoch_57.pt
# To warm-start from a prior model instead, point --checkpoint_path at an
# existing .pt (architecture is unchanged); the relaxed loader handles mismatches.
