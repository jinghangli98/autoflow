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
#SBATCH --exclude=gpu-n80,gpu-n82

# Usage:
#   sbatch train_flow.sh                      (default: all anatomies, all artifacts)
#   sbatch train_flow.sh "brain"              (single anatomy, all artifacts)
#   sbatch train_flow.sh "brain" "aniso"      (specialized anisotropic brain model)
#   sbatch train_flow.sh "brain knee" "aniso spike"  (multiple of each)
#   sbatch train_flow.sh "brain" "" "aniso 0.7"   (all artifacts, aniso up to 70%)
#   sbatch train_flow.sh "brain" "" ""        (disable the aniso up-weighting)
#
# Data: /vast multi-anatomy dataset (brain, knee, prostate). Each anatomy spans
# all its acquisitions and the artifact->raw / artifact->denoised /
# raw->denoised restoration tasks. Artifact families: undersampled, spike, aniso
# (the clean raw->denoised task is always kept regardless of the artifact filter).
#
# ARTIFACT_FRACTION (3rd arg, "FAMILY FRACTION") up-weights one artifact family
# to a fixed fraction of each anatomy's per-epoch samples. Default "aniso 0.7"
# makes aniso 70% of every anatomy that has it (others keep --samples_per_contrast),
# so the new bravo_ax_3T / flair_sag_3T aniso data dominates training. Pass "" to
# disable. Only applies with --balance_by anatomy_artifact.
DATA_ROOT=/vast/tibrahim/jil202/data
CONTRASTS=${1:-brain knee prostate}
ARTIFACTS=${2:-}
ARTIFACT_FRACTION=${3:-aniso 0.7}
echo "Data: root=$DATA_ROOT anatomies=[$CONTRASTS] artifacts=[${ARTIFACTS:-all}] artifact_fraction=[${ARTIFACT_FRACTION:-none}]"

# Pass --artifact only when a non-empty second arg is given.
ARTIFACT_ARG=""
if [ -n "$ARTIFACTS" ]; then
    ARTIFACT_ARG="--artifact $ARTIFACTS"
fi

# Pass --artifact_fraction only when a non-empty third arg is given.
ARTIFACT_FRACTION_ARG=""
if [ -n "$ARTIFACT_FRACTION" ]; then
    ARTIFACT_FRACTION_ARG="--artifact_fraction $ARTIFACT_FRACTION"
fi

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
        $ARTIFACT_ARG \
        $ARTIFACT_FRACTION_ARG \
        --data_root $DATA_ROOT \
        --distributed --fp16 --save_model --compile \
        --batch_size 8 --max_epochs 100 --sample 100 \
        --num_sampling_steps 2 --samples_per_contrast 2000 --balance_by anatomy_artifact --val_images_per_group 600 \
        --cfg_dropout_prob 0.1 --size 192 --checkpoint_path /vast/tibrahim/jil202/autoflow/checkpoints_uncertainty/flow_matching_3d_brain_all_best_epoch_158.pt

# To warm-start from a prior model instead, point --checkpoint_path at an
# existing .pt (architecture is unchanged); the relaxed loader handles mismatches.
