#!/bin/bash
#SBATCH --job-name=swinir-2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --mail-user=jil202@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --time=0-48:00:00
#SBATCH --gres=gpu:4

# Supervised SwinIR 2D restoration baseline (sibling of train_superformer.sh).
#
# Usage:
#   sbatch train_swinir.sh                  (default: brain, raw GT, from scratch)
#   sbatch train_swinir.sh "brain" "raw"    (artifact -> fully sampled)
#   sbatch train_swinir.sh "brain" "md"     (artifact/raw -> denoised+biascorr)
#   sbatch train_swinir.sh "brain knee" "raw" "aniso"   (restrict artifacts)
#   sbatch train_swinir.sh "brain" "raw" "" /path/to/ckpt.pt   (warm-start weights)
#
# Positional args:
#   1 CONTRASTS    anatomy group(s): brain | knee | prostate (space-separated)
#   2 TARGET       ground truth: raw (fully sampled) | md/denoised (denoised+biascorr)
#   3 ARTIFACTS    optional artifact families: undersampled spike aniso (default all)
#   4 PRETRAINED   optional checkpoint (.pt) to warm-start model weights from
#                  (default: train from scratch). Pass "" for ARTIFACTS to skip it.
#
# Environment overrides:
#   FG_FRACTION=0.25   min foreground fraction to accept a crop without
#                      re-drawing (same rule as train_flow; 0.0 = accept all)
#   RUN_TAG=090726_fg0 inserted into checkpoint names so runs never overwrite
#                      each other; default <mmddyy>. Also the W&B run name.
#   VAL_INTERVAL=5     validate / consider best every N epochs (train_flow's 5)
#   EMA_DECAY=0.9999   per-step EMA of the weights used for validation/best
#                      (train_flow's 0.9999; 0 disables EMA)
#
# Data: whole-volume nii layout (dataset_2d.py on top of dataset.py); same
# patient-level val split and percentile normalization as train_flow. A step
# holds batch_size volume pairs x slices_per_volume 2D crops per GPU.
DATA_ROOT=/vast/tibrahim/jil202/nii
CONTRASTS=${1:-brain}
TARGET=${2:-raw}
ARTIFACTS=${3:-}
PRETRAINED=${4:-}
FG_FRACTION=${FG_FRACTION:-0.25}
RUN_TAG=${RUN_TAG:-$(date +%m%d%y)}
VAL_INTERVAL=${VAL_INTERVAL:-5}
EMA_DECAY=${EMA_DECAY:-0.9999}
echo "Data: root=$DATA_ROOT anatomies=[$CONTRASTS] target=[$TARGET] artifacts=[${ARTIFACTS:-all}] pretrained=[${PRETRAINED:-none}] fg_fraction=${FG_FRACTION} run_tag=${RUN_TAG}"

# Pass --artifact only when a non-empty third arg is given.
ARTIFACT_ARG=""
if [ -n "$ARTIFACTS" ]; then
    ARTIFACT_ARG="--artifact $ARTIFACTS"
fi

# Pass --checkpoint_path only when a non-empty fourth arg is given.
CKPT_ARG=""
if [ -n "$PRETRAINED" ]; then
    CKPT_ARG="--checkpoint_path $PRETRAINED"
fi

source activate vsr
nvidia-smi

# One process per allocated GPU.
NGPU=$(nvidia-smi -L | wc -l)
echo "Launching on $NGPU GPUs"

python -m torch.distributed.run --nproc_per_node=$NGPU train_swinir.py \
        --contrast $CONTRASTS \
        --target_type $TARGET \
        $ARTIFACT_ARG \
        --data_root $DATA_ROOT \
        --distributed --fp16 --save_model \
        --batch_size 4 --slices_per_volume 8 --size 96 --max_epochs 500 --sample 100 \
        --fg_fraction $FG_FRACTION --run_tag $RUN_TAG --num_workers 12 \
        --val_interval $VAL_INTERVAL --ema_decay $EMA_DECAY \
        --scheduler_milestones 15000 25000 29000 \
        --lr 1e-4 --wandb \
        --samples_per_contrast 1000 --balance_by anatomy_artifact --val_images_per_group 600 --compile \
        $CKPT_ARG

# Notes:
#   * Trains from scratch by default. Pass a 4th positional arg (a .pt checkpoint)
#     to warm-start the model weights; only shape-matching tensors are loaded
#     (size/window-dependent buffers are skipped and recomputed).
#   * --size 96 matches the flow model's 96x96(x7) training crops, so both
#     models see the same crop geometry; every volume fits a 96 in-plane
#     crop along at least one axis (192 did not: small-FOV FLAIR 7T).
#   * --size must be a multiple of --window_size (default 8).
#   * --compile is off by default (Swin's dynamic shift masks can be slow to
#     compile); add it if a fixed patch shape compiles cleanly.
