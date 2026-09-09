#!/bin/bash
#SBATCH --job-name=realesrgan-2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --mail-user=jil202@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --time=0-48:00:00
#SBATCH --gres=gpu:4

# 2D Real-ESRGAN restoration baseline (sibling of train_swinir.sh). Trains with
# L1 only for the first --gan_start_epoch epochs, then adds perceptual+GAN loss
# and starts training the discriminator (real basicsr RRDBNet + UNetDiscriminatorSN).
#
# Usage:
#   sbatch train_realesrgan.sh                       (default: brain, raw GT)
#   sbatch train_realesrgan.sh "brain" "raw"         (artifact -> fully sampled)
#   sbatch train_realesrgan.sh "brain" "md"          (artifact/raw -> denoised+biascorr)
#   sbatch train_realesrgan.sh "brain knee" "raw" "aniso"   (restrict artifacts)
#   sbatch train_realesrgan.sh "brain" "raw" "" /path/to/ckpt.pt   (warm-start generator)
#
# Positional args:
#   1 CONTRASTS   anatomy group(s): brain | knee | prostate (space-separated)
#   2 TARGET      ground truth: raw (fully sampled) | md/denoised (denoised+biascorr)
#   3 ARTIFACTS   optional artifact families: undersampled spike aniso (default all)
#   4 PRETRAINED  optional checkpoint (.pt) to warm-start the generator from
#                 (default: train from scratch). Pass "" for ARTIFACTS to skip it.
#
# Environment overrides:
#   FG_FRACTION=0.25   min foreground fraction to accept a crop without
#                      re-drawing (same rule as train_flow; 0.0 = accept all)
#   RUN_TAG=090726_fg0 inserted into checkpoint names so runs never overwrite
#                      each other; default <mmddyy>. Also the W&B run name.
#   GAN_START_EPOCH=N  epochs of pure L1 before the discriminator+perceptual
#                      losses join. Default 50 from scratch, but only 10 when
#                      PRETRAINED is given: a warm-started generator just needs
#                      a short L1 phase to adapt (e.g. to the percentile
#                      normalization) before adversarial training begins.
#   VAL_INTERVAL=5     validate / consider best every N epochs (train_flow's 5)
#   EMA_DECAY=0.9999   per-step EMA of the generator used for validation/best
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
if [ -n "$PRETRAINED" ]; then
    GAN_START_EPOCH=${GAN_START_EPOCH:-10}
else
    GAN_START_EPOCH=${GAN_START_EPOCH:-50}
fi
echo "Data: root=$DATA_ROOT anatomies=[$CONTRASTS] target=[$TARGET] artifacts=[${ARTIFACTS:-all}] pretrained=[${PRETRAINED:-none}] fg_fraction=${FG_FRACTION} run_tag=${RUN_TAG} gan_start_epoch=${GAN_START_EPOCH}"

ARTIFACT_ARG=""
if [ -n "$ARTIFACTS" ]; then
    ARTIFACT_ARG="--artifact $ARTIFACTS"
fi

CKPT_ARG=""
if [ -n "$PRETRAINED" ]; then
    CKPT_ARG="--checkpoint_path $PRETRAINED"
fi

source activate vsr
nvidia-smi

NGPU=$(nvidia-smi -L | wc -l)
echo "Launching on $NGPU GPUs"

python -m torch.distributed.run --nproc_per_node=$NGPU train_realesrgan.py \
        --contrast $CONTRASTS \
        --target_type $TARGET \
        $ARTIFACT_ARG \
        $CKPT_ARG \
        --data_root $DATA_ROOT \
        --distributed --fp16 --save_model \
        --batch_size 4 --slices_per_volume 8 --size 96 --max_epochs 150 --sample 100 \
        --fg_fraction $FG_FRACTION --run_tag $RUN_TAG --num_workers 12 \
        --val_interval $VAL_INTERVAL --ema_decay $EMA_DECAY \
        --scheduler_milestones 5000 8000 11000 \
        --nf 96 --nb 32 --gc 48 --nf_d 96 \
        --gan_start_epoch $GAN_START_EPOCH --wandb \
        --samples_per_contrast 1000 --balance_by anatomy_artifact --val_images_per_group 600 --compile

# Notes:
#   * Trains from scratch by default. Pass a 4th positional arg (a
#     train_realesrgan.py checkpoint) to warm-start the generator weights.
#   * --gan_start_epoch 20 means the first 20 epochs are pure L1 (no
#     discriminator forward/backward at all); perceptual+GAN join at epoch 20.
#   * --feature_weight 0 disables the VGG19 perceptual loss (and skips
#     building VGG). --gan_type lsgan for least-squares GAN.
#   * --size must stay a multiple of 4 (RRDBNet's internal pixel-unshuffle).
#   * --nf 96 --nb 32 --gc 48 --nf_d 96 ("bigger all around") -> ~52M param
#     generator (~209MB) + ~9.8M param discriminator (~39MB), ~248MB checkpoint
#     (G+D), vs. the paper-default --nf 64 --nb 23 --gc 32 --nf_d 64 (~84MB).
#     If you OOM, lower --batch_size before shrinking these back down.
