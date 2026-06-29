#!/bin/bash
#SBATCH --job-name=superformer-3d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --mail-user=jil202@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --time=0-99:00:00
#SBATCH --gres=gpu:8
#SBATCH --constraint=80g
#SBATCH --exclude=gpu-n80,gpu-n82

# Supervised SuperFormer 3D restoration baseline (sibling of train_flow.sh).
#
# Usage:
#   sbatch train_superformer.sh                  (default: brain, raw GT)
#   sbatch train_superformer.sh "brain" "raw"    (artifact -> fully sampled)
#   sbatch train_superformer.sh "brain" "md"     (artifact/raw -> denoised+biascorr)
#   sbatch train_superformer.sh "brain knee" "raw" "aniso"   (restrict artifacts)
#
# Positional args:
#   1 CONTRASTS   anatomy group(s): brain | knee | prostate (space-separated)
#   2 TARGET      ground truth: raw (fully sampled) | md/denoised (denoised+biascorr)
#   3 ARTIFACTS   optional artifact families: undersampled spike aniso (default all)
DATA_ROOT=/vast/tibrahim/jil202/data
CONTRASTS=${1:-brain}
TARGET=${2:-raw}
ARTIFACTS=${3:-}
echo "Data: root=$DATA_ROOT anatomies=[$CONTRASTS] target=[$TARGET] artifacts=[${ARTIFACTS:-all}]"

# Pass --artifact only when a non-empty third arg is given.
ARTIFACT_ARG=""
if [ -n "$ARTIFACTS" ]; then
    ARTIFACT_ARG="--artifact $ARTIFACTS"
fi

source activate vsr
nvidia-smi

# One process per allocated GPU.
NGPU=$(nvidia-smi -L | wc -l)
echo "Launching on $NGPU GPUs"

python -m torch.distributed.run --nproc_per_node=$NGPU train_superformer.py \
        --contrast $CONTRASTS \
        --target_type $TARGET \
        $ARTIFACT_ARG \
        --data_root $DATA_ROOT \
        --distributed --fp16 --save_model \
        --batch_size 1 --size 192 --max_epochs 100 --sample 100 \
        --lr 2e-4 --wandb --compile \
        --samples_per_contrast 2000 --balance_by anatomy_artifact --val_images_per_group 600

# Notes:
#   * --no_pretrained trains from scratch instead of the released SuperFormer weights.
#   * SuperFormer (patch_size=2) is memory-heavy at --size 192; drop --size (e.g.
#     128/96) or --batch_size if you hit OOM. --size must be a multiple of 16.
#   * --compile is omitted (SuperFormer's dynamic masks make it slow to compile);
#     add it if a fixed patch shape compiles cleanly in your environment.
