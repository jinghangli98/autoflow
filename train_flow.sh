#!/bin/bash
#SBATCH --job-name=flow-brain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --cluster=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --mail-user=jil202@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --time=0-48:00:00
#SBATCH --gres=gpu:8
#SBATCH --constraint=80g

# Usage:
#   sbatch train_flow.sh                      (default: all anatomies, all artifacts)
#   sbatch train_flow.sh "brain"              (single anatomy, all artifacts, groups balanced)
#   sbatch train_flow.sh "brain" "aniso"      (specialized anisotropic brain model)
#   sbatch train_flow.sh "brain knee" "aniso spike"  (multiple of each)
#   sbatch train_flow.sh "brain" "" "aniso 0.7"   (all artifacts, aniso up-weighted to 70%)
#   sbatch train_flow.sh "brain knee prostate" "" "" "brain=0.2 knee=0.4 prostate=0.4"
#       (20/40/40 epoch split, artifacts uniform within each anatomy)
#   sbatch train_flow.sh "brain" "" "" "" 10  (pilot: 10% of the volume pairs)
#   SAMPLES_PER_CONTRAST=none sbatch train_flow.sh "brain" "" "" "" 10
#                                             (same, natural artifact proportions)
#
# Positional args:
#   1 CONTRASTS          anatomies, subset of "brain knee prostate"
#   2 ARTIFACTS          artifact families to keep (undersampled / spike / aniso);
#                        the clean raw->denoised task is always kept. "" = all.
#   3 ARTIFACT_FRACTION  "FAMILY FRACTION" up-weights one family to that fraction
#                        of each anatomy's per-epoch pairs. Default "" = the four
#                        (anatomy, artifact) groups get equal shares of each step.
#                        Only applies with --balance_by anatomy_artifact.
#   4 ANATOMY_WEIGHT     "ANAT=FRAC ..." target epoch fractions per anatomy (must
#                        sum to 1.0). "" = equal groups.
#   5 SAMPLE             percent of training *volume pairs* to use (default 100).
#                        Deterministic subset (seed 42); validation is not subsampled.
#
# Environment overrides (export before sbatch, or prefix the command):
#   CHECKPOINT=/path/to.pt   warm-start weights (default: none, train from scratch).
#                            Loaded loosely (utils.relaxed_state_dict): a
#                            128-256-512 checkpoint warm-starts a 256-256-512
#                            model by copying each tensor's overlapping block;
#                            the new channels keep their random init. The log
#                            lists every partially copied tensor.
#   RUN_TAG=256ch_083026     inserted into checkpoint names so runs never overwrite
#                            each other; default "<C0>ch_<mmddyy>" from CHANNELS and
#                            the submission date. Also the W&B run name.
#   TEXT_COND=0              train without RadBERT text conditioning (ablation)
#   CHANNELS="256 256 512"   UNet widths per level (220M params; the original
#                            "128 256 512" is 195M). Stored in the checkpoint.
#   PATCHES_PER_VOLUME=8     patches drawn per volume pair per visit
#   BATCH_SIZE=4             volume pairs per GPU step (=> BATCH_SIZE*PATCHES_PER_VOLUME
#                            patches per GPU). Activation memory measured at
#                            ~2.3 GB/patch for 256-256-512 (~1.5 GB for 128-256-512)
#                            under bf16, so 32 patches ~ 75 GB + ~5 GB states on a
#                            96 GB RTX 6000 Pro; for 128-256-512 use BATCH_SIZE=6.
#                            If it OOMs, drop BATCH_SIZE by 1.
#   SAMPLES_PER_CONTRAST=1000 volume pairs per (anatomy, artifact) group per epoch
#                            (balanced sampler; 4 brain groups -> 4000 pairs/epoch,
#                            ~74% of the 5435 train pairs, redrawn every epoch);
#                            "none" = one pass over all selected pairs per epoch
#                            in their natural artifact proportions
#   NUM_WORKERS=12           DataLoader workers per GPU process (96 CPUs / 8 GPUs)
#   LR=2e-4 LR_MIN=1e-6      peak / floor of the cosine schedule
#   WARMUP_STEPS=1000        linear LR warmup (optimizer steps)
#   FG_FRACTION=0.25         min foreground fraction to accept a crop without
#                            re-drawing (0.0 = accept every crop, air included)
#   MAX_STEPS=               total optimizer steps: the cosine spans them and
#                            training stops there. Empty = max_epochs * steps/epoch.
#                            Size it to the allocation: read it/s off tqdm in the
#                            first minutes, then steps ~= 0.9 * hours * 3600 * it/s
#                            (e.g. 1.2 it/s * 99 h -> ~385000). Or leave empty and
#                            set MAX_EPOCHS instead.
#   MAX_EPOCHS=100000        epoch cap (an epoch = SAMPLES_PER_CONTRAST * groups
#                            pairs / (BATCH_SIZE * GPUs) steps, ~62 steps here);
#                            with MAX_STEPS set this just needs to be large
#   VAL_INTERVAL=5           validate / consider best every N epochs
#   SAVE_LAST_EVERY=10       write *_last.pt every N epochs (survives a killed job)
#   EMA_DECAY=0.9999         per-step EMA of the weights used for validation/best
#
# Data (dataset.py): whole-volume .nii.gz under /vast/tibrahim/jil202/nii laid out
# <root>/<anatomy>/<acq>/<subject>/*.nii.gz, no JSON sidecars.
#   * file roles by name: <stem> raw, md<stem> denoised+biascorrected target,
#     <stem>_R*/_SPIKE_R*/_ANISO_* and r*_lowres artifacts (d<stem> is unused)
#   * tasks: artifact->raw, artifact->denoised, raw->denoised
#   * intensities are normalized per volume at load (percentiles 0.5/99.5 -> [0,1])
#   * each visit of a pair yields PATCHES_PER_VOLUME random 96x96x7 crops; the thin
#     axis is drawn from all three array axes (sagittal/coronal/axial slabs) except
#     for TSE, which only uses its through-plane axis; the thin axis is moved last
#   * prompts are composed from the folder name (sequence, field), the NIfTI header
#     (resolution) and the plane of each patch, e.g.
#     "Input: anisotropic undersampled brain MRI. Target: Fully sampled coronal 7T
#      brain T1-weighted MPRAGE MRI of resolution 0.75 x 0.75 x 0.75 mm."
#   * validation = --val_fraction (10%) patient-level split inside the train root
#   * the held-out test set (~10% of patients, make_test_split.py) lives in
#     /vast/tibrahim/jil202/nii_test and is never read during training
#
# Schedule (train_flow.py): the LR is stepped per optimizer step -- linear warmup,
# then cosine LR -> LR_MIN over MAX_STEPS -- because epochs here are only a few
# dozen steps and an epoch-keyed cosine would be compressed onto the step axis.
# Checkpoints go to ./checkpoints_uncertainty/flow_matching_3d_<anatomies>_<artifacts>_<RUN_TAG>_{best,last,final}.pt.
DATA_ROOT=/vast/tibrahim/jil202/nii
CONTRASTS=${1:-brain knee prostate}
ARTIFACTS=${2:-}
ARTIFACT_FRACTION=${3:-}
ANATOMY_WEIGHT=${4:-}
SAMPLE=${5:-100}

CHECKPOINT=${CHECKPOINT:-}
TEXT_COND=${TEXT_COND:-1}
CHANNELS=${CHANNELS:-128 256 512}
RUN_TAG=${RUN_TAG:-${CHANNELS%% *}ch_$(date +%m%d%y)}
PATCHES_PER_VOLUME=${PATCHES_PER_VOLUME:-8}
BATCH_SIZE=${BATCH_SIZE:-4}
SAMPLES_PER_CONTRAST=${SAMPLES_PER_CONTRAST:-1000}
NUM_WORKERS=${NUM_WORKERS:-12}
LR=${LR:-2e-4}
LR_MIN=${LR_MIN:-1e-6}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
MAX_STEPS=${MAX_STEPS:-}
MAX_EPOCHS=${MAX_EPOCHS:-100000}
VAL_INTERVAL=${VAL_INTERVAL:-5}
SAVE_LAST_EVERY=${SAVE_LAST_EVERY:-10}
EMA_DECAY=${EMA_DECAY:-0.9999}
FG_FRACTION=${FG_FRACTION:-0.25}

echo "Data: root=$DATA_ROOT anatomies=[$CONTRASTS] artifacts=[${ARTIFACTS:-all}] artifact_fraction=[${ARTIFACT_FRACTION:-none}] anatomy_weight=[${ANATOMY_WEIGHT:-uniform}] sample=${SAMPLE}%"
echo "Model: channels=[${CHANNELS}] run_tag=${RUN_TAG} -> s_uncertainty/flow_matching_3d_<anatomies>_<artifacts>_${RUN_TAG}_{best,last,final}.pt"
echo "Patches: 96x96x7, ${PATCHES_PER_VOLUME}/pair, batch=${BATCH_SIZE} pairs, ${SAMPLES_PER_CONTRAST} pairs/group/epoch; text_conditioning=${TEXT_COND}; checkpoint=[${CHECKPOINT:-scratch}]"
echo "Schedule: lr=${LR} -> ${LR_MIN}, warmup=${WARMUP_STEPS} steps, max_steps=[${MAX_STEPS:-max_epochs*steps/epoch}], max_epochs=${MAX_EPOCHS}, val every ${VAL_INTERVAL} epochs, last ckpt every ${SAVE_LAST_EVERY}, ema=${EMA_DECAY}"

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

# Pass --anatomy_weight only when a non-empty fourth arg is given.
ANATOMY_WEIGHT_ARG=""
if [ -n "$ANATOMY_WEIGHT" ]; then
    ANATOMY_WEIGHT_ARG="--anatomy_weight $ANATOMY_WEIGHT"
fi

CHECKPOINT_ARG=""
if [ -n "$CHECKPOINT" ]; then
    CHECKPOINT_ARG="--checkpoint_path $CHECKPOINT"
fi

TEXT_COND_ARG=""
if [ "$TEXT_COND" = "0" ]; then
    TEXT_COND_ARG="--no_text_conditioning"
fi

MAX_STEPS_ARG=""
if [ -n "$MAX_STEPS" ]; then
    MAX_STEPS_ARG="--max_steps $MAX_STEPS"
fi

# Balanced per-group sampling unless SAMPLES_PER_CONTRAST=none.
BALANCE_ARG="--samples_per_contrast $SAMPLES_PER_CONTRAST --balance_by anatomy_artifact"
if [ "$SAMPLES_PER_CONTRAST" = "none" ]; then
    BALANCE_ARG=""
    if [ -n "$ARTIFACT_FRACTION_ARG" ]; then
        echo "ERROR: ARTIFACT_FRACTION requires balanced sampling; pass \"\" as the 3rd arg with SAMPLES_PER_CONTRAST=none" >&2
        exit 1
    fi
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
        $ANATOMY_WEIGHT_ARG \
        $CHECKPOINT_ARG \
        $TEXT_COND_ARG \
        $MAX_STEPS_ARG \
        --data_root $DATA_ROOT \
        --distributed --fp16 --save_model --compile \
        --channels $CHANNELS --run_tag $RUN_TAG \
        --size 96 --depth 7 --patches_per_volume $PATCHES_PER_VOLUME --val_fraction 0.10 \
        --fg_fraction $FG_FRACTION \
        --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --sample $SAMPLE \
        --max_epochs $MAX_EPOCHS --val_interval $VAL_INTERVAL --save_last_every $SAVE_LAST_EVERY \
        --lr $LR --lr_min $LR_MIN --warmup_steps $WARMUP_STEPS --ema_decay $EMA_DECAY \
        --num_sampling_steps 2 $BALANCE_ARG --val_images_per_group 600 \
        --cfg_dropout_prob 0.1
