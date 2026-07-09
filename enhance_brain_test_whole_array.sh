#!/bin/bash
#SBATCH --job-name=brain-flow-enh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --mail-user=jil202@pitt.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --time=0-72:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-588

set -euo pipefail

# One array task processes one degraded/artifacted brain volume.
#
# Default task:
#   artifacted image -> fully sampled/raw target prompt from <subject>.json
#
# Usage:
#   sbatch enhance_brain_test_whole_array.sh
#
# Count current artifact tasks before changing --array:
#   bash enhance_brain_test_whole_array.sh --count
#
# Inspect generated tasks:
#   bash enhance_brain_test_whole_array.sh --manifest | less
#
# Optional overrides:
#   CHECKPOINT_PATH=/path/to/model.pt sbatch enhance_brain_test_whole_array.sh
#   PLANES="axial coronal sagittal" sbatch enhance_brain_test_whole_array.sh
#   OVERWRITE=1 sbatch enhance_brain_test_whole_array.sh
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=18 bash enhance_brain_test_whole_array.sh

REPO_ROOT=/vast/tibrahim/jil202/autoflow
DATA_ROOT=/vast/tibrahim/jil202/data/test_whole/brain
OUT_ROOT=/vast/tibrahim/jil202/data/test_whole/brain_ai
CHECKPOINT_PATH=${CHECKPOINT_PATH:-/vast/tibrahim/jil202/autoflow/checkpoints_uncertainty/flow_matching_3d_brain_all_best_062926.pt}

NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-10}
NON_OVERLAP=${NON_OVERLAP:-8}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
PLANES=${PLANES:-}
OVERWRITE=${OVERWRITE:-0}
DRY_RUN=${DRY_RUN:-0}

if [[ -z "$PLANES" ]]; then
    # Default inference plane: axial for all brain contrasts except sagittal FLAIR.
    USE_FOLDER_PLANE=1
else
    USE_FOLDER_PLANE=0
fi

build_manifest() {
    python - "$DATA_ROOT" <<'PY'
import os
import re
import sys

data_root = sys.argv[1]
artifact_re = re.compile(r"_(ANISO|SPIKE|R\d)")

tasks = []
for dirpath, _, filenames in os.walk(data_root):
    for filename in filenames:
        if not filename.endswith(".nii.gz"):
            continue
        if not artifact_re.search(filename):
            continue

        input_path = os.path.join(dirpath, filename)
        stem = filename[:-7]
        base = re.sub(r"_(ANISO_.+|SPIKE_R\d+|R\d+)$", "", stem)
        prompt_json = os.path.join(dirpath, f"{base}.json")
        if not os.path.exists(prompt_json):
            print(f"WARNING missing prompt JSON for {input_path}: {prompt_json}", file=sys.stderr)
            continue

        rel = os.path.relpath(input_path, data_root)
        contrast = rel.split(os.sep, 1)[0]
        if contrast in {"flair_sag_3T", "flair_3t_sagittal"}:
            plane = "sagittal"
        else:
            plane = "axial"

        tasks.append((input_path, prompt_json, rel, plane))

for task in sorted(tasks, key=lambda x: x[2]):
    print("\t".join(task))
PY
}

if [[ "${1:-}" == "--count" ]]; then
    build_manifest | wc -l
    exit 0
fi

if [[ "${1:-}" == "--manifest" ]]; then
    build_manifest
    exit 0
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
MANIFEST=${SLURM_TMPDIR:-/tmp}/brain_flow_enhance_manifest_${SLURM_JOB_ID:-manual}.tsv
build_manifest > "$MANIFEST"

NUM_TASKS=$(wc -l < "$MANIFEST")
if (( TASK_ID < 1 || TASK_ID > NUM_TASKS )); then
    echo "Task ${TASK_ID} is outside manifest size ${NUM_TASKS}; exiting."
    exit 0
fi

line=$(sed -n "${TASK_ID}p" "$MANIFEST")
IFS=$'\t' read -r INPUT_PATH PROMPT_JSON REL_PATH FOLDER_PLANE <<< "$line"
OUTPUT_PATH="${OUT_ROOT}/${REL_PATH}"
mkdir -p "$(dirname "$OUTPUT_PATH")"

WOULD_SKIP=0
if [[ "$OVERWRITE" != "1" && -s "$OUTPUT_PATH" ]]; then
    WOULD_SKIP=1
fi

if [[ "$WOULD_SKIP" == "1" && "$DRY_RUN" != "1" ]]; then
    echo "Output exists, skipping: $OUTPUT_PATH"
    exit 0
fi

if (( USE_FOLDER_PLANE )); then
    RUN_PLANES=("$FOLDER_PLANE")
else
    read -r -a RUN_PLANES <<< "$PLANES"
fi

echo "SLURM task ${TASK_ID}/${NUM_TASKS}"
echo "Input:       $INPUT_PATH"
echo "Prompt JSON: $PROMPT_JSON"
echo "Output:      $OUTPUT_PATH"
echo "Planes:      ${RUN_PLANES[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
    if [[ "$WOULD_SKIP" == "1" ]]; then
        echo "Existing output found; a real run would skip unless OVERWRITE=1."
    fi
    echo "Dry run only; not launching enhance_flow_3d.py."
    exit 0
fi

if [[ -f /ihome/tibrahim/jil202/miniconda3/etc/profile.d/conda.sh ]]; then
    source /ihome/tibrahim/jil202/miniconda3/etc/profile.d/conda.sh
    conda activate vsr
else
    source activate vsr
fi
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd "$REPO_ROOT"
nvidia-smi

python enhance_flow_3d.py \
      --checkpoint_path "$CHECKPOINT_PATH" \
      --input_path "$INPUT_PATH" \
      --output_path "$OUTPUT_PATH" \
      --prompt_json "$PROMPT_JSON" \
      --num_sampling_steps "$NUM_SAMPLING_STEPS" \
      --batch_size "$BATCH_SIZE" \
      --euler \
      --fp16 \
      --non_overlap "$NON_OVERLAP" \
      --guidance_scale "$GUIDANCE_SCALE" \
      --compile \
      --planes "${RUN_PLANES[@]}" \
      --autocrop --rescale
