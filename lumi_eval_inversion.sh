#!/bin/bash -e
#SBATCH --job-name=vec2text_eval
#SBATCH --account=project_465002358
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/vec2text_eval_%j.out
#SBATCH --error=logs/vec2text_eval_%j.err

# LUMI evaluation launcher for eval_inversion.py.
# Example:
# sbatch lumi_eval_inversion.sh \
#   saves/inverters/coco_clip \
#   clip

set -x

wd=$(pwd)
echo "Working directory: ${wd}"

mkdir -p logs

export HF_HOME="/scratch/project_465002358/.cache"
export HF_DATASETS_CACHE="/scratch/project_465002358/.cache/datasets"
export DATASET_CACHE_PATH="/scratch/project_465002358/.cache"
export EBU_USER_PREFIX=/scratch/project_465002358/

echo "Transformers cache $HF_HOME"
echo "HF datasets cache $HF_DATASETS_CACHE"

#### Set up for ROCm.
export NCCL_P2P_LEVEL=PHB
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3

export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

SIF=/scratch/project_465002358/multivec2text.sif
echo "Using container: ${SIF}"

chmod +x $HF_HOME
chmod +x $HF_DATASETS_CACHE

MODEL_PATH=${1}
VICTIM_NAME=${2}
BATCH_SIZE=${3:-32}
NUM_BEAMS=${4:-4}
MAX_NEW_TOKENS=${5:-64}
MAX_SAMPLES=${6:--1}
OUTPUT=${7:-"${MODEL_PATH}/eval_results_${NUM_BEAMS}.json"}
SPLIT=${8:-"test"}
EMBED_ROOT=${9:-""}
CAPTION_ROOT=${10:-""}

echo "Configuration:"
echo "  Model path:     ${MODEL_PATH}"
echo "  Victim name:    ${VICTIM_NAME}"
echo "  Batch size:     ${BATCH_SIZE}"
echo "  Num beams:      ${NUM_BEAMS}"
echo "  Max new tokens: ${MAX_NEW_TOKENS}"
echo "  Max samples:    ${MAX_SAMPLES}"
echo "  Output:         ${OUTPUT}"
echo "  Split:          ${SPLIT}"
echo "  Embed root:     ${EMBED_ROOT:-<default>}"
echo "  Caption root:   ${CAPTION_ROOT:-<default>}"

EVAL_ARGS=(
    --model_path "${MODEL_PATH}"
    --victim_name "${VICTIM_NAME}"
    --batch_size "${BATCH_SIZE}"
    --num_beams "${NUM_BEAMS}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --max_samples "${MAX_SAMPLES}"
    --split "${SPLIT}"
    --output "${OUTPUT}"
)
if [ -n "${EMBED_ROOT}" ]; then
    EVAL_ARGS+=(--embed_root "${EMBED_ROOT}")
fi
if [ -n "${CAPTION_ROOT}" ]; then
    EVAL_ARGS+=(--caption_root "${CAPTION_ROOT}")
fi

echo ""
srun singularity exec \
    -B /scratch/project_465002358:/scratch/project_465002358 \
    -B ${wd}:${wd} \
    -B ${HF_HOME}:${HF_HOME} \
    -B ${HF_DATASETS_CACHE}:${HF_DATASETS_CACHE} \
    ${SIF} bash -c "cd '${wd}' && export PYTHONPATH='${wd}:\${PYTHONPATH}' && python eval_inversion.py ${EVAL_ARGS[*]}"

echo ""
echo "Evaluation completed! Results saved to ${OUTPUT}"
