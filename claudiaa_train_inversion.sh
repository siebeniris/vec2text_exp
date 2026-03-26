#!/bin/bash -e
#SBATCH --job-name=vec2text_inversion
#SBATCH --output=logs/vec2text_inversion_%j.out
#SBATCH --error=logs/vec2text_inversion_%j.err
#SBATCH --mem=80GB
#SBATCH --time=2-00:00:00

# Single-GPU inversion training launcher for Claudiaa/server usage.
# Example:
# sbatch claudiaa_train_inversion.sh \
#   coco_victim_first_caption \
#   nomic \
#   saves/inverters/coco_nomic \
#   google/mt5-base \
#   nomic-ai/nomic-embed-text-v1 \
#   32 \
#   20 \
#   2e-4 \
#   32

set -x

wd=$(pwd)
echo "Working directory: ${wd}"

mkdir -p logs

SIF=/home/cs.aau.dk/ng78zb/pytorch_25.06-py3.sif
echo "Using container: ${SIF}"

DATASET_NAME=${1}
VICTIM_EMBEDDING=${2}
OUTPUT_DIR=${3}
MODEL_NAME=${4}
EMBEDDER_MODEL=${5}
BATCH_SIZE=${6}
NUM_EPOCHS=${7}
LEARNING_RATE=${8}
MAX_SEQ_LENGTH=${9}
NUM_REPEAT_TOKENS=${10:-16}
MAX_EVAL_SAMPLES=${11:-200}
EVAL_STEPS=${12:-20000}
WARMUP_STEPS=${13:-""}
EMBEDDING_OUTPUT=${14:-"last_hidden_state"}
USE_RANDOM_EMBEDDINGS=${15:-0}
USE_FROZEN_EMBEDDINGS=${16:-"True"}
USE_WANDB=${17:-0}
EXP_GROUP_NAME=${18:-"server-inversion"}
EXP_NAME=${19:-"${DATASET_NAME}_${VICTIM_EMBEDDING}"}
USE_LESS_DATA=${20:--1}
EARLY_STOPPING=${21:-"no"}
OVERWRITE_OUTPUT_DIR=${22:-0}

if [ -z "${WARMUP_STEPS}" ]; then
  WARMUP_STEPS=${EVAL_STEPS}
fi

echo "Configuration:"
echo "  Dataset: ${DATASET_NAME}"
echo "  Victim embedding: ${VICTIM_EMBEDDING}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Model name: ${MODEL_NAME}"
echo "  Embedder model: ${EMBEDDER_MODEL}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Max seq length: ${MAX_SEQ_LENGTH}"
echo "  Num repeat tokens: ${NUM_REPEAT_TOKENS}"
echo "  Max eval samples: ${MAX_EVAL_SAMPLES}"
echo "  Eval steps: ${EVAL_STEPS}"
echo "  Warmup steps: ${WARMUP_STEPS}"
echo "  Embedding output: ${EMBEDDING_OUTPUT}"
echo "  Use random embeddings: ${USE_RANDOM_EMBEDDINGS}"
echo "  Use frozen embeddings: ${USE_FROZEN_EMBEDDINGS}"
echo "  Use W&B: ${USE_WANDB}"
echo "  Exp group: ${EXP_GROUP_NAME}"
echo "  Exp name: ${EXP_NAME}"
echo "  Use less data: ${USE_LESS_DATA}"
echo "  Early stopping metric: ${EARLY_STOPPING}"
echo "  Overwrite output dir: ${OVERWRITE_OUTPUT_DIR}"

TRAIN_ARGS=(
  --dataset_name "${DATASET_NAME}"
  --victim_embedding_name "${VICTIM_EMBEDDING}"
  --output_dir "${OUTPUT_DIR}"
  --model_name_or_path "${MODEL_NAME}"
  --embedder_model_name "${EMBEDDER_MODEL}"
  --per_device_train_batch_size "${BATCH_SIZE}"
  --per_device_eval_batch_size "${BATCH_SIZE}"
  --num_train_epochs "${NUM_EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --max_seq_length "${MAX_SEQ_LENGTH}"
  --num_repeat_tokens "${NUM_REPEAT_TOKENS}"
  --max_eval_samples "${MAX_EVAL_SAMPLES}"
  --eval_steps "${EVAL_STEPS}"
  --save_steps "${EVAL_STEPS}"
  --warmup_steps "${WARMUP_STEPS}"
  --experiment inversion
  --embedding_output "${EMBEDDING_OUTPUT}"
  --use_frozen_embeddings_as_input "${USE_FROZEN_EMBEDDINGS}"
  --embedder_no_grad True
  --ddp_find_unused_parameters True
  --use_wandb "${USE_WANDB}"
  --exp_group_name "${EXP_GROUP_NAME}"
  --exp_name "${EXP_NAME}"
  --use_less_data "${USE_LESS_DATA}"
  --apply_early_stopping_metric "${EARLY_STOPPING}"
)

if [ "${USE_RANDOM_EMBEDDINGS}" -eq 1 ]; then
  TRAIN_ARGS+=(--use_random_embeddings)
fi

if [ "${OVERWRITE_OUTPUT_DIR}" -eq 1 ]; then
  TRAIN_ARGS+=(--overwrite_output_dir)
fi

echo ""
srun singularity exec \
  --nv --cleanenv --bind "${wd}:${wd}" \
  "${SIF}" bash -lc "cd '${wd}' && export PYTHONPATH='${wd}:\${PYTHONPATH}' && python -m vec2text.run ${TRAIN_ARGS[*]}"

echo ""
echo "Inversion training completed!"
