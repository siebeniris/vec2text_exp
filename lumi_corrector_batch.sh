#!/bin/bash -e

# Batch launcher: train one corrector per (victim, sample_size) combination
# on top of the matching inverter checkpoint. Skips pairs where the inverter
# folder doesn't exist (so e.g. clip_1 is handled automatically).
#
# Usage:
#   bash lumi_corrector_batch.sh

set -x

# ---------------------------------------------------------------------------
# Shared training hyperparameters
# ---------------------------------------------------------------------------
DATASET_NAME="coco_image_victim_first_caption"
MODEL_NAME="google/flan-t5-base"
BATCH_SIZE=32
NUM_EPOCHS=20
LEARNING_RATE="2e-4"
MAX_SEQ_LENGTH=32
NUM_REPEAT_TOKENS=16
MAX_EVAL_SAMPLES=500
EVAL_STEPS=20000
EMBEDDING_OUTPUT="last_hidden_state"
USE_RANDOM_EMBEDDINGS=0
USE_WANDB=0
EXP_GROUP="lumi-image-corrector-sweep"
USE_LESS_DATA=-1
EARLY_STOPPING="no"
OVERWRITE_OUTPUT_DIR=1

# Inference-time iteration calls the embedder; training doesn't (frozen embeds).
# Set this to any HF-resolvable name — used as a placeholder.
EMBEDDER_MODEL_PLACEHOLDER="nomic-ai/nomic-embed-text-v1"

INVERTERS_ROOT="saves/inverters"
CORRECTORS_ROOT="saves/correctors"

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
VICTIMS=(clip cohere gemini2 nomic nvidia random)
SAMPLE_SIZES=(1 10 100 1000 10000)

submitted=0
skipped=0

for VICTIM in "${VICTIMS[@]}"; do
  for N in "${SAMPLE_SIZES[@]}"; do

    INVERSION_MODEL_PATH="${INVERTERS_ROOT}/coco_image_${VICTIM}_${N}"
    OUTPUT_DIR="${CORRECTORS_ROOT}/coco_image_${VICTIM}_${N}"
    EXP_NAME="${VICTIM}_${N}"

    if [ ! -d "${INVERSION_MODEL_PATH}" ]; then
      echo "Skipping ${VICTIM}_${N}: inverter not found at ${INVERSION_MODEL_PATH}"
      skipped=$((skipped + 1))
      continue
    fi

    echo "Submitting corrector: victim=${VICTIM}  samples=${N}  inverter=${INVERSION_MODEL_PATH}"

    sbatch lumi_train_corrector.sh \
      "${DATASET_NAME}" \
      "${VICTIM}" \
      "${OUTPUT_DIR}" \
      "${MODEL_NAME}" \
      "${EMBEDDER_MODEL_PLACEHOLDER}" \
      "${BATCH_SIZE}" \
      "${NUM_EPOCHS}" \
      "${LEARNING_RATE}" \
      "${MAX_SEQ_LENGTH}" \
      "${INVERSION_MODEL_PATH}" \
      "${NUM_REPEAT_TOKENS}" \
      "${MAX_EVAL_SAMPLES}" \
      "${EVAL_STEPS}" \
      "" \
      "${EMBEDDING_OUTPUT}" \
      "${USE_RANDOM_EMBEDDINGS}" \
      "${USE_WANDB}" \
      "${EXP_GROUP}" \
      "${EXP_NAME}" \
      "${USE_LESS_DATA}" \
      "${EARLY_STOPPING}" \
      "${OVERWRITE_OUTPUT_DIR}" \
      "${N}"

    submitted=$((submitted + 1))
  done
done

echo ""
echo "Done. Submitted ${submitted} corrector jobs, skipped ${skipped} missing pairs."
