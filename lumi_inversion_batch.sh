#!/bin/bash -e

# Batch launcher: train one inverter per (victim, sample_size) combination.
# Submits 4 victims × 4 sample sizes = 16 sbatch jobs.
#
# Usage:
#   bash lumi_inversion_batch.sh

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
EVAL_STEPS=20000
EXP_GROUP="lumi-image-inversion-sweep"

# ---------------------------------------------------------------------------
# Victim → embedder model name mapping
# Embeddings are precomputed (use_frozen_embeddings=True), so the embedder is
# only used for tokenizer/config shape. Any HF-resolvable name works.
# ---------------------------------------------------------------------------
declare -A EMBEDDER_MODEL
EMBEDDER_MODEL["cohere"]="nomic-ai/nomic-embed-text-v1"
EMBEDDER_MODEL["gemini2"]="nomic-ai/nomic-embed-text-v1"
EMBEDDER_MODEL["nvidia"]="nomic-ai/nomic-embed-text-v1"
EMBEDDER_MODEL["random"]="nomic-ai/nomic-embed-text-v1"

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
VICTIMS=(cohere gemini2 nvidia random)
SAMPLE_SIZES=(1 10 100 1000 10000)

for VICTIM in "${VICTIMS[@]}"; do
  for N in "${SAMPLE_SIZES[@]}"; do

    OUTPUT_DIR="saves/inverters/coco_image_${VICTIM}_${N}"
    EXP_NAME="${VICTIM}_${N}"
    EMBEDDER="${EMBEDDER_MODEL[$VICTIM]}"

    echo "Submitting: victim=${VICTIM}  samples=${N}  output=${OUTPUT_DIR}"

    sbatch lumi_train_inversion.sh \
      "${DATASET_NAME}" \
      "${VICTIM}" \
      "${OUTPUT_DIR}" \
      "${MODEL_NAME}" \
      "${EMBEDDER}" \
      "${BATCH_SIZE}" \
      "${NUM_EPOCHS}" \
      "${LEARNING_RATE}" \
      "${MAX_SEQ_LENGTH}" \
      "${NUM_REPEAT_TOKENS}" \
      "${N}" \
      "${EVAL_STEPS}" \
      "" \
      "last_hidden_state" \
      0 \
      "True" \
      0 \
      "${EXP_GROUP}" \
      "${EXP_NAME}"

  done
done

echo ""
echo "All jobs submitted."
