#!/bin/bash -e

# Batch evaluation launcher: evaluate one inverter per (victim, sample_size) combination.
# Submits 4 victims × 4 sample sizes = 16 sbatch jobs.
#
# Usage:
#   bash lumi_eval_inversion_batch.sh

set -x

# ---------------------------------------------------------------------------
# Shared eval hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE=32
NUM_BEAMS=4
MAX_NEW_TOKENS=64
MAX_SAMPLES=-1   # -1 = evaluate on all samples in the split
SPLIT="val"
EMBED_ROOT="data/coco2014captions/embeds/victim_embeddings"
CAPTION_ROOT="data/coco2014captions"

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
VICTIMS=(cohere gemini2 nvidia random)
SAMPLE_SIZES=(1 10 100 1000 10000)

for VICTIM in "${VICTIMS[@]}"; do
  for N in "${SAMPLE_SIZES[@]}"; do

    MODEL_PATH="saves/inverters/coco_image_${VICTIM}_${N}"
    OUTPUT="${MODEL_PATH}/eval_${SPLIT}_results.json"

    echo "Submitting: victim=${VICTIM}  samples=${N}  split=${SPLIT}  model=${MODEL_PATH}"

    sbatch lumi_eval_inversion.sh \
      "${MODEL_PATH}" \
      "${VICTIM}" \
      "${BATCH_SIZE}" \
      "${NUM_BEAMS}" \
      "${MAX_NEW_TOKENS}" \
      "${MAX_SAMPLES}" \
      "${OUTPUT}" \
      "${SPLIT}" \
      "${EMBED_ROOT}" \
      "${CAPTION_ROOT}"

  done
done

echo ""
echo "All evaluation jobs submitted."
