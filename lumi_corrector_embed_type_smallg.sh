#!/bin/bash -e
#SBATCH --job-name=corrector
#SBATCH --account=project_465000909
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=3-00:00:00
#SBATCH --output=corrector_%j.out
#SBATCH --error=corrector_%j.err




set -x

LANG=$1
EMBEDDER=$2
DATASET=$3
EXP_GROUP_NAME=$4
BATCH_SIZE=$5
MAX_LENGTH=$6
LEARNING_RATE=$7
CORRECTOR_ALIAS=$8
EPOCHS=$9
EARLY_STOPPING=${10}
OVERWRITE_OUTPUT_DIR=${11}



wd=$(pwd)
echo "working directory ${wd}"

export OPENAI_API_KEY="sk-proj-9GTzPysUslKPyHRxDWUxT3BlbkFJt9KdXvzK18UtedxlsWqK"
export HF_HOME="/scratch/project_465000909/.cache"
export HF_DATASETS_CACHE="/scratch/project_465000909/.cache/datasets"
export DATASET_CACHE_PATH="/scratch/project_465000909/.cache"
export EBU_USER_PREFIX=/scratch/project_465000909/
export WANDB_CACHE_DIR="/scratch/project_465000909/.cache/wandb/artifcats/"

echo "Trnasformers cache $HF_HOME"
echo "HF datasets cache $HF_DATASETS_CACHE"

echo "language $LANG "
echo "model mt5 embedder $EMBEDDER, epochs $EPOCHS,batch size $BATCH_SIZE max length $MAX_LENGTH " # google/mt5-base
echo "apply early stopping metric => $EARLY_STOPPING"
echo "dataset $DATASET" # mt-ms_fin_Latn
echo "exp_group_name $EXP_GROUP_NAME"
# echo "over write ouptutdir $OVERWRITE_OUTPUT_DIR"

#### set up for ROCm.
export NCCL_P2P_LEVEL=PHB
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT

# see if actual kernels are called, if they finish or get stuck
#export AMD_LOG_LEVEL=4

# https://pytorch.org/docs/stable/notes/cuda.html#environment-variables
# allocations can later be expanded to better handle changing batch size.
# export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True  # this does not work
# ProcessGroupNCCL's watchdog got stuck for 600seconds without making progress in monitoring enqueued collectives.
# export TORCH_NCCL_ENABLE_MONITORING=0
# tokenization timeouts handling
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
#export TORCH_NCCL_TRACE_BUFFER_SIZE > 0

# Set interfaces to be used by RCCL.
# This is needed as otherwise RCCL tries to use a network interface it has
# no access to on LUMI.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3

# solve the problem of leaked semaphore objects error.
#export SINGULARITYENV_CXI_FORK_SAFE=0
#export SINGULARITYENV_CXI_FORK_SAFE_HP=0

export MASTER_PORT=25900
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)

echo "Rank $SLURM_PROCID --> $(taskset -p $$); GPU $ROCR_VISIBLE_DEVICES"

# pytorch multiprocessing. semaphore.
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

SIF=/scratch/project_465000909/multivec2text.sif

# each GPU has a mask, communicating with the closest CPUs.
CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

echo $SIF
chmod +x $HF_HOME
chmod +x $HF_DATASETS_CACHE

if [ $OVERWRITE_OUTPUT_DIR -eq 1 ]; then
  srun --cpu-bind=mask_cpu:$CPU_BIND_MASKS singularity exec \
    -B /scratch/project_465000909:/scratch/project_465000909 \
    -B ${wd}:${wd} \
    -B ${HF_HOME}:${HF_HOME} \
    -B ${HF_DATASETS_CACHE}:${HF_DATASETS_CACHE} \
    ${SIF} bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID
      python -m vec2text.run --per_device_train_batch_size ${BATCH_SIZE} \
          --per_device_eval_batch_size ${BATCH_SIZE} --max_seq_length ${MAX_LENGTH} \
          --model_name_or_path google/mt5-base \
          --dataset_name ${DATASET} --embedder_model_name ${EMBEDDER} \
          --num_repeat_tokens 16 --embedder_no_grad True --num_train_epochs ${EPOCHS} --max_eval_samples 500 \
          --eval_steps 20000 --warmup_steps 10000 --experiment corrector \
          --exp_group_name ${EXP_GROUP_NAME} --exp_name ${LANG} \
          --output_dir ./saves/correctors/mt5_${EMBEDDER}_${DATASET}_${MAX_LENGTH}_2layers_prefix --save_steps 2000 \
          --apply_early_stopping_metric ${EARLY_STOPPING} \
          --learning_rate ${LEARNING_RATE} \
          --corrector_model_alias ${CORRECTOR_ALIAS} \
          --ddp_find_unused_parameters True \
          --use_frozen_embeddings_as_input True \
          --embedding_output first_last \
          --overwrite_output_dir"
else
  echo "no overwrite parameters"
  srun --cpu-bind=mask_cpu:$CPU_BIND_MASKS singularity exec \
    -B /scratch/project_465000909:/scratch/project_465000909 \
    -B ${wd}:${wd} \
    -B ${HF_HOME}:${HF_HOME} \
    -B ${HF_DATASETS_CACHE}:${HF_DATASETS_CACHE} \
    ${SIF} bash -c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID
      python -m vec2text.run --per_device_train_batch_size ${BATCH_SIZE} \
          --per_device_eval_batch_size ${BATCH_SIZE} --max_seq_length ${MAX_LENGTH} \
          --model_name_or_path google/mt5-base \
          --dataset_name ${DATASET} --embedder_model_name ${EMBEDDER} \
          --num_repeat_tokens 16 --embedder_no_grad True --num_train_epochs ${EPOCHS} --max_eval_samples 500 \
          --eval_steps 20000 --warmup_steps 10000 --experiment corrector \
          --exp_group_name ${EXP_GROUP_NAME} --exp_name ${LANG} \
          --output_dir ./saves/correctors/mt5_${EMBEDDER}_${DATASET}_${MAX_LENGTH}_2layers_prefix --save_steps 2000 \
          --apply_early_stopping_metric ${EARLY_STOPPING} \
          --corrector_model_alias ${CORRECTOR_ALIAS} \
          --ddp_find_unused_parameters True \
          --use_frozen_embeddings_as_input True \
          --embedding_output first_last \
          --learning_rate ${LEARNING_RATE} "
fi
