#!/bin/bash -e

#SBATCH --job-name=eval
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err
#SBATCH --mem=100GB
#SBATCH --time=2-00:00:00

set -x

MODEL_NAME=$1


wd=$(pwd)
echo "working directory ${wd}"

export HF_HOME="${wd}/.cache"
export HF_DATASETS_CACHE="${wd}/.cache/datasets"
export DATASET_CACHE_PATH="${wd}/.cache"
export WANDB_CACHE_DIR="${wd}/.cache/wandb/artifcats/"


export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
# tokenization timeouts handling
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0



SIF=/home/cs.aau.dk/ng78zb/pytorch_23.10-py3.sif
echo "sif ${SIF}"


echo "launch evaluation ${MODEL_NAME}"

srun singularity exec --nv --cleanenv --bind ${wd}:${wd} ${SIF} \
  python -m evaluation ${MODEL_NAME}