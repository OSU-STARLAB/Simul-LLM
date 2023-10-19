#!/bin/bash

#SBATCH -t 5-00:00:00
#SBATCH -J falcon_finetune
#SBATCH -A eecs
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gres=gpu:4
##SBATCH -p share,gpu
##SBATCH --gres=gpu:1
#SBATCH --constraint="a40|rtx8000"
#SBATCH --mem=120G

#SBATCH -o logs/simulmt/falcon_1_epoch_finetune.log

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=agostinv@oregonstate.edu

#pip --cache-dir .cache
export HUGGINGFACE_HUB_CACHE=".cache"
export HF_DATASETS_CACHE=".cache/datasets"

export ROOT="/nfs/hpc/share/agostinv/llm-simultaneous-translation/falcon"
export VENV_ROOT="/nfs/hpc/share/agostinv/llama_simulst_venv"

source ${VENV_ROOT}/bin/activate
cd ${ROOT}

python train.py
