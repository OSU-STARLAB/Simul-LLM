#!/bin/bash

#SBATCH -t 5-00:00:00
#SBATCH -J falcon_finetune
#SBATCH -A eecs
#SBATCH -c 4
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --constraint="a40|rtx8000"
#SBATCH --mem=120G

#SBATCH -o logs/finetune/falcon_1_epoch_finetune.log

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<some_email_you_would_use>

#pip --cache-dir .cache
export ROOT="<PATH_TO_REPO_ROOT>"
export VENV_ROOT="<PATH_TO_VIRTUAL_ENV>"

source ${VENV_ROOT}/bin/activate
cd ${ROOT}

export HUGGINGFACE_HUB_CACHE="${ROOT}/.cache"
export HF_DATASETS_CACHE="${ROOT}/.cache/datasets"

export PYTHONPATH="${PYHTONPATH}:."

python cli/finetune.py \
    --model ybelkada/falcon-7b-sharded-bf16 \
    --training-set maxolotl/falcon_w3_en_es_v2 \
    --lora-alpha 16 --lora-dropout 0.1 --lora-r 64 \
    --use-4bit --bnb-4bit-compute-dtype float16 --bnb-4bit-quant-type nf4 \
    --bsz 4 --update-freq 4 \
    --optim paged_adamw_32bit --lr 2e-4 --lr-scheduler constant \
    --warmup-ratio 0.03 --max-grad-norm 0.3 \
    --save-interval 5000 --log-interval 100 --max-updates 100000 \
    --max-seq-length 1024 \
    --output-dir ./model \
