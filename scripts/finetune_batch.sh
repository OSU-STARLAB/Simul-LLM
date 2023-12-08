#!/bin/bash

export ROOT="<PATH_TO_REPO_ROOT>"
export VENV_ROOT="<PATH_TO_VIRTUAL_ENV>"

source ${VENV_ROOT}/bin/activate
cd ${ROOT}

export HUGGINGFACE_HUB_CACHE="${ROOT}/.cache"
export HF_DATASETS_CACHE="${ROOT}/.cache/datasets"

export PYTHONPATH="${PYHTONPATH}:${ROOT}"

python cli/finetune.py \
    --model ybelkada/falcon-7b-sharded-bf16 \
    --training-set maxolotl/must-c-en-es-wait3-02 \
    --peft --lora-alpha 16 --lora-dropout 0.1 --lora-r 64 \
    --use-4bit --bnb-4bit-compute-dtype float32 --bnb-4bit-quant-type nf4 \
    --bsz 4 --update-freq 4 \
    --optim paged_adamw_32bit --lr 2e-4 --lr-scheduler constant \
    --warmup-ratio 0.03 --max-grad-norm 0.3 \
    --save-interval 5000 --eval-interval 5000 --log-interval 100 --max-updates 100000 \
    --max-seq-length 512 \
    --output-dir ./model \
