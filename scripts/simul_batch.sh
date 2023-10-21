#!/bin/bash

#SBATCH -t 5-00:00:00
#SBATCH -J falcon_simulmt
#SBATCH -A eecs
#SBATCH -c 4
#SBATCH -p dgx
#SBATCH --gres=gpu:1
##SBATCH --constraint="a40|rtx8000"
#SBATCH --mem=120G

#SBATCH -o logs/simulmt/falcon_simulmt_test.log

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<some_email_you_would_use>

export ROOT="<PATH_TO_REPO_ROOT>"
export VENV_ROOT="<PATH_TO_VIRTUAL_ENV>"

source ${VENV_ROOT}/bin/activate
cd ${ROOT}

export HUGGINGFACE_HUB_CACHE="${ROOT}/.cache"
export HF_DATASETS_CACHE="${ROOT}/.cache/datasets"

export PYTHONPATH="${PYHTONPATH}:."

cd SimulEval

simuleval \
    --agent <PATH_TO_AGENT> \
    --source /nfs/hpc/share/starlab/victor/en-es/data/tst-COMMON/txt/tst-COMMON-fixed-reduced.en \
    --target /nfs/hpc/share/starlab/victor/en-es/data/tst-COMMON/txt/tst-COMMON-fixed-reduced.es \
    --model-path <PATH_TO_MODEL_CHECKPOINT> \
    --output output_falcon_waitk_3_basic_test \
    --waitk 3 \
