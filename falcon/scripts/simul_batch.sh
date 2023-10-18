#!/bin/bash

#SBATCH -t 5-00:00:00
#SBATCH -J falcon_simulst
#SBATCH -A eecs
#SBATCH -c 4
#SBATCH -p dgx
#SBATCH --gres=gpu:1
##SBATCH -p share,gpu
##SBATCH --gres=gpu:1
##SBATCH --constraint="a40|rtx8000"
#SBATCH --mem=120G

#SBATCH -o logs/simulmt/falcon_simulmt_test.log

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=agostinv@oregonstate.edu

export HUGGINGFACE_HUB_CACHE="/nfs/hpc/share/agostinv/llama_simulst/falcon/.cache"
export HF_DATASETS_CACHE="/nfs/hpc/share/agostinv/llama_simulst/falcon/.cache/datasets"

source /nfs/hpc/share/agostinv/llama_simulst_venv/bin/activate
cd /nfs/hpc/share/agostinv/llama_simulst/SimulEval

simuleval \
    --agent /nfs/hpc/share/agostinv/llama_simulst/falcon/falcon_simulmt_agent.py \
    --source /nfs/hpc/share/starlab/victor/en-es/data/tst-COMMON/txt/tst-COMMON-01-fixed.en \
    --target /nfs/hpc/share/starlab/victor/en-es/data/tst-COMMON/txt/tst-COMMON-01-fixed.es\
    --model-path /nfs/hpc/share/agostinv/llama_simulst/falcon/model/checkpoint-1000 \
    --output output_falcon_waitk_3_basic_test \
    --waitk 3 \
    #--force-finish \
