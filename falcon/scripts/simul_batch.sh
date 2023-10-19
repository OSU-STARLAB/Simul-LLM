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

export ROOT="/nfs/hpc/share/agostinv/llm-simultaneous-translation/falcon"
export VENV_ROOT="/nfs/hpc/share/agostinv/llama_simulst_venv"
export SIMUL_ROOT="/nfs/hpc/share/agostinv/llm-simultaneous-translation/SimulEval"

source ${VENV_ROOT}/bin/activate
cd ${SIMUL_ROOT}

simuleval \
    --agent /nfs/hpc/share/agostinv/llama_simulst/falcon/falcon_simulmt_agent.py \
    --source /nfs/hpc/share/starlab/victor/en-es/data/tst-COMMON/txt/tst-COMMON-fixed-reduced.en \
    --target /nfs/hpc/share/starlab/victor/en-es/data/tst-COMMON/txt/tst-COMMON-fixed-reduced.es \
    --model-path /nfs/hpc/share/agostinv/llama_simulst/falcon/model/checkpoint-1000 \
    --output output_falcon_waitk_3_basic_test \
    --waitk 3 \
    #--force-finish \
