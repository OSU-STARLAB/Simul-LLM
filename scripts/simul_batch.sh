#!/bin/bash

export ROOT="<PATH_TO_REPO_ROOT>"
export VENV_ROOT="<PATH_TO_VIRTUAL_ENV>"

source ${VENV_ROOT}/bin/activate
cd ${ROOT}

export HUGGINGFACE_HUB_CACHE="${ROOT}/.cache"
export HF_DATASETS_CACHE="${ROOT}/.cache/datasets"

export PYTHONPATH="${PYTHONPATH}:${ROOT}"

cd SimulEval

# quantization disabled for inference by default, can be turned back on
# with --quantize-4bit

simuleval \
    --agent <PATH_TO_AGENT> \
    --source <PATH_TO_SOURCE_TEXT> \
    --target <PATH_TO_TARGET_TEXT> \
    --model <PATH_TO_MODEL> \
    --adapter-path <PATH_TO_PEFT_LORA_CHECKPOINT> \
    --output output_falcon_waitk_3_basic_test \
    --waitk 3 --source-lang en --target-lang es \
    --device cuda --compute-dtype float32 \
    --force-finish \
