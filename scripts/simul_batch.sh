#!/bin/bash

export ROOT="<PATH_TO_REPO_ROOT>"
export VENV_ROOT="<PATH_TO_VIRTUAL_ENV>"

source ${VENV_ROOT}/bin/activate
cd ${ROOT}

export HUGGINGFACE_HUB_CACHE="${ROOT}/.cache"
export HF_DATASETS_CACHE="${ROOT}/.cache/datasets"

export PYTHONPATH="${PYTHONPATH}:${ROOT}"

# quantization disabled for inference by default, can be turned back on
# e.g.  --quantize-4bit

# some extensions are available in 'llmsimul/simuleval_extensions/' including
# a "computationally aware" LAAL for text and a framework for enabling 
# computationally aware options for other latency metrics, such metrics 
# cannot be 1:1 compared to their non-CA counterparts as their units differ
# (i.e. LAAL is in tokens/chunks compared to "CT_LAAL" which is in seconds of computation)
# e.g.  --computation-aware --latency-metrics CT_LAAL LAAL \

# alternatively, other translation quality metrics can also be used via a similar extension
# process set up in 'llmsimul/simuleval_extensions' (e.g. COMET)
# e.g.  --quality-metrics COMET --comet-source-text <PATH_TO_SOURCE_TEXT> \

# beam rescoring options are provided, as well, via the --rescorer argument
# by default this is set to 'none', but RALCP is supported out of the box:
# e.g.  --nmt-prompt \
#       --decoding-strategy multi_word_beam_search --num-beams 5 --num-chunks 3 --window-size 15 \
#       --rescorer ralcp --ralcp-thresh 0.6 \

python cli/simuleval_wrapper.py \
    --agent <PATH_TO_AGENT> \
    --source <PATH_TO_SOURCE_TEXT> \
    --target <PATH_TO_TARGET_TEXT> \
    --model <PATH_TO_MODEL> \
    --adapter-path <PATH_TO_PEFT_LORA_CHECKPOINT> \
    --output output_falcon_waitk_3_basic_test \
    --translation-scheduler waitk --k 3 --source-lang en --target-lang es \
    --device cuda --compute-dtype float32 \
    --force-finish \
