# Example Pipeline of Simultaneous Speech-to-Text Translation (SimulST) with LLMs

SimulST is a natural next step for LLM usage in simultaneous translation, and we provide an example pipeline with additional extensions here. This particular pipeline utilizes Whisper models for transcription before feeding the transcription to existing evaluation agents in Simul-LLM, using LLMs fine-tuned for NMT or SimulMT. 

For this example, no further fine-tuning is necessary (although tweaking to the NMT or SimulMT fine-tuning pipeline to mirror transcription characteristics, i.e. significant noise, might help), so we only provide information related to evaluation.

---

## Preprocessing

To be added soon, based on fairseq MuST-C preprocessing scripts...

---

## Evaluation

The following script is employed to evaluate a Falcon-based model fine-tuned for NMT and normally used for SimulMT evaluations.

```bash
export ROOT=<PATH_TO_ROOT>
export VENV_ROOT=<PATH_TO_VENV_ROOT>
export ADAPTER_PATH=${ROOT}/<PATH_TO_ADAPTER>
export OUTPUT=${ROOT}/<PATH_TO_OUTPUT_DIR>
export WAV_LIST=<PATH_TO_WAV_LIST>
export TARGET_LIST=<PATH_TO_TARGET_LIST>

source ${VENV_ROOT}/bin/activate
cd ${ROOT}

export HUGGINGFACE_HUB_CACHE="${ROOT}/.cache"
export HF_DATASETS_CACHE="${ROOT}/.cache/datasets"

export PYTHONPATH="${PYTHONPATH}:${ROOT}"

python examples/basic_speech_to_text/simuleval_wrapper.py \
    --agent-class examples.basic_speech_to_text.falcon_simulst_pipeline.WhisperToFalconAgentPipeline \
    --source ${WAV_LIST} \
    --target ${TARGET_LIST} \
    --model ybelkada/falcon-7b-sharded-bf16 \
    --adapter-path ${ADAPTER_PATH} \
    --output ${OUTPUT} \
    --translation-scheduler waitk --k 7 --source-lang en --target-lang es \
    --device cuda --compute-dtype float32 \
    --nmt-prompt \
    --model-size tiny --source-segment-size 500 \
    --force-finish \
```

On an extremely small, reduced test set from MuST-C for en-es, the following preliminary results were achieved (latency metrics are non-CA):

| BLEU | LAAL | AL | AP | DAL | ATD |
|------|------|----|----|-----|-----|
| 32.35 | 5734 | 5702 | 0.862 | 6273 | 4366 |
