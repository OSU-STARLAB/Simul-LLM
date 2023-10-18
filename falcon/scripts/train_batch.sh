source /nfs/hpc/share/agostinv/llama_simulst_venv/bin/activate

#pip --cache-dir .cache
export HUGGINGFACE_HUB_CACHE=".cache"
export HF_DATASETS_CACHE=".cache/datasets"

python train.py
