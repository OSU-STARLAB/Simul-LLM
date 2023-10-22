
# Before using this, it requires that you call the bash command
# huggingface-cli login
# and log in to your account

# If there are jemalloc background thread creation errors, may need to run the command:
# export OPENBLAS_NUM_THREADS=4

from datasets import load_dataset

DATA_FILES = {
    "train": "/nfs/hpc/share/wildma/dataset_prep/datasets/2023-fall/en-es/wait3/must-c-en-es-01-train.json",
    "test": "/nfs/hpc/share/wildma/dataset_prep/datasets/2023-fall/en-es/wait3/must-c-en-es-01-test.json",
    "validation": "/nfs/hpc/share/wildma/dataset_prep/datasets/2023-fall/en-es/wait3/must-c-en-es-01-validation.json"
}

HUGGINGFACE_DATASET_NAME = 'must-c-en-es-wait3-02'


print(f'Attempting to load the dataset file...')
new_dataset = load_dataset("json", data_files=DATA_FILES)

print(f'Successfully read file')

print(f'Attemping to push to hub...')
new_dataset.push_to_hub(HUGGINGFACE_DATASET_NAME)

print(f'Script concluded')
