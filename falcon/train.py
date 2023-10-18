import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from transformers import FalconForCausalLM
from peft import LoraConfig

from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

### Training configuration variables

# model_name = 'tiiuae/falcon-7b'
model_name = 'ybelkada/falcon-7b-sharded-bf16'
training_set = 'maxolotl/falcon_w3_en_es_v2'

local_rank = -1  # Used for multi-gpu

#per_device_train_batch_size = 4
#per_device_eval_batch_size = 1
#gradient_accumulation_steps = 4
#learning_rate = 2e-4
#max_grad_norm = 0.3
#weight_decay = 0.001
#max_seq_length = 1024
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

#use_4bit = False  # Activate 4bit precision base model loading, does NOT work in Google Colab
use_4bit = True  # Activate 4bit precision base model loading, does NOT work in Google Colab
use_nested_quant = False  # Activate nested quantization for 4bit base models
bnb_4bit_compute_dtype = 'float16'  # Compute dtype for 4bit base models
bnb_4bit_quant_type = 'nf4'  # Quantization type fp4 or nf4
num_train_epochs = 1  # The number of training epochs for the reward model
fp16 = False  # Enables fp16 training
bf16 = False  # Enables bf16 training
packing = False  # Use packing dataset creating
gradient_checkpointing = True  # Enables gradient checkpointing
optim = 'paged_adamw_32bit'  # The optimizer to use
lr_scheduler_type = 'constant'  # Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis
max_steps = 10000  # How many optimizer update steps to take
warmup_ratio = 0.03  # Fraction of steps to do a warmup for
group_by_length = True  # Group sequences into batches with same length. Saves memory and speeds up training considerably
save_steps = 10  # Save checkpoint every X updates steps
logging_steps = 10  # Log every X updates steps


### Create and prepare model

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16, you can accelerate training with bf16")
        print("=" * 80)

#device_map = {"": 0}
device_map = "auto"

#model = AutoModelForCausalLM.from_pretrained(
model = FalconForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map=device_map, trust_remote_code=True
)

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ],  # , "word_embeddings", "lm_head"],
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


### set up dataset and SFTTrainer, which should handle LoRA adapters
dataset = load_dataset(training_set, split="train[0:100000]")

### training object config
output_dir = "./model"
per_device_train_batch_size = 18
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
#save_steps = 10000
#logging_steps = 500
learning_rate = 2e-4
max_grad_norm = 0.3
#max_steps = 1000000
warmup_ratio = 0.03
lr_scheduler_type = "constant"

# temporary options
save_steps = 100
logging_steps = 10
max_steps = 1260

print(f"Beginning process of starting up training...")
print(tokenizer)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    max_steps=max_steps,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    fp16=True,
    group_by_length=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_arguments,
)

### normal floating point for normalization?
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()
trainer.save_model("test-output")
trainer.mode.save_config("test-output/config.json")
