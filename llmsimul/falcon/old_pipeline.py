'''
Old fine-tuning pipeline. Will be removed in a future commit.
'''

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from transformers import FalconForCausalLM
from peft import LoraConfig

from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

import argparse
from common_args import add_args

# quick argparse
parser = argparse.ArgumentParser()
add_args(parser)
args = parser.parse_args()


### Create and prepare model
model_name = 'ybelkada/falcon-7b-sharded-bf16'
training_set = 'maxolotl/falcon_w3_en_es_v2'

compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
if compute_dtype == torch.float16 and args.use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16, you can accelerate training with bf16")
        print("=" * 80)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=args.use_4bit,
    bnb_4bit_quant_type=args.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=args.use_nested_quant,
)

model = FalconForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
)

peft_config = LoraConfig(
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    r=args.lora_r,
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
dataset = load_dataset(training_set, split="train")

### training object config, looks like gradient checkpointing is currently broken
training_arguments = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.bsz,
    gradient_accumulation_steps=args.update_freq,
    optim=args.optim,
    save_steps=args.save_interval,
    logging_steps=args.log_interval,
    max_steps=args.max_updates,
    learning_rate=args.lr,
    max_grad_norm=args.max_grad_norm,
    warmup_ratio=args.warmup_ratio,
    lr_scheduler_type=args.lr_scheduler,
    fp16=True,
    group_by_length=True,
    #gradient_checkpointing=args.gradient_checkpointing,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

### normalization automatically cast to float16,
### we can restore precision at very little cost
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()
trainer.save_model("test-output")
trainer.mode.save_config("test-output/config.json")
