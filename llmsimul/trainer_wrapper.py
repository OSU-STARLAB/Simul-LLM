import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from transformers import FalconForCausalLM
from peft import LoraConfig

from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments

from accelerate import Accelerator as accelerator, PartialState

import argparse
from argparse import ArgumentParser, Namespace
import os

'''
The below class serves as a general wrapper to SFTTrainer for training simultaneous
large language models. Some expected behavior is outlined in functions that remain
to be implemented. If additional setup is required for certain config objects, 
relevant setup functions should be overwritten.

When extending this class to other LLMs, it is expected that the remaining functions
are implemented and that self.train() details any additional expected behavior.
'''

class LLMSimulSFTTrainerWrapper:
    def __init__(self, args: Namespace):
        self.model_name = args.model
        self.training_set = args.training_set
        self.training_subset = args.training_subset
        self.source = args.source_lang
        self.target = args.target_lang
        self.peft = args.peft
        self.bnb = args.bnb
        self.adapter_path = args.adapter_path

        if self.source == "en":
            self.source_lang = "English"
        elif self.source == "de":
            self.source_lang = "German"
        elif self.source == "fr":
            self.source_lang = "French"
        elif self.source == "nl":
            self.source_lang = "Dutch"
        elif self.source == "ro":
            self.source_lang = "Romanian"
        elif self.source == "it":
            self.source_lang = "Italian"

        if self.target == "en":
            self.target_lang = "English"
        elif self.target == "de":
            self.target_lang = "German"
        elif self.target == "fr":
            self.target_lang = "French"
        elif self.target == "nl":
            self.target_lang = "Dutch"
        elif self.target == "ro":
            self.target_lang = "Romanian"
        elif self.target == "it":
            self.target_lang = "Italian"

        assert self.source != self.target, "Source and target languages should not be the same."
        
        if not self.peft:
            PartialState().print(f"Warning: PEFT-LoRA is set to {self.peft}, indicating full model fine-tuning is desirable. This is not recommended for most hardware setups.")
        
        self.fsdp = False
        if os.getenv("ACCELERATE_USE_FSDP"):
            self.fsdp = True

        self.setup_bnb_config(args)
        self.setup_peft_config(args)
        self.setup_training_arguments(args)
        self.setup_model_and_tokenizer(args)
        self.setup_trainer(args)


    '''
    The below arguments should be common to fine-tuning LLMs outside of Falcon.
    Defaults are configured to allow for fine-tuning on a single V100. Assuming
    a sharded model, however, batch size can be scaled significantly.

        Lora adapter-related arguments govern the size of the added network via PEFT.

        Quantization framework-related arguments govern exact model size and parameter
        drift from model compression.

        Direct training arguments govern fine-tuning behavior in general. For Falcon,
        gradient checkpointing is currently non-functional with the assumed sharded 7B
        model hardcoded in the training pipeline.
    '''
    
    @classmethod
    def add_args(cls, parser: ArgumentParser):
        # basic arguments required for fine-tuning
        parser.add_argument("--model", type=str, required=True, 
            help="Path to model you want to use, currently only works with models on Huggingface Hub.",
        )
        parser.add_argument("--training-set", type=str, required=True, 
            help="Path to dataset you want to use, currently only works with datasets on Huggingface Hub.",
        )
        
        # lora adapter arguments
        parser.add_argument("--peft", action="store_true")
        parser.add_argument("--adapter-path", type=str, 
            help="Path to locally stored adapter checkpoint that you want loaded.",
        )
        parser.add_argument("--lora-alpha", type=int, default=16)
        parser.add_argument("--lora-dropout", type=float, default=0.1)
        parser.add_argument("--lora-r", type=int, default=64)

        # quantization framework arguments
        parser.add_argument("--bnb", action="store_true")
        parser.add_argument("--use-4bit", action="store_true")
        parser.add_argument("--use-nested-quant", action="store_true")
        parser.add_argument("--bnb-4bit-compute-dtype", type=str, default="float16")
        parser.add_argument("--bnb-4bit-quant-type", type=str, default="nf4")

        # more direct training arguments
        parser.add_argument("--output-dir", type=str, default="./model")
        parser.add_argument("--bsz", type=int, default=4)
        parser.add_argument("--update-freq", type=int, default=1)
        parser.add_argument("--gradient-checkpointing", action="store_true")
        parser.add_argument("--optim", type=str, default="paged_adamw_32bit")
        parser.add_argument("--lr", type=float, default=2e-4)
        parser.add_argument("--lr-scheduler", type=str, default="constant")
        parser.add_argument("--max-grad-norm", type=float, default=0.3)
        parser.add_argument("--warmup-ratio", type=float, default=0.03)
        parser.add_argument("--save-interval", type=int, default=1000)
        parser.add_argument("--log-interval", type=int, default=100)
        parser.add_argument("--eval-interval", type=int, default=500)
        parser.add_argument("--max-updates", type=int, default=-1)
        parser.add_argument("--max-seq-length", type=int, default=1024)

        parser.add_argument("--source-lang", type=str, default="en")
        parser.add_argument("--target-lang", type=str, default="es")
        parser.add_argument("--save-strategy", type=str, default="epoch")
        parser.add_argument("--num-train-epochs", type=int, default=1)
        parser.add_argument("--training-subset", type=str, required=False, default="",
            help="Path to dataset subset you want to use, currently only works with datasets on Huggingface Hub.",
        )
        parser.add_argument(
            "--user-dir",
            type=str,
            default=None,
            required=False,
            help="The directory to personal project files.",
        )


    def setup_bnb_config(self, args):
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        if compute_dtype == torch.float16 and args.use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                PartialState().print("=" * 80)
                PartialState().print("Your GPU supports bfloat16, you can accelerate training with bf16")
                PartialState().print("=" * 80)

        # necessary to align compute_dtype with storage val for FSDP to work
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_storage=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
        )



    '''
    Annoying to change evaluation strategy, but if an epoch-based one is desired, 
    this function should be overridden in the child wrapper.

    Gradient checkpointing is bugged for Falcon-based models. This function should
    be overridden if you want to enable it.
    '''
    def setup_training_arguments(self, args):
        self.training_arguments = TrainingArguments(
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
            evaluation_strategy="steps",
            eval_steps=args.eval_interval,
            group_by_length=True,
            fp16=(args.bnb_4bit_compute_dtype == "float16"),
            bf16=(args.bnb_4bit_compute_dtype == "bfloat16"),
            #gradient_checkpointing=args.gradient_checkpointing,
        )
        PartialState().print(f'fp16: {(args.bnb_4bit_compute_dtype == "float16")}, bf16: {(args.bnb_4bit_compute_dtype == "bfloat16")}')


    def load_dataset(self):
        self.training = load_dataset(self.training_set, self.training_subset, split="train")
        self.validation = load_dataset(self.training_set, self.training_subset, split="validation")


    def setup_peft_config(self, args):
        raise NotImplementedError


    def setup_model_and_tokenizer(self, args):
        raise NotImplementedError

    
    def setup_trainer(self, args):
        raise NotImplementedError
    
    
    def formatting_func(self, example):
        raise NotImplementedError


    def train(self):
        raise NotImplementedError
