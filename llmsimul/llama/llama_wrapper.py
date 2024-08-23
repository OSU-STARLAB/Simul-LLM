import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from transformers import LlamaForCausalLM
from peft import LoraConfig

from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments

from accelerate import PartialState

import argparse
from argparse import ArgumentParser, Namespace

from llmsimul.trainer_wrapper import LLMSimulSFTTrainerWrapper

'''
The below class serves as a wrapper for fine-tuning Llama for simultaneous translation
via SFTTrainer. This extends from LLMSimulSFTTrainerWrapper and implements remaining 
unimplemented behavior from the parent wrapper.
'''

class LlamaSFTTrainerWrapper(LLMSimulSFTTrainerWrapper):
    def __init__(self, args: Namespace):
        self.naive_training = args.naive_training
        self.nmt_augment = args.nmt_augment
        super().__init__(args)

        print(f"Identified source language as {self.source_lang}, target language as {self.target_lang}")
        print(f"Prompt structure is NMT: {self.naive_training}")
        print(f"If NMT prompt structure, prompt is augmented: {self.nmt_augment}")

    
    @classmethod
    def add_args(cls, parser: ArgumentParser):
        super().add_args(parser)
        parser.add_argument("--naive-training", action="store_true")
        parser.add_argument("--nmt-augment", action="store_true")


    def setup_peft_config(self, args):
        self.peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ) 


    def setup_model_and_tokenizer(self, args):
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config if self.bnb else None,
            device_map={'':PartialState().process_index},
            trust_remote_code=True,
            torch_dtype=compute_dtype,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remove_code=True,
        )
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.padding_side = 'right'
 

    def setup_trainer(self, args):
        self.load_dataset()

        response_template = "<a>:"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
        
        formatting = self.formatting_func
        if self.naive_training:
            formatting = self.formatting_func_nmt

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.training,
            eval_dataset=self.validation,
            peft_config=self.peft_config if self.peft else None,
            max_seq_length=args.max_seq_length,
            tokenizer=self.tokenizer,
            data_collator=collator,
            formatting_func=formatting,
            args=self.training_arguments,
        )

        # handle PEFT+FSDP case, ripped from a PEFT+QLoRA example
        if self.peft:
            if getattr(self.trainer.accelerator.state, "fsdp_plugin", None):
                from peft.utils.other import fsdp_auto_wrap_policy

                fsdp_plugin = self.trainer.accelerator.state.fsdp_plugin
                fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(self.trainer.model)
   

    '''
    Formatting function takes care of prompt specification for a given LLM and allows the data
    collator to handle our data better. Example sentence at start of wait-3 translation:

       <h>: Given the English sentence {I'll tell you}, and the current translation in Spanish {},
       what's the next translated word? <a>: {Les}

    '''

    def formatting_func(self, example):
        output_texts = []
        for i in range(len(example['current_source'])):
            text = f"<h>: Given the {self.source_lang} sentence {{{example['current_source'][i]}}} and the current translation in {self.target_lang} {{{example['current_target'][i]}}}, what's the next translated word? <a>: {{{example['target_token'][i]}}}"
            output_texts.append(text)
        return output_texts 

    
    def formatting_func_nmt(self, example):
        output_texts = []
        for i in range(len(example[self.source])):
            text = f"<h>: Translate from {self.source_lang} to {self.target_lang}: {{{example[self.source][i]}}} <a>: {example[self.target][i]} <\s>"
            output_texts.append(text)
        return output_texts


    def train(self):
        self.trainer.train()

        # cover FSDP final state, collect it in case state_dict_type wasn't already FULL_STATE_DICT
        if self.trainer.is_fsdp_enabled:
            self.trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        self.trainer.save_model("test-output")
