import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from transformers import BloomForCausalLM
from peft import LoraConfig

from datasets import load_dataset
from datasets.fingerprint import Hasher
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments

from accelerate import Accelerator as accelerator, PartialState
from functools import partial

import argparse
from argparse import ArgumentParser, Namespace

from llmsimul.trainer_wrapper import LLMSimulSFTTrainerWrapper

'''
The below class serves as a wrapper for fine-tuning Bloom for simultaneous translation
via SFTTrainer. This extends from LLMSimulSFTTrainerWrapper and implements remaining 
unimplemented behavior from the parent wrapper.
'''

class BloomSFTTrainerWrapper(LLMSimulSFTTrainerWrapper):
    def __init__(self, args: Namespace):
        self.naive_training = args.naive_training
        super().__init__(args)

        PartialState().print(f"Identified source language as {self.source_lang}, target language as {self.target_lang}")
        PartialState().print(f"Prompt structure is NMT: {self.naive_training}")

    
    @classmethod
    def add_args(cls, parser: ArgumentParser):
        super().add_args(parser)
        parser.add_argument("--naive-training", action="store_true")


    def setup_peft_config(self, args):
        self.peft_config = LoraConfig(
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
    

    def setup_model_and_tokenizer(self, args):
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        self.model = BloomForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config if self.bnb else None,
            device_map=({'':PartialState().process_index} if self.fsdp else "auto"),
            trust_remote_code=True,
            torch_dtype=compute_dtype,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remove_code=True,
        )
        self.tokenizer.padding_side = 'right'
 

    def setup_trainer(self, args):
        self.load_dataset()

        response_template = "\nAssistant:"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
        
        # potentially enable new token for NMT prompts
        formatting = partial(formatting_func, source_lang=self.source_lang, target_lang=self.target_lang) 
        if self.naive_training:
            formatting = partial(formatting_func_nmt, source_lang=self.source_lang, target_lang=self.target_lang, source=self.source, target=self.target)


        if self.adapter_path is not None:
            self.model.load_adapter(args.adapter_path)


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
        if self.peft and self.fsdp:
            if getattr(self.trainer.accelerator.state, "fsdp_plugin", None):
                from peft.utils.other import fsdp_auto_wrap_policy

                fsdp_plugin = self.trainer.accelerator.state.fsdp_plugin
                fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(self.trainer.model)
  

    def train(self):
        self.trainer.train()
        
        # cover FSDP final state, collect it in case state_dict_type wasn't already FULL_STATE_DICT
        if self.trainer.is_fsdp_enabled:
            self.trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        self.trainer.save_model("test-output")
   

'''
NOTE: cannot easily be a class function when attempting to cache post-processed dataset, especially 
      problematic during FSDP, because if anything in the class changes the hash of these functions
      also appears to change

Formatting function takes care of prompt specification for a given LLM and allows the data
collator to handle our data better. Example sentence at start of wait-3 translation:

   <human>: Given the English sentence {I'll tell you}, and the current translation in Spanish {},
   what's the next translated word? <assistant>: {Les}

'''

def formatting_func(example, source_lang, target_lang):
    output_texts = []
    for i in range(len(example['current_source'])):
        text = f"Given the {source_lang} sentence {{{example['current_source'][i]}}} and the current translation in {target_lang} {{{example['current_target'][i]}}}, what's the next translated word? #\nAssistant: {{{example['target_token'][i]}}}"
        output_texts.append(text)
    return output_texts 


def formatting_func_nmt(example, source_lang, target_lang, source, target):
    output_texts = []
    for i in range(len(example[source])):
        text = f"Translate from {source_lang} to {target_lang}: {{{example[source][i]}}} #\nAssistant: {example[target][i]} <|endoftext|>"
        output_texts.append(text)
    return output_texts

