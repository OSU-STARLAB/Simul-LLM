import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from transformers import FalconForCausalLM
from peft import LoraConfig

from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

import argparse
from argparse import ArgumentParser, Namespace

from llmsimul.trainer_wrapper import LLMSimulSFTTrainerWrapper

'''
The below class serves as a wrapper for fine-tuning Falcon for simultaneous translation
via SFTTrainer. This extends from LLMSimulSFTTrainerWrapper and implements remaining 
unimplemented behavior from the parent wrapper.
'''

class FalconSFTTrainerWrapper(LLMSimulSFTTrainerWrapper):
    def __init__(self, args: Namespace):
        super().__init__(args)

    
    @classmethod
    def add_args(cls, parser: ArgumentParser):
        super().add_args(parser)


    def setup_model_and_tokenizer(self, args):
        self.model = FalconForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remove_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token


    def setup_trainer(self, args):
        self.load_dataset()

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            peft_config=self.peft_config,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
        )
    
    
    def formatting_func(self):
        assert NotImplementedError


    def train(self):
        self.trainer.train()
        self.trainer.save_model("test-output")
