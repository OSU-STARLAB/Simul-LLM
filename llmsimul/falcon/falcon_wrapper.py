import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from transformers import FalconForCausalLM
from peft import LoraConfig

from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
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


    '''
    The direct SFTTrainer setup is relatively straightforward, but it's worth mentioning
    why the data collator is included. Without it, the entire input is assumed to be 
    what the model should attempt to predict. By providing the response template, the 
    collator will ensure that the model ignores all previous indices for the query during
    self-attention. This should result in some throughput increases.
    '''

    def setup_trainer(self, args):
        self.load_dataset()

        response_template = "<assistant>: "
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.training,
            eval_dataset=self.validation,
            peft_config=self.peft_config,
            max_seq_length=args.max_seq_length,
            tokenizer=self.tokenizer,
            data_collator=collator,
            formatting_func=self.formatting_func,
            args=self.training_arguments,
        )
   

    '''
    Formatting function takes care of prompt specification for a given LLM and allows the data
    collator to handle our data better. Example sentence at start of wait-3 translation:

       <human>: Given the English sentence {I'll tell you}, and the current translation in Spanish {},
       what's the next translated word? <assistant>: {Les}

    '''

    def formatting_func(self, example):
        output_texts = []
        for i in range(len(example['current_source'])):
            text = f"<human>: Given the English sentence {{{example['current_source'][i]}}} \
                    and the current translation in Spanish {{{example['current_target'][i]}}}, \
                    what's the next translated word? <assistant>: {{{example['target_token'][i]}}}"
            output_texts.append(text)
        return output_texts 


    def train(self):
        self.trainer.train()
        self.trainer.save_model("test-output")
