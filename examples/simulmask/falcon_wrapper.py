from transformers import AutoTokenizer
from .falcon_simulmt_model import FalconForCausalLMSimulMT
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from argparse import ArgumentParser, Namespace
from llmsimul.trainer_wrapper import LLMSimulSFTTrainerWrapper
import os
from functools import partial

'''
The below class serves as a wrapper for fine-tuning Falcon for simultaneous translation
via SFTTrainer. This extends from LLMSimulSFTTrainerWrapper and implements remaining 
unimplemented behavior from the parent wrapper.
'''

class FalconSFTTrainerWrapper(LLMSimulSFTTrainerWrapper):
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.resume_from_checkpoint = args.resume_from_checkpoint
        
    
    @classmethod
    def add_args(cls, parser: ArgumentParser):
        super().add_args(parser)
        parser.add_argument("--waitk", type=int, default=1, help="Number of words to wait before generating a translation.")
        parser.add_argument("--attmask-type", type=str, default="simul", help="Type of attention mask to use for the model.  Options are 'simul', 'causal', or full.")
        parser.add_argument("--no-pos", action="store_true", help="Removes positional encodings from the model.",)
        parser.add_argument("--resume-from-checkpoint", action="store_true")
        parser.add_argument("--old-alibi", action="store_true", help="Use the old alibi method (not modified alibi).")


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
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remove_code=True,
        )
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        #English to German
        prompt1 = f"Translate the following sentence from {self.source_lang} to {self.target_lang}:"
        prompt2 = "\nAssistant:"
        self.model = FalconForCausalLMSimulMT.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config if self.bnb else None,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            tokenizer=self.tokenizer, 
            waitk=args.waitk, 
            attmask_type=args.attmask_type, 
            no_pos=args.no_pos,
            old_alibi=args.old_alibi,
            prompt1=prompt1, prompt2=prompt2
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
 
 
    '''
    The direct SFTTrainer setup is relatively straightforward, but it's worth mentioning
    why the data collator is included. Without it, the entire input is assumed to be 
    what the model should attempt to predict. By providing the response template, the 
    collator will ensure that the model ignores all previous indices for the query during
    self-attention. This should result in some throughput increases.
    '''

    def setup_trainer(self, args):
        self.load_dataset()

        response_template = "\nAssistant:"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)

        if self.adapter_path is not None:
            self.model.load_adapter(args.adapter_path)

        self.waitk = args.waitk

        formatting = partial(formatting_func, source_lang=self.source_lang, target_lang=self.target_lang, source=self.source, target=self.target) 

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

    def train(self):
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        self.trainer.save_model("test-output")
        
'''
Formatting function takes care of prompt specification for a given LLM and allows the data
collator to handle our data better. Designed for the IWSLT 2017 dataset (IWSLT/iwslt2017), but can be adapted.
'''

def formatting_func(example, source_lang, target_lang, source, target):
    output_texts = []
    for pair in example['translation']:
        text = f"Translate the following sentence from {source_lang} to {target_lang}: {pair[source]}\nAssistant: {pair[target]}<|endoftext|>"
        output_texts.append(text)
        
    return output_texts 
        