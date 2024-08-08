from simuleval.utils import entrypoint
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from argparse import Namespace, ArgumentParser
import random
from queue import Queue

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from transformers import FalconForCausalLM
from peft import LoraConfig, AutoPeftModelForCausalLM

from transformers.generation.stopping_criteria import StoppingCriteria
from llmsimul.falcon.falcon_stopping_criteria import StopTokenAndMaxLengthCriteria

from llmsimul.schedulers.waitk import WaitkScheduler

from llmsimul.utils_and_misc.beam_rescoring import ralcp_sort, rescoring_add_args

from llmsimul.basic_eval_agent import BasicLLMTextAgent

@entrypoint
class FalconTextAgent(BasicLLMTextAgent):

    source_type: str = "comp-text"
    target_type: str = "text"


    def __init__(self, args: Namespace):
        super().__init__(args)
       
        
    @staticmethod
    def add_args(parser: ArgumentParser):
        BasicLLMTextAgent.basic_add_args(parser)


    # sometimes has quirks that are LLM-specific, so we implement here
    def load_model_and_vocab(self, args):
        # load model, quantize on extremely memory constrained rigs
        if self.bnb:
            if self.device == "cuda":
                self.bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=args.compute_dtype,
                    bnb_4bit_use_double_quant=args.use_nested_quant,
                )
                self.model = FalconForCausalLM.from_pretrained(
                    args.model,
                    quantization_config=self.bnb_config,     
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                print("BitsAndBytes seemingly only supports deep quantization for GPUs. Try loading the model in a reduced precision format, such as float16.")
        
        else:
            print("Model device map currently set to 'auto,' fix incoming soon.")
            self.model = FalconForCausalLM.from_pretrained(
                args.model,
                device_map="auto",
                torch_dtype=self.compute_dtype,
                trust_remote_code=True,
            )

        # load tokenizer, could also enable local loading
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # load PEFT checkpoint
        if args.adapter_path is not None:
            self.model.load_adapter(args.adapter_path)
        else:
            print("No PEFT-LoRA adapter path was provided, this is not recommended for most hardware setups and indicates full model fine-tuning was done.")

        self.eoseq_ids = self.tokenizer("}", return_tensors="pt").input_ids.to(self.device)


    # leaving here in case users want to make any changes
    def policy(self):
        return super().policy()

    
    def buffer_commit(self, current_source, current_target):
        return super().buffer_commit(current_source, current_target)


    def make_inference_translation(self, source, current_translation, num_beams=1, num_chunks=1, window_size=10):
        """
        Call upon a specific model to do a translation request, the model input
        format is defined in this function
        """

        if current_translation is None:
            current_translation = ' '

        # unclear if falcon models really require specific "human" and "assistant" tokens/setup, seem to perform
        # fine regardless of these values
        if self.nmt_prompt and not self.nmt_augment:
            input_prompt = f'<h>: Translate from {self.source_lang} to {self.target_lang}: {{{source}}} <a>: {current_translation}'
        elif self.nmt_prompt and self.nmt_augment:
            input_prompt = f'<human>: Translate from {self.source_lang} to {self.target_lang}: {{{source}}} <assistant>: {current_translation} [SEP] '
        else:
            input_prompt = f'<human>: Given the {self.source_lang} sentence {{{source}}}, and the current translation in {self.target_lang} {{{current_translation}}}, what\'s the next translated word? <assistant>: '

        if self.rescorer == "ralcp":
            return_seq = num_beams
        else:
            return_seq = 1

        # 8 max tokens seems to be a pretty safe value to catch multi-token words
        # reducing this results in possibly losing tokens, custom stopping criteria
        # presents a nice alternative that should be efficient
        encoding = self.tokenizer(input_prompt, return_tensors="pt")
        stopping_criteria = StopTokenAndMaxLengthCriteria(
            start_length=encoding.input_ids.shape[-1],
            max_new_tokens=window_size,
            eoseq_ids=self.eoseq_ids,
        )

        # length penalty is positive, encourages long sequences
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids = encoding.input_ids.to(self.device),
                attention_mask = encoding.attention_mask,
                stopping_criteria=[stopping_criteria] if (self.decoding_strategy == "greedy" and not self.nmt_prompt) else None,
                max_new_tokens=window_size,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=num_beams,
                num_return_sequences=return_seq,
                use_cache=True,
                #length_penalty=1.0,
                #no_repeat_ngram_size=1,
            )

        top_output = self.tokenizer.decode(outputs[0])
        
        all_output = []
        for i in range(return_seq):
            all_output.append(self.tokenizer.decode(outputs[i]))
        #all_output = self.tokenizer.decode(outputs)
        #print(all_output)
        
        # Slice the returned array to remove the input that we fed the model
        # offset of 5 is to account for BOS in Falcon and Llama
        if not self.rescorer == "ralcp":
            return all_output[0][len(input_prompt):]
        else:
            ralcp_list = []
            for i in range(return_seq):
                ralcp_list.append(all_output[i][len(input_prompt):])
            return ralcp_list
   