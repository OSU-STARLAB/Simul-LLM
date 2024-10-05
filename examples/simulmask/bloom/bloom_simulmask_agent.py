from simuleval.utils import entrypoint
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from argparse import Namespace, ArgumentParser
import os
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from examples.simulmask.bloom.bloom_simulmt_model import BloomForCausalLMSimulMT

from examples.simulmask.simulmask_stopping_criteria import StopTokenCriteria
import random
import sys
sys.setrecursionlimit(1500)

@entrypoint
class BloomSimulMaskTextAgent(TextToTextAgent):
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.waitk = args.waitk
        self.recompute_size = args.recompute_size
        self.decoding_strategy = args.decoding_strategy
        self.device = args.device
        self.quantize_4bits = args.quantize_4bits
        self.max_lag = args.max_lag

        self.source = args.source_lang
        self.target = args.target_lang

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

        if args.compute_dtype == "float32":
            self.compute_dtype = torch.float32
        elif args.compute_dtype == "float16":
            self.compute_dtype = torch.float16
        elif args.compute_dtype == "bfloat16":
            self.compute_dtype = torch.bfloat16
        elif args.compute_dtype == "float8":
            self.compute_dtype = torch.float8
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        #self.tokenizer.pad_token = self.tokenizer.eos_token

        prompt1 = f"Translate the following sentence from {self.source_lang} to {self.target_lang}:"
        prompt2 = "#\nAssistant:"

        if self.quantize_4bits:
            if self.device == "cuda":
                self.bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=args.compute_dtype,
                    bnb_4bit_use_double_quant=args.use_nested_quant,
                )
                
                self.model = BloomForCausalLMSimulMT.from_pretrained(
                    args.model,
                    quantization_config=self.bnb_config,     
                    device_map=self.device,
                    trust_remote_code=True,
                    tokenizer=self.tokenizer, 
                    waitk=args.waitk, 
                    attmask_type=args.attmask_type,  
                    no_pos=args.no_pos,
                    prompt1=prompt1, prompt2=prompt2
                )
            else:
                print("BitsAndBytes seemingly only supports deep quantization for GPUs. Try loading the model in a reduced precision format, such as float16.")
        
        else:
            self.model = BloomForCausalLMSimulMT.from_pretrained(
                args.model,
                device_map=self.device,
                torch_dtype=self.compute_dtype,
                trust_remote_code=True,
                tokenizer=self.tokenizer, 
                waitk=args.waitk, 
                attmask_type=args.attmask_type, 
                no_pos=args.no_pos,
                prompt1=prompt1, prompt2=prompt2
            )
            
        self.model.resize_token_embeddings(len(self.tokenizer))
        if args.adapter_path is not None:
            self.model.load_adapter(args.adapter_path)

        self.eoseq_ids = self.tokenizer(['<|endoftext|>'], return_tensors="pt").input_ids.to(self.device)

        self.prompt_past_key_values = None
        self.pred_past_key_values = None
        self.prev_source_count = 0
        torch.manual_seed(0)
        random.seed(0)
        
    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("--waitk", type=int, default=3, help="Number of words to wait before generating a translation.")
        parser.add_argument("--decoding-strategy", type=str, default="greedy")
        parser.add_argument("--compute-dtype", type=str, default="float32")
        parser.add_argument("--quantize-4bits", action="store_true")
        parser.add_argument("--bnb-4bit-quant-type", type=str, default="nf4")
        parser.add_argument("--use-nested-quant", action="store_true")
        parser.add_argument("--adapter-path", type=str, default=None,
                                help="Path to your PEFT checkpoint.")
        parser.add_argument("--model", type=str, required=True,
                        help="Model name.")
        parser.add_argument("--no-pos", action="store_true", help="Removes positional encodings from the model.")
        parser.add_argument("--attmask-type", type=str, default="causal", help="Type of attention mask to use for the model.  Options are 'simul', 'causal', or full.")
        parser.add_argument("--source-lang", type=str, default="en")
        parser.add_argument("--target-lang", type=str, default="de")
        parser.add_argument("--max-lag", type=int, default=10)
        parser.add_argument("--recompute-size", type=int, default=0, help="Number of tokens to recompute at each prediction step.")

    """
    Enforces the waitk policy for the agent.
    """
    def policy(self):
        lagging = len(self.states.source) - len(self.states.target)
        #print(len(self.states.source), len(self.states.target))
        if lagging >= self.waitk or self.states.source_finished:
            current_source = " ".join(self.states.source)
            current_target = " ".join(self.states.target)

            if self.decoding_strategy == "greedy":
                complete_prompt = f"Translate the following sentence from {self.source_lang} to {self.target_lang}: {current_source}#\nAssistant:"

                if self.prev_source_count < len(self.states.source):
                    partial_prompt = f"Translate the following sentence from {self.source_lang} to {self.target_lang}: {current_source}"
                    self.process_prompt(partial_prompt, max_new_tokens=1)
                    if self.pred_past_key_values is not None and self.recompute_size > 0:
                        self.recompute_pred(partial_prompt, complete_prompt, current_target, max_new_tokens=1)
                    self.prev_source_count = len(self.states.source)
                prediction = self.make_inference_translation(complete_prompt, current_target, max_new_tokens=50)
            else:
                raise NotImplementedError

            prediction_send = prediction
            if 'endoftext' in prediction or lagging < -self.max_lag:
                prediction_send = ""
                self.prompt_past_key_values = None
                self.pred_past_key_values = None
                self.prev_source_count = 0

           
            # will need to modify finish condition at a later date
            return WriteAction(prediction_send, finished=('endoftext' in prediction or lagging < -self.max_lag))
        else:
            return ReadAction()

    '''
    Recomputes the representations for the previously predicted hidden states.  Prevents positional confusion for models fine-tuned without
    Simulmask and demonstrates performance improvements provided by adjusting recompute size.
    '''
    def recompute_pred(self, partial_prompt, complete_prompt, current_target, max_new_tokens):
        partial_encoding = self.tokenizer(partial_prompt, return_tensors="pt").input_ids
        complete_encoding = self.tokenizer(complete_prompt, return_tensors="pt").input_ids
        if len(current_target) > 0:
            target_encoding = self.tokenizer(" " + current_target, return_tensors="pt").input_ids
        else:
            target_encoding = self.tokenizer(current_target, return_tensors="pt").input_ids
        encoding = torch.cat((complete_encoding, target_encoding), dim=1).to(device=self.device, dtype=complete_encoding.dtype)

        encoding = encoding[:, :partial_encoding.size(1)+min(self.recompute_size, self.pred_past_key_values[0][0].size(2))]

        model_kwargs = {'past_key_values': self.prompt_past_key_values}

        stopping_criteria = StopTokenCriteria(
            max_new_tokens=max_new_tokens,
            tokenizer=self.tokenizer
        )

        with torch.inference_mode():
            outputs = self.model.generate_out(
                input_ids = encoding.to(self.device),
                stopping_criteria=[stopping_criteria],
                **model_kwargs
            )

            past_key_values = outputs['past_key_values']

            recompute_pred_past_key_values, _ = self.separate_pred_prompt(past_key_values)
            self.pred_past_key_values = self.swap_preds(recompute_pred_past_key_values)
    

    """
    Generates the unprocessed hidden states associated with the LLM prompt.  The incremental source stream is considered a part of the prompt.
    """

    def process_prompt(self, input_prompt, max_new_tokens):
        # 8 max tokens seems to be a pretty safe value to catch multi-token words
        # reducing this results in possibly losing tokens, custom stopping criteria
        # presents a nice alternative that should be efficient
        encoding = self.tokenizer(input_prompt, return_tensors="pt").input_ids
        stopping_criteria = StopTokenCriteria(
            max_new_tokens=max_new_tokens,
            tokenizer=self.tokenizer
        )

        model_kwargs = {'past_key_values': self.prompt_past_key_values}

        if self.prompt_past_key_values is not None:
            assert encoding.size(1) > self.prompt_past_key_values[0][0].size(2)

        with torch.inference_mode():
            outputs = self.model.generate_out(
                input_ids = encoding.to(self.device),
                stopping_criteria=[stopping_criteria],
                **model_kwargs
            )            

            self.prompt_past_key_values = outputs['past_key_values']

        assert encoding.size(1) == self.prompt_past_key_values[0][0].size(2)
            
    """
    Creates a single word prediction.
    """

    def make_inference_translation(self, input_prompt, current_target, max_new_tokens):
        """
        Call upon a specific model to do a translation request, the model input
        format is defined in this function
        """
        encoding_1 = self.tokenizer(input_prompt, return_tensors="pt").input_ids
        if len(current_target) > 0:
            encoding_2 = self.tokenizer(" " + current_target, return_tensors="pt").input_ids
        else:
            encoding_2 = self.tokenizer(current_target, return_tensors="pt").input_ids
        encoding = torch.cat((encoding_1, encoding_2), dim=1).to(device=self.device, dtype=encoding_1.dtype)
        #encoding = self.tokenizer(input_prompt, return_tensors="pt").input_ids.to(self.device)

        if self.pred_past_key_values is None:
            model_kwargs = {'past_key_values': self.prompt_past_key_values}
        else:
            model_kwargs = {'past_key_values': self.merge_pred_prompt()}
        
        stopping_criteria = StopTokenCriteria(
            max_new_tokens=max_new_tokens,
            tokenizer=self.tokenizer
        )
        #print(f"Prompt: '{input_prompt}'", flush=True)
        #assert encoding.size(1) > model_kwargs["past_key_values"][0][0].size(2)
        with torch.inference_mode():
            outputs = self.model.generate_out(
                input_ids = encoding,
                stopping_criteria=[stopping_criteria],
                **model_kwargs
            )

            past_key_values = outputs['past_key_values']
            output_list = self.tokenizer.decode(outputs['prediction']).split()
            #assert encoding.size(1) + outputs['prediction'].size(0) - 1 == past_key_values[0][0].size(2)

            if output_list == []: 
                output = ""
            else: 
                output = output_list[0]
            

        self.pred_past_key_values, self.prompt_past_key_values = self.separate_pred_prompt(past_key_values)

        assert self.pred_past_key_values[0][0].size(2) + self.prompt_past_key_values[0][0].size(2) == past_key_values[0][0].size(2)
        
        # Slice the returned array to remove the input that we fed the model
        return output

    """
    The helper functions for processing the inputs and outputs from the LLM.
    """

    def swap_preds(self, recompute_pred):
        pred_keys_values = []
        for pred, recompute in zip(self.pred_past_key_values, recompute_pred):
            key = torch.cat((recompute[0], pred[0][:, :, recompute[0].size(2):]), dim=2)
            val = torch.cat((recompute[1], pred[1][:, :, recompute[1].size(2):]), dim=2)
            pred_keys_values.append((key, val))
        assert pred_keys_values[0][0].size() == self.pred_past_key_values[0][0].size()
        return tuple(pred_keys_values)

    def merge_pred_prompt(self):
        merge_keys_values = []
        for pred, prompt in zip(self.pred_past_key_values, self.prompt_past_key_values):
            merge_keys_values.append((torch.cat((prompt[0], pred[0]), dim=2), torch.cat((prompt[1], pred[1]), dim=2)))
        return tuple(merge_keys_values)

    def separate_pred_prompt(self, past_key_values):
        pred_keys_values = []
        prompt_keys_values = []
        prompt_size = self.prompt_past_key_values[0][0].size()[2]
        pred_size = past_key_values[0][0].size()[2] - prompt_size 
        for key, val in past_key_values:
            pred_keys_values.append((key[:, :, prompt_size:prompt_size+pred_size, :], val[:, :, prompt_size:prompt_size+pred_size, :]))
            prompt_keys_values.append((key[:, :, :prompt_size, :], val[:, :, :prompt_size, :]))
        assert tuple(pred_keys_values)[0][0].size(2) + tuple(prompt_keys_values)[0][0].size(2) == past_key_values[0][0].size(2)
        return tuple(pred_keys_values), tuple(prompt_keys_values)