from simuleval.utils import entrypoint
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from argparse import Namespace, ArgumentParser
import random
from queue import Queue

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from transformers import MistralForCausalLM
from peft import LoraConfig, AutoPeftModelForCausalLM

from transformers.generation.stopping_criteria import StoppingCriteria
from llmsimul.mistral.mistral_stopping_criteria import StopTokenAndMaxLengthCriteria

@entrypoint
class MistralWaitkTextAgent(TextToTextAgent):
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.waitk = args.waitk
        self.decoding_strategy = args.decoding_strategy
        self.device = args.device
        self.bnb = args.bnb
        self.num_beams = args.num_beams
        self.num_chunks = args.num_chunks
        self.window_size = args.window_size
        self.force_finish = args.force_finish
        self.maximum_length_delta = args.maximum_length_delta

        self.nmt_prompt = args.nmt_prompt
        self.nmt_augment = args.nmt_augment
        
        self.write_buffer = Queue()

        if args.source_lang== "en":
            self.source_lang = "English"
        elif args.source_lang== "es":
            self.source_lang = "Spanish"
        elif args.source_lang== "de":
            self.source_lang = "German"
        
        if args.target_lang== "en":
            self.target_lang = "English"
        elif args.target_lang== "es":
            self.target_lang = "Spanish"
        elif args.target_lang== "de":
            self.target_lang = "German"

        if args.compute_dtype == "float32":
            self.compute_dtype = torch.float32
        elif args.compute_dtype == "float16":
            self.compute_dtype = torch.float16
        elif args.compute_dtype == "bfloat16":
            self.compute_dtype = torch.bfloat16
        elif args.compute_dtype == "float8":
            self.compute_dtype = torch.float8

        print(f"Identified source language as {self.source_lang} and target language as {self.target_lang}. Computing in {self.compute_dtype} for inference.")

        self.load_model_and_vocab(args)
        
    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("--waitk", type=int, default=3)
        parser.add_argument("--source-lang", type=str, default="en")
        parser.add_argument("--target-lang", type=str, default="es")
        parser.add_argument("--num-beams", type=int, default=1)
        parser.add_argument("--num-chunks", type=int, default=1)
        parser.add_argument("--window-size", type=int, default=10)
        parser.add_argument("--decoding-strategy", type=str, default="greedy")
        parser.add_argument("--compute-dtype", type=str, default="float32")
        parser.add_argument("--bnb", action="store_true")
        parser.add_argument("--bnb-4bit-quant-type", type=str, default="nf4")
        parser.add_argument("--use-nested-quant", action="store_true")
        parser.add_argument("--force-finish", action="store_true")
        parser.add_argument("--maximum-length-delta", type=int, default=10)
        parser.add_argument("--model", type=str, required=True,
                                help="Path to your model checkpoint or HuggingFace Hub.")
        parser.add_argument("--adapter-path", type=str,
                                help="Path to your PEFT-LoRA checkpoint. Highly recommended.")
        parser.add_argument("--nmt-prompt", action="store_true",
                                help="Enables NMT prompt formatting for basic NMT fine-tuning.")
        parser.add_argument("--nmt-augment", action="store_true",
                                help="Enables NMT prompt augmented with SimulMT behavior.")


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
                self.model = MistralForCausalLM.from_pretrained(
                    args.model,
                    quantization_config=self.bnb_config,     
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                print("BitsAndBytes seemingly only supports deep quantization for GPUs. Try loading the model in a reduced precision format, such as float16.")
        
        else:
            print("Model device map currently set to 'auto,' fix incoming soon.")
            self.model = MistralForCausalLM.from_pretrained(
                args.model,
                device_map="auto",
                torch_dtype=self.compute_dtype,
                trust_remote_code=True,
            )

        # load PEFT checkpoint
        if args.adapter_path is not None:
            self.model.load_adapter(args.adapter_path)
        else:
            print("No PEFT-LoRA adapter path was provided, this is not recommended for most hardware setups and indicates full mode fine-tuning was done.")

        # load tokenizer, could also enable local loading
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        #if self.nmt_augment:
        #    self.tokenizer.add_special_tokens({'additional_special_tokens': ['<assistant>: ', '<human>: ', '[SEP] ']})
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<assistant>:', '<human>:']})
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.eoseq_ids = self.tokenizer("}", return_tensors="pt").input_ids.to(self.device)
    

    # determines READ/WRITE behavior
    def policy(self):
        lagging = len(self.states.source) - len(self.states.target)
        current_source = " ".join(self.states.source)
        current_target = " ".join(self.states.target)

        current_target = " ".join(current_target.split())
                
        # useful for multi-word beam search schemes
        if not self.write_buffer.empty():
            return self.buffer_commit(current_source, current_target) 

        if lagging >= self.waitk or self.states.source_finished:
            if self.nmt_prompt and self.decoding_strategy == "greedy":
                model_output = self.make_inference_translation(current_source, current_target)
                prediction = model_output.strip().strip("{").split(' ')[0]
            elif self.decoding_strategy == "greedy":
                model_output = self.make_inference_translation(current_source, current_target)
                prediction = self.find_string_between_symbols(model_output)
            elif self.decoding_strategy == "subword_beam_search":
                model_output = self.make_inference_translation(current_source, current_target, num_beams=self.num_beams)
                prediction = self.find_string_between_symbols(model_output)
            elif self.decoding_strategy == "multi_word_beam_search":
                # small fix for no trailing beam behavior for SBS, 100 window size is arbitrary
                model_output = self.make_inference_translation(
                    current_source, 
                    current_target, 
                    num_beams=self.num_beams, 
                    num_chunks=self.num_chunks, 
                    window_size=self.window_size if self.states.source_finished else 100,
                )
                model_output = model_output.strip().strip("{").split(' ')
                prediction = model_output[0]
                for i in range(1, min(self.num_chunks + 1, len(model_output))):
                    self.write_buffer.put(model_output[i])
            else:
                raise NotImplementedError


            # handle some edge cases, don't want these in translation of course
            prediction_send = prediction
            if "\s" in prediction or "}" in prediction:
                prediction_send = ""
           
            finished = ("\s" in prediction or "}" in prediction or lagging < -self.maximum_length_delta) and (not self.force_finish or self.states.source_finished)

            # logging and account for extra read actions on force finish
            if finished:
                print(f"Finished sentence: \n\tSource: '{current_source}'\n\t Target: '{current_target + ' ' + prediction}'", flush=True)
            elif prediction_send == "" and (self.force_finish and not self.states.source_finished):
                return ReadAction()


            # will need to modify finish condition at a later date
            return WriteAction(prediction_send, finished=finished)
        else:
            return ReadAction()


    ''' 
    Required for multiple consecutive translation decisions during Speculative Beam Search.
    '''
    def buffer_commit(self, current_source, current_target):
        # prediction management with write buffer
        prediction = self.write_buffer.get()
        prediction_send = prediction
        if "\s" in prediction:
            prediction_send = ""
        elif "}" in prediction:
            prediction_send = prediction.strip("}")
       
        finished = ("\s" in prediction or "}" in prediction or lagging < -self.maximum_length_delta) and (not self.force_finish or self.states.source_finished)

        # logging and account for extra read actions on force finish
        if finished:
            print(f"Finished sentence: \n\tSource: '{current_source}'\n\t Target: '{current_target + ' ' + prediction}'", flush=True)
        elif prediction_send == "" and (self.force_finish and not self.states.source_finished):
            # flush the buffer
            while not self.write_buffer.empty():
                self.write_buffer.get()
            return ReadAction()

        # will need to modify finish condition at a later date
        return WriteAction(prediction_send, finished=finished)


    ''' 
    The following functions are entirely the design of Max Wild and detail translation wrappers for Mistral 
    '''
    def make_inference_translation(self, source, current_translation, num_beams=1, num_chunks=1, window_size=10):
        """
        Call upon a specific model to do a translation request, the model input
        format is defined in this function
        """

        if current_translation is None:
            current_translation = ' '

        if self.nmt_prompt and not self.nmt_augment:
            input_prompt = f'<human>: Translate from {self.source_lang} to {self.target_lang}: {{{source}}} <assistant>: {{{current_translation}'
        elif self.nmt_prompt and self.nmt_augment:
            input_prompt = f'<human>: Translate from {self.source_lang} to {self.target_lang}: {{{source}}} <assistant>: {current_translation} [SEP] '
        else:
            input_prompt = f'<human>: Given the {self.source_lang} sentence {{{source}}}, and the current translation in {self.target_lang} {{{current_translation}}}, what\'s the next translated word? <assistant>: '

        print(input_prompt)

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
                num_return_sequences=1,
                use_cache=True,
                #length_penalty=1.0,
                #no_repeat_ngram_size=1,
            )

        all_output = self.tokenizer.decode(outputs[0])

        # Slice the returned array to remove the input that we fed the model
        return all_output[len(input_prompt):]


    # may not need to call this, 99% of the time the first and last token will be '{' and '}'
    def find_string_between_symbols(self, source, start_symbol=['{'], end_symbol=['}', '<']):
        """
        Returns the content between the first instance of the start symbol and
        end symbol, where the depth of the level of the symbols is maintained
            - so "{{in}in{in}in} {out}" returns "{in}in{in}in", not "{in"
                  ^            ^
                  First instance of start/end
        """

        content_inside = ''
        start_symbols_found = 0
        found_initial_start_symbol = False

        for i in range(len(source)):

            if source[i] in start_symbol:
                found_initial_start_symbol = True
                start_symbols_found += 1

            if found_initial_start_symbol:
                content_inside += source[i]

            if found_initial_start_symbol and source[i] in end_symbol:
                if source[i] == '<' and source[i-1] == '{':
                    return source[1:]

                start_symbols_found -= 1
                if start_symbols_found <= 0:
                    break

        return content_inside[1:-1]  # Ignore the initial and ending '{' and '}'
