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

@entrypoint
class FalconWaitkTextAgent(TextToTextAgent):

    source_type: str = "comp-text"
    target_type: str = "text"

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.translation_scheduler = args.translation_scheduler
        self.decoding_strategy = args.decoding_strategy
        self.device = args.device
        self.bnb = args.bnb
        self.num_beams = args.num_beams
        self.num_chunks = args.num_chunks
        self.window_size = args.window_size
        self.force_finish = args.force_finish
        self.maximum_length_delta = args.maximum_length_delta
        
        # experimental RALCP arguments, can be extended to other
        # rescoring options, a little unhappy with this implementation
        self.rescorer = args.rescorer
        self.ralcp_thresh = args.ralcp_thresh

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

        # only waitk implemented, but easy to replace with other options
        if self.translation_scheduler == "waitk":
            self.scheduler = WaitkScheduler(args)
        else:
            raise NotImplementedError

        self.load_model_and_vocab(args)
        
    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("--translation-scheduler", type=str, default="waitk")
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

        # grab all rescorer related arguments        
        rescoring_add_args(parser)

        # entrance for schedulers
        WaitkScheduler.add_args(parser)


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
    

    # determines READ/WRITE behavior
    def policy(self):
        lagging = len(self.states.source) - len(self.states.target)
        current_source = " ".join(self.states.source)
        current_target = " ".join(self.states.target)

        # dealing with a white space edge case
        current_target = " ".join(current_target.split())
                
        # useful for multi-word beam search schemes
        if not self.write_buffer.empty():
            return self.buffer_commit(current_source, current_target) 

        # currently only set up for waitk, but can be easily replaced
        decision = self.scheduler(len(self.states.source), len(self.states.target))

        if decision or self.states.source_finished:
            if self.nmt_prompt and self.decoding_strategy == "greedy":
                model_output = self.make_inference_translation(current_source, current_target)
                prediction = model_output.strip().strip("{").split(' ')[0]
            elif self.decoding_strategy == "greedy":
                model_output = self.make_inference_translation(current_source, current_target)
                prediction = model_output.strip().strip("{").strip("}").strip().split(' ')[0]
            elif self.decoding_strategy == "subword_beam_search":
                model_output = self.make_inference_translation(current_source, current_target, num_beams=self.num_beams)
                prediction = model_output.strip().strip("{").strip("}").strip().split(' ')[0]
            elif self.decoding_strategy == "multi_word_beam_search" and not ralcp:
                # small fix for no trailing beam behavior for SBS, 100 window size is arbitrary
                model_output = self.make_inference_translation(
                    current_source, 
                    current_target, 
                    num_beams=self.num_beams, 
                    num_chunks=self.num_chunks, 
                    window_size=self.window_size if not self.states.source_finished else 100,
                )
                model_output = model_output.strip().strip("{").strip("}").strip().split(' ')
                prediction = model_output[0]
                for i in range(1, min(self.num_chunks, len(model_output))):
                    self.write_buffer.put(model_output[i])
            elif self.decoding_strategy == "multi_word_beam_search" and self.rescorer != "none":
                # small fix for no trailing beam behavior for SBS, 100 window size is arbitrary
                model_output = self.make_inference_translation(
                        current_source,
                        current_target,
                        num_beams=self.num_beams,
                        num_chunks=self.num_chunks,
                        window_size=self.window_size if not self.states.source_finished else 100,
                )
                new_list = []
                for i in range(len(model_output)):
                    new_list.append(model_output[i].strip().strip("{").split(' '))
                
                # begin rescoring here, ralcp supported at the moment
                if self.rescorer == "ralcp":
                    predictions = ralcp_sort(new_list, self.ralcp_thresh)
                else:
                    raise NotImplementedError

                prediction = predictions[0]
                for i in range(1, min(self.num_chunks, len(predictions))):
                    self.write_buffer.put(predictions[i])
            else:
                raise NotImplementedError


            # handle some edge cases, don't want these in translation of course
            prediction_send = prediction
            
            # looks redundant, but covering an edge case issue
            if "}" in prediction:
                prediction_send = prediction_send.strip("}")

            if "endoftext" in prediction:
                prediction_send = ""
           
            finished = ("endoftext" in prediction or "}" in prediction or lagging < -self.maximum_length_delta) and (not self.force_finish or self.states.source_finished)

            # logging and account for extra read actions on force finish
            if finished:

                # covers buffer edge case
                while not self.write_buffer.empty():
                    self.write_buffer.get()

                print(f"Finished sentence: \n\tSource: '{current_source}'\n\t Target: '{current_target + ' ' + prediction_send}'", flush=True)
            elif prediction_send == "" and (self.force_finish and not self.states.source_finished):
                return ReadAction()


            # will need to modify finish condition at a later date
            return WriteAction(prediction_send, finished=finished)
        else:
            return ReadAction()


    """
    Required for multiple consecutive translation decisions during Speculative Beam Search.
    """
    def buffer_commit(self, current_source, current_target):
        # prediction management with write buffer
        prediction = self.write_buffer.get()
        prediction_send = prediction
        if "endoftext" in prediction:
            prediction_send = ""
        elif "}" in prediction:
            prediction_send = prediction.strip("}")
       
        finished = ("endoftext" in prediction or "}" in prediction or lagging < -self.maximum_length_delta) and (not self.force_finish or self.states.source_finished)

        # logging and account for extra read actions on force finish
        if finished:
            while not self.write_buffer.empty():
                self.write_buffer.get()
            print(f"Finished sentence: \n\tSource: '{current_source}'\n\t Target: '{current_target + ' ' + prediction}'", flush=True)
        elif prediction_send == "" and (self.force_finish and not self.states.source_finished):
            # flush the buffer
            while not self.write_buffer.empty():
                self.write_buffer.get()
            return ReadAction()

        # will need to modify finish condition at a later date
        return WriteAction(prediction_send, finished=finished)


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

        if self.ralcp:
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
        if not self.ralcp:
            return all_output[0][len(input_prompt):]
        else:
            ralcp_list = []
            for i in range(return_seq):
                ralcp_list.append(all_output[i][len(input_prompt):])
            return ralcp_list
   