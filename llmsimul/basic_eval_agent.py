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

"""
    A pretty basic version of what is expected to be provided on an agent-by-agent basis.
    Functions with NotImplementedError raises are, of course, expected to be implemented by
    the child function. 
"""

# no longer an entrypoint class for SimulEval
class BasicLLMTextAgent(TextToTextAgent):

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

        # assertion necessary but a little clunky, revision maybe be TODO
        assert self.rescorer == "none" or self.decoding_strategy == "multi_word_beam_search", \
            "Rescorer only supported for multi-word beam search (i.e. chunk-wise SBS and variants)."

        self.nmt_prompt = args.nmt_prompt
        self.nmt_augment = args.nmt_augment

        if self.decoding_strategy == "multi_word_beam_search" and self.nmt_prompt:
            print("Warning: SBS implementations work best out of the box with an NMT styled prompt, not a firmly bounded one" \
                   + "(i.e. {...} for a single word). May not work out of the box and errors/odd outputs should be expected.", flush=True)
        
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
    def basic_add_args(parser: ArgumentParser):
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
        raise NotImplementedError


    # determines READ/WRITE behavior, implemented here as behavior SHOULD be consistent
    def policy(self):
        lagging = len(self.states.source) - len(self.states.target)
        current_source = " ".join(self.states.source)
        current_target = " ".join(self.states.target)

        # dealing with a white space edge case
        current_target = " ".join(current_target.split())
                
        # useful for multi-word beam search schemes
        if not self.write_buffer.empty():
            return self.buffer_commit(current_source, current_target, lagging) 

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
            elif self.decoding_strategy == "multi_word_beam_search" and self.rescorer == "none":
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
    def buffer_commit(self, current_source, current_target, lagging):
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

    # not implemented here, as we assume there can be some LLM-specific quirks (see Llama issues)
    def make_inference_translation(self, source, current_translation, num_beams=1, num_chunks=1, window_size=10):
        raise NotImplementedError