from simuleval.utils import entrypoint
from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import ReadAction, WriteAction
from argparse import Namespace, ArgumentParser
import random

import os

import torch
from fairseq import checkpoint_utils, tasks

'''
Agent intended for use with Transformer models from fairseq, trained
for NMT with wait-k masking augmentations to attention mechanisms.
'''

@entrypoint
class ClassicWaitkTextAgent(TextToTextAgent):
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.waitk = args.waitk
        self.decoding_strategy = args.decoding_strategy
        self.device = args.device
        self.force_finish = args.force_finish

        if args.compute_dtype == "float32":
            self.compute_dtype = torch.float32
        elif args.compute_dtype == "float16":
            self.compute_dtype = torch.float16
        elif args.compute_dtype == "bfloat16":
            self.compute_dtype = torch.bfloat16
        elif args.compute_dtype == "float8":
            self.compute_dtype = torch.float8

        self.load_model_vocab(args) 

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("--waitk", type=int, default=3)
        parser.add_argument("--data-bin", type=str, required=True)
        parser.add_argument("--decoding-strategy", type=str, default="greedy")
        parser.add_argument("--compute-dtype", type=str, default="float32")
        parser.add_argument("--force-finish", action="store_true")
        parser.add_argument("--model-path", type=str, required=True,
                                help="Path to your basic t2t fairseq translation model.")

    def load_model_vocab(self, args):

        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError("Model file not found: {}".format(filename))

        state = checkpoint_utils.load_checkpoint_to_cpu(filename)

        task_args = state["cfg"]["task"]
        task_args.data = args.data_bin

        task = tasks.setup_task(task_args)

        # build model for ensemble
        state["cfg"]["model"].load_pretrained_encoder_from = None
        state["cfg"]["model"].load_pretrained_decoder_from = None

        self.model = task.build_model(state["cfg"]["model"])
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        self.model.share_memory()

        self.model.to(self.compute_dtype).to(self.device)

        # Set dictionary
        self.vocab = {}
        self.vocab["tgt"] = task.target_dictionary
        self.vocab["src"] = task.source_dictionary

    
    def reset(self):
        super().reset()
        self.states.encoder_states = {}
        self.states.incremental_states = {}

    def update_model_encoder(self):
        if len(self.states.source) == 0:
            return

        src_indices = [
            self.vocab['src'].index(x)
            for x in self.states.source
        ]

        if self.states.source_finished:
            # Append the eos index when the prediction is over
            src_indices += [self.vocab["tgt"].eos_index]

        src_indices = torch.LongTensor(src_indices).unsqueeze(0).to(self.device)

        src_lengths = torch.LongTensor([src_indices.size(1)]).to(self.compute_dtype).to(self.device)


        self.states.encoder_states = self.model.encoder(src_indices, src_lengths)

        torch.cuda.empty_cache()


    def policy(self):
        lagging = len(self.states.source) - len(self.states.target)
        
        if lagging >= self.waitk or self.states.source_finished:
            
            # begin by updating model encoder, could be rendered a bit more efficient
            # as we really only want to update when receiving new information, but
            # fine for now
            self.update_model_encoder()   
            
            # Current steps
            self.states.incremental_states["steps"] = {
                "src": self.states.encoder_states["encoder_out"][0].size(0),
                "tgt": 1 + len(self.states.target),
            }
            
            # encode previous predicted target tokens
            tgt_indices = torch.LongTensor(
                [self.model.decoder.dictionary.eos()]
                + [
                    self.vocab['tgt'].index(x)
                    for x in self.states.target
                    if x is not None
                ]
            ).unsqueeze(0).to(self.device)

            x, outputs = self.model.decoder.forward(
                prev_output_tokens=tgt_indices,
                encoder_out=self.states.encoder_states,
                incremental_state=self.states.incremental_states,
            )
            token = self.predict(x)

            torch.cuda.empty_cache()
            
            # don't allow for an early finish erroneously, depending on config
            if token == self.vocab['tgt'].eos_word and (self.force_finish and not self.states.source_finished):
                return ReadAction()

            # finishing generation condition
            finished = (token == self.vocab['tgt'].eos_word or lagging < -10) and (not self.force_finish or self.states.source_finished)
            
            # don't send token when finished
            if finished:
                token = ''

            # will need to modify finish condition at a later date
            return WriteAction(token, finished=finished)
        else:
            return ReadAction()


    def predict(self, decoder_states):
        # Predict target token from decoder states
        lprobs = self.model.get_normalized_probs(
            [decoder_states[:, -1:]], log_probs=True
        )

        index = lprobs.argmax(dim=-1)[0, 0].item()

        if index != self.vocab['tgt'].eos_index:
            token = self.vocab['tgt'].string([index])
        else:
            token = self.vocab['tgt'].eos_word

        return token
