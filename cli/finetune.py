"""
Command-line interface for simultaneous LLM fine-tuning.
Just setups up arguments for a given LLM SFTTrainer wrapper,
initializes the wrapper, and then begins fine-tuning.

Adding new models should be as simple as creating a new wrapper
with expected behavior defined and then copying the structure used
here for falcon.
"""

import argparse
import os
import sys
from accelerate import Accelerator as accelerator, PartialState

def main():
    llm, _ = model_parser().parse_known_args(sys.argv[1:])
    parser = argparse.ArgumentParser()

    if llm.user_dir == "examples/simulmask":
        from examples.simulmask.simulmask_wrapper import SimulMaskSFTTrainerWrapper
        SimulMaskSFTTrainerWrapper.add_args(parser)
        trainer = SimulMaskSFTTrainerWrapper(parser.parse_args())
        PartialState().print(f"Successfully loaded SimulMask fine-tuning wrapper. Attempting to begin fine-tuning now...", flush=True)
     
    if "falcon" in llm.model.lower():
        PartialState().print(f"Identified Falcon as the intended target LLM, attempting to create fine-tuning wrapper...", flush=True)
        if llm.user_dir is None:
            from llmsimul.falcon.falcon_wrapper import FalconSFTTrainerWrapper
            FalconSFTTrainerWrapper.add_args(parser)
            trainer = FalconSFTTrainerWrapper(parser.parse_args())
            PartialState().print(f"Successfully loaded Falcon fine-tuning wrapper. Attempting to begin fine-tuning now...", flush=True)

    elif "llama" in llm.model.lower():
        PartialState().print(f"Identified Llama3 as the intended target LLM, attempting to create fine-tuning wrapper...", flush=True)

        if "llama2" in llm.model.lower():
            PartialState().print(f"WARNING: Intended for use with Llama3 family, detected Llama2.", flush=True)

        from llmsimul.llama3.llama3_wrapper import LlamaSFTTrainerWrapper
        LlamaSFTTrainerWrapper.add_args(parser)
        trainer = LlamaSFTTrainerWrapper(parser.parse_args())
        PartialState().print(f"Successfully loaded Llama3 fine-tuning wrapper. Attempting to begin fine-tuning now...", flush=True)
        
    elif "mistral" in llm.model.lower():
        PartialState().print(f"Identified Mistral as the intended target LLM, attempting to create fine-tuning wrapper...", flush=True)
        from llmsimul.mistral.mistral_wrapper import MistralSFTTrainerWrapper
        MistralSFTTrainerWrapper.add_args(parser)
        trainer = MistralSFTTrainerWrapper(parser.parse_args())
        PartialState().print(f"Successfully loaded Mistral fine-tuning wrapper. Attempting to begin fine-tuning now...", flush=True)

    elif llm.user_dir is None:
        PartialState().print("=" * 75)
        PartialState().print("Either a valid model was not provided or some infrastructure is missing.")
        PartialState().print("=" * 75)
        raise NotImplementedError

    trainer.train()

def model_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        required=True,
        help="Path of model that you'd like to finetune, stored on the Huggingface Hub.",
    )
    parser.add_argument(
        "--user-dir",
        type=str,
        default=None,
        required=False,
        help="The directory to personal project files.",
    )
    return parser

if __name__ == "__main__":
    main()
