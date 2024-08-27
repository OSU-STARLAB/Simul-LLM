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

    if "falcon" in llm.model.lower():
        PartialState().print(f"Identified Falcon as the intended target LLM, attempting to create fine-tuning wrapper...", flush=True)
        from llmsimul.falcon.falcon_wrapper import FalconSFTTrainerWrapper
        FalconSFTTrainerWrapper.add_args(parser)
        trainer = FalconSFTTrainerWrapper(parser.parse_args())
        PartialState().print(f"Successfully loaded Falcon fine-tuning wrapper. Attempting to begin fine-tuning now...", flush=True)

    elif "llama" in llm.model.lower():
        PartialState().print(f"Identified LLaMA as the intended target LLM, attempting to create fine-tuning wrapper...", flush=True)
        from llmsimul.llama.llama_wrapper import LlamaSFTTrainerWrapper
        LlamaSFTTrainerWrapper.add_args(parser)
        trainer = LlamaSFTTrainerWrapper(parser.parse_args())
        PartialState().print(f"Successfully loaded LLaMA fine-tuning wrapper. Attempting to begin fine-tuning now...", flush=True)
        
    elif "mistral" in llm.model.lower():
        PartialState().print(f"Identified Mistral as the intended target LLM, attempting to create fine-tuning wrapper...", flush=True)
        from llmsimul.mistral.mistral_wrapper import MistralSFTTrainerWrapper
        MistralSFTTrainerWrapper.add_args(parser)
        trainer = MistralSFTTrainerWrapper(parser.parse_args())
        PartialState().print(f"Successfully loaded Mistral fine-tuning wrapper. Attempting to begin fine-tuning now...", flush=True)

    else:
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
    return parser

if __name__ == "__main__":
    main()
