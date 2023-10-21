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

def main():
    llm, _ = model_parser().parse_known_args(sys.argv[1:])
    parser = argparse.ArgumentParser()

    if "falcon" in llm.model:
        print(f"Identified Falcon as the intended target LLM, attempting to create fine-tuning wrapper...", flush=True)
        from llmsimul.falcon.falcon_wrapper import FalconSFTTrainerWrapper
        FalconSFTTrainerWrapper.add_args(parser)
        trainer = FalconSFTTrainerWrapper(parser.parse_args())
        print(f"Successfully loaded Falcon fine-tuning wrapper. Attempting to begin fine-tuning now...", flush=True)

    else:
        print("=" * 75)
        print("Either a valid model was not provided or some infrastructure is missing.")
        print("=" * 75)
        raise NotImplementedError

    trainer.train()

def model_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        required=True,
        help="Path of model that you'd like to finetune, stored on the Hugginface Hub.",
    )
    return parser

if __name__ == "__main__":
    main()
