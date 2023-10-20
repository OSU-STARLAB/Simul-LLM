import argparse
from argparse import ArgumentParser

'''
The below arguments should be common to fine-tuning LLMs outside of Falcon.
Defaults are configured to allow for fine-tuning on a single V100. Assuming
a sharded model, however, batch size can be scaled significantly.

    Lora adapter-related arguments govern the size of the added network via PEFT.

    Quantization framework-related arguments govern exact model size and parameter
    drift from model compression.

    Direct training arguments govern fine-tuning behavior in general. For Falcon,
    gradient checkpointing is currently non-functional with the assumed sharded 7B
    model hardcoded in the training pipeline.
'''

def add_args(parser: ArgumentParser):

    # lora adapter arguments
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=64)

    # quantization framework arguments
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--use-nested-quant", action="store_true")
    parser.add_argument("--bnb-4bit-compute-dtype", type=str, default="float16")
    parser.add_argument("--bnb-4bit-quant-type", type=str, default="nf4")

    # more direct training arguments
    parser.add_argument("--output-dir", type=str, default="./model")
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--update-freq", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-scheduler", type=str, default="constant")
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--max-updates", type=int, default=100000)
    parser.add_argument("--max-seq-length", type=int, default=1024)
