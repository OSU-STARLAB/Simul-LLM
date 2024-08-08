"""
    Models wait-k behavior. Requires only the current source context
    step and the target context step to function.

    NOTE: we do not directly return READ/WRITE decisions, as there is
          information we may not have access to here. We only return
          whether the criteria for the scheduler was met.
"""

from argparse import Namespace, ArgumentParser
from llmsimul.schedulers.translation_scheduler import TranslationScheduler

class WaitkScheduler(TranslationScheduler):
    def __init__(self, args: Namespace):
        super().__init__()
        self.k = args.k
    
    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("--k", type=int, default=3)
    
    def __call__(self, src_len: int, tgt_len: int):
        lagging = src_len - tgt_len
        return lagging >= self.k 