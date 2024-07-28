"""
    Basic chassis for a translation scheduler. Mostly present to
    ease extensibility and provide some structure.
"""

from argparse import Namespace, ArgumentParser

class TranslationScheduler:
    def __init__(self):
        return
    
    @staticmethod
    def add_args(parser: ArgumentParser):
        return

    def decision(self):
        raise NotImplementedError