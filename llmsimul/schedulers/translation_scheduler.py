"""
    Basic chassis for a translation scheduler. Mostly present to
    ease extensibility and provide some structure.

    The idea is to separate this functionality from the agent to 
    enable better drop-in behavior. Remains to be seen how simple
    it will be for adaptive strategies, in particular. 
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