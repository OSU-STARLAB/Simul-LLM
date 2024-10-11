from llmsimul.falcon.falcon_simulmt_agent import FalconTextAgent
from argparse import ArgumentParser, Namespace

class DummyFalconTextAgent(FalconTextAgent):
    
    def __init__(self, args: Namespace):
        super().__init__(args)

    @staticmethod
    def add_args(parser: ArgumentParser):
        FalconTextAgent.add_args(parser)
