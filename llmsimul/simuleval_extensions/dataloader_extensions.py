from simuleval.data.dataloader.dataloader import *
from simuleval.data.dataloader.t2t_dataloader import TextToTextDataloader

"""
Enables appropriate translation to t2t dataloader for SimulEval.
"""

@register_dataloader("comp-text-to-text")
class CompTextToTextDataloader(TextToTextDataloader):
    @classmethod
    def from_args(cls, args: Namespace):
        args.source_type = "comp-text"
        args.target_type = "text"
        return cls.from_files(args.source, args.target)

SUPPORTED_SOURCE_MEDIUM.append("comp-text")
