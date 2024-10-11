from simuleval.data.dataloader.dataloader import *
from simuleval.data.dataloader.t2t_dataloader import TextToTextDataloader
from simuleval.data.dataloader.s2t_dataloader import SpeechToTextDataloader

"""
    Enables appropriate translation to t2t dataloader and s2t dataloader
    for SimulEval when interfacing with LLM eval agents and their source
    and target types.
"""

@register_dataloader("comp-text-to-text")
class CompTextToTextDataloader(TextToTextDataloader):
    @classmethod
    def from_args(cls, args: Namespace):
        args.source_type = "comp-text"
        args.target_type = "text"
        return cls.from_files(args.source, args.target)

@register_dataloader("speech-to-comp-text")
class SpeechToCompTextDataloader(SpeechToTextDataloader):
    @classmethod
    def from_args(cls, args: Namespace):
        args.source_type = "speech"
        args.target_type = "comp-text"
        return cls.from_files(args.source, args.target)

SUPPORTED_SOURCE_MEDIUM.append("comp-text")
SUPPORTED_TARGET_MEDIUM.append("comp-text")
