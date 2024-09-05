from simuleval.data.dataloader.dataloader import *
from simuleval.data.dataloader.t2t_dataloader import TextToTextDataloader
from simuleval.data.dataloader.s2t_dataloader import SpeechToTextDataloader

@register_dataloader("custom-speech-to-comp-text")
class CustomSpeechToCompTextDataloader(SpeechToTextDataloader):
    @classmethod
    def from_args(cls, args: Namespace):
        args.source_type = "custom-speech"
        args.target_type = "comp-text"
        return cls.from_files(args.source, args.target)

@register_dataloader("custom-speech-to-text")
class CustomSpeechToTextDataloader(SpeechToTextDataloader):
    @classmethod
    def from_args(cls, args: Namespace):
        args.source_type = "custom-speech"
        args.target_type = "text"
        return cls.from_files(args.source, args.target, args.tgt_lang)

SUPPORTED_SOURCE_MEDIUM.append("custom-speech")
