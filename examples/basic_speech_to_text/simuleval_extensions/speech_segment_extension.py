from simuleval.agents.agent import *
from dataclasses import dataclass

@dataclass
class CustomSpeechSegment(SpeechSegment):
    pass

SEGMENT_TYPE_DICT["custom-speech"] = CustomSpeechSegment
