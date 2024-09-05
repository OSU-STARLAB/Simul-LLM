from simuleval.agents.agent import *
from dataclasses import dataclass

@dataclass
class CompTextSegment(TextSegment):
    pass

SEGMENT_TYPE_DICT["comp-text"] = CompTextSegment
