from typing import Optional
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.data.segments import SpeechSegment
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction

from llmsimul.schedulers.waitk import WaitkScheduler

import whisper
import numpy

"""
    Based on the example found in SimulEval for a similar agent, but adapted to
    align more closely with the construction of typical agents in Simul-LLM in addition
    to being a bit more modular.
"""

@entrypoint
class WaitkWhisperAgent(SpeechToTextAgent):
    """
    The agent derives the number of seconds from input audio and provided sampling info.
    """

    def __init__(self, args):
        super().__init__(args)
        self.source_segment_size = args.source_segment_size
        self.source_language = args.source_language
        self.continuous_write = args.continuous_write
        self.model_size = args.model_size
        
        # only waitk implemented, but easy to replace with other options
        # we assume for waitk whisper that this is the only scheduler we would
        # really use
        if self.translation_scheduler == "waitk":
            self.scheduler = WaitkScheduler(args)
        else:
            raise NotImplementedError
        
        self.model = whisper.load_model(self.model_size)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--translation-scheduler", type=str, default="waitk",
                                help="Name of translation scheduler of choice. Wait-k is the default.")
        parser.add_argument("--source-lang", default="en", type=str)
        parser.add_argument("--model-size", default="tiny", type=str)
        
        # entrance for schedulers
        WaitkScheduler.add_args(parser)


    def policy(self, states: Optional[AgentStates] = None):
        # vestigial behavior for non state-based agents
        if states is None:
            states = self.states

        if states.source_sample_rate == 0:
            # empty source, source_sample_rate not set yet
            length_in_seconds = 0
        else:
            length_in_seconds = float(len(states.source)) / states.source_sample_rate

        previous_translation = " ".join(states.target)

        decision = self.scheduler(length_in_seconds * 1000 / self.source_segment_size, len(states.target))

        if not states.source_finished:
            if not decision:
                return ReadAction()

        previous_translation = " ".join(states.target)
        # We use the previous translation as a prefix.
        options = whisper.DecodingOptions(
            prefix=previous_translation,
            language=self.source_language,
            without_timestamps=True,
            fp16=False,
        )

        # We encode the whole audio to get the full transcription each time a new audio chunk is received.
        audio = whisper.pad_or_trim(numpy.array(states.source).astype("float32"))
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        output = self.model.decode(mel, options)
        prediction = output.text.split()[len(states.target):len(states.target) + 1]

        return WriteAction(
            content=" ".join(prediction),
            finished=states.source_finished,
        )
