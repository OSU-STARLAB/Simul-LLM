from typing import Optional
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.agents import SpeechToTextAgent
from simuleval.data.segments import SpeechSegment
from simuleval.agents.actions import WriteAction, ReadAction

from llmsimul.schedulers.waitk import WaitkScheduler

import whisper
import numpy

from examples.basic_speech_to_text.falcon_dummy_text_agent import DummyFalconTextAgent

"""
    Based on the example found in SimulEval for a similar agent, but adapted to
    align more closely with the construction of typical agents in Simul-LLM in addition
    to being a bit more modular.

    Target_type must be set to comp-text for pipeline compatibility with agents elsewhere
    in the framework, e.g. Falcon-based agents.
"""

class WaitkWhisperAgent(SpeechToTextAgent):
    
    source_type: str = "speech"
    target_type: str = "comp-text"

    """
    The agent derives the number of seconds from input audio and provided sampling info.
    """

    def __init__(self, args):
        super().__init__(args)
        self.source_lang = args.source_lang
        self.model_size = args.model_size
        
        # reuse translation scheduler for this example, easy to change
        # in this example, waitk structure is based on segment size
        self.translation_scheduler = args.translation_scheduler
        self.source_segment_size = args.source_segment_size

        # only waitk implemented, but easy to replace with other options
        # we assume for waitk whisper that this is the only scheduler we would
        # really use
        if self.translation_scheduler == "waitk":
            self.scheduler = WaitkScheduler(args)
        else:
            raise NotImplementedError
       
        if not self.source_lang == "en":
            print(f"Source language set to {self.source_lang}, double check that you're loading a multilingual Whisper model.")

        self.model = whisper.load_model(self.model_size)


    @staticmethod
    def add_args(parser):
        parser.add_argument("--model-size", default="tiny", type=str)
        
        # entrance for schedulers, normally necessary but removed for now
        # due to having to import add_args from other agents in pipeline
        # WaitkScheduler.add_args(parser)

        # temporary solution, have to add_args for non-entrypoint agents
        # takes care of args.source_lang and args.translation_scheduler
        # DummyFalconTextAgent.add_args(parser)


    def policy(self, states: Optional[AgentStates] = None):
        # vestigial behavior for non state-based agents
        if states is None:
            states = self.states

        if states.source_sample_rate == 0:
            # empty source, source_sample_rate not set yet
            length_in_seconds = 0
        else:
            length_in_seconds = float(len(states.source)) / states.source_sample_rate

        previous_transcription = " ".join(states.target)

        decision = self.scheduler(length_in_seconds * 1000 / self.source_segment_size, len(states.target))

        if not states.source_finished:
            if not decision:
                return ReadAction()

        previous_transcription = " ".join(states.target)
        # We use the previous transcription as a prefix.
        options = whisper.DecodingOptions(
            prefix=previous_transcription,
            language=self.source_lang,
            without_timestamps=True,
            fp16=False,
        )

        # We encode the whole audio to get the full transcription each time a new audio chunk is received.
        audio = whisper.pad_or_trim(numpy.array(states.source).astype("float32"))
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        output = self.model.decode(mel, options)
        
        # force only a single outputted word for strict k-lagging factor, can easily be set up
        # with a blockwise output where multiple are written at once or a buffer enables
        # more efficient computation
        if len(output.text.split()) == len(states.target):
            prediction = []
        else:
            prediction = output.text.split()[len(states.target):len(states.target) + 1]

        return WriteAction(
            content=" ".join(prediction),
            finished=(states.source_finished and (len(output.text.split()) == len(states.target))),
        )
