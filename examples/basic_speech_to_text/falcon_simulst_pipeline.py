from simuleval.agents import AgentPipeline
from simuleval.utils import entrypoint
from examples.basic_speech_to_text.whisper_agent import WaitkWhisperAgent
from examples.basic_speech_to_text.falcon_dummy_text_agent import DummyFalconTextAgent

@entrypoint
class WhisperToFalconAgentPipeline(AgentPipeline):
    pipeline = [
        WaitkWhisperAgent,
        DummyFalconTextAgent,
    ]
