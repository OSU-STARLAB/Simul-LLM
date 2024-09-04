from simuleval.agents import AgentPipeline
from llmsimul.examples.basic_speech_to_text.whisper_agent import WaitkWhisperAgent
from llmsimul.falcon.falcon_simulmt_agent import FalconWaitkTextAgent

class WhisperToFalconAgentPipeline(AgentPipeline):
    pipeline = [
        WaitkWhisperAgent,
        FalconWaitkTextAgent,
    ]
