from simuleval.evaluator.instance import *

"""
Adds the possibility of a "computationally aware" text input instance.
Really just allows for easy step-by-step computation latency measurement
without having to fork SimulEval.
"""

class CompTextInputInstance(TextInputInstance):
    
    def __init__(
        self,
        index: int,
        dataloader: Optional[Union[SpeechToTextDataloader, TextToTextDataloader]],
        args: Optional[Namespace],
    ):
        super().__init__(index, dataloader, args)
        self.source_latency, _ = latency_parser().parse_known_args(sys.argv[1:])
    
    def send_source(self, config_dict: Optional[Dict]):
        if self.step == 0:
            self.start_time = time.time()
        return super().send_source(config_dict)
    
    def step_to_elapsed(self, step, current_time):
        return step * self.source_latency + (current_time - self.start_time) * 1000
    
    def latency_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--inherent-latency",
            type=int,
            default=360,
            help="Inherent latency of modeled transcription. Assumes around 167 wpm in speech by default, slightly faster than average.",
        )
        return parser

class CompTextToTextInstance(CompTextInputInstance, TextOutputInstance):
    pass

INSTANCE_TYPE_DICT["comp-text-text"] = CompTextToTextInstance
