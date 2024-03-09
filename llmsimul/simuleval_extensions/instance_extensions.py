from simuleval.evaluator.instance import *

"""
Adds the possibility of a "computationally aware" text input instance.
Really just allows for easy step-by-step computation latency measurement
without having to fork SimulEval.
"""

class CompTextInputInstance(TextInputInstance):
    def step_to_elapsed(self, step, current_time):
        return (current_time - self.start_time) * 1000

class CompTextToTextInstance(CompTextInputInstance, TextOutputInstance):
    pass

INSTANCE_TYPE_DICT["comp-text-text"] = CompTextToTextInstance
