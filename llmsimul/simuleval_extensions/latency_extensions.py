import logging
from statistics import mean
from typing import List, Union, Dict
from simuleval.evaluator.scorers.latency_scorer import (
        register_latency_scorer,
        LatencyScorer,
        LAALScorer,
)
from simuleval.evaluator.instance import Instance

@register_latency_scorer("CT_LAAL")
class CompText_LAALScorer(LAALScorer):
    """
    A simple modification of the SimulEval LAAL scorer to allow for some
    measurement of text-to-text latency when factoring in computation.
    This should only be used when one is expressly interested in the
    computational costs of specific methods. 

    Unlike in speech-to-text or speech-to-speech modalities, there is no
    explicit timing implied in text-to-text that would allow for the 
    simple addition of computational latency. Given that, it is difficult
    to compare raw text input LAAL (raw token lagging) to CT_LAAL (seconds)
    values and they should be considered separately.
    """

    # simply ignores the RuntimeError raised in the original generic
    # latency scorer
    def __call__(self, instances: Dict[int, Instance]) -> float:
        scores = []
        for index, ins in instances.items():
            delays = getattr(ins, self.timestamp_type, None)
            if delays is None or len(delays) == 0:
                logger.warn(f"Instance {index} has no delay information. Skipped")
                continue
            
            score = self.compute(ins)
            ins.metrics[self.metric_name] = score
            scores.append(score)

        return mean(scores)
    
    def compute(self, ins: Instance):
        """
        Function to compute latency on one sentence (instance).

        Args:
            ins: Instance: one instance

        Returns:
            float: the latency score on one sentence.
        """
        delays, source_length, target_length = self.get_delays_lengths(ins)
        if delays[0] > source_length:
            return delays[0]


        # necessary to avoid deflating CA latency, first latency measurement is "fraudulent"
        # in that t2t has no timing implications in the first place, so there is no
        # effective cost for loading in multiple input chunks without a cost enforced
        # by the environment
            
        if len(delays) > 1:
            delays = delays[1:]

        LAAL = 0
        gamma = max(len(delays), target_length) / source_length
        tau = 0
        for t_minus_1, d in enumerate(delays):
            LAAL += d - t_minus_1 / gamma
            tau = t_minus_1 + 1

            if d >= source_length:
                break
        LAAL /= tau
        return LAAL
