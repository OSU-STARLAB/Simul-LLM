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

    def __init__(
        self, computation_aware: bool = False, use_ref_len: bool = True
    ) -> None:
        super().__init__(computation_aware, use_ref_len)
        self.source_latency, _ = latency_parser().parse_known_args(sys.argv[1:])

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

    # modified to factor in assumed inherent token latency
    def compute(self, ins: Instance):
        """
        Function to compute latency on one sentence (instance).

        Args:
            ins: Instance: one instance

        Returns:
            float: the latency score on one sentence.
        """
        delays, source_length, target_length = self.get_delays_lengths(ins)
        source_length = source_length * self.source_latency
        if delays[0] > source_length:
            return delays[0]

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

    def latency_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--inherent-latency",
            type=int,
            default=360,
            help="Inherent latency of modeled transcription. Assumes around 167 wpm in speech by default, slightly faster than average.",
        )
        return parser
