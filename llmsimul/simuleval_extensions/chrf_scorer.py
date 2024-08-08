import logging
from typing import Dict
from sacrebleu.metrics import CHRF

from simuleval.evaluator.scorers.quality_scorer import QualityScorer, register_quality_scorer

@register_quality_scorer("CHRF")
class CHRFScorer(QualityScorer):
    """
    CHRF Scorer for semantic similarity scoring

    """

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger("simuleval.scorer.chrf")

        self.chrf = CHRF(word_order=2)
        

    def __call__(self, instances: Dict) -> float:
        try:
            return (
               self.chrf.corpus_score(
                    [ins.prediction for ins in instances.values()],
                    [[ins.reference for ins in instances.values()]],
                ).score)
        except Exception as e:
            self.logger.error(str(e))
            return 0
        
    @classmethod
    def from_args(cls, args):
        return cls()
