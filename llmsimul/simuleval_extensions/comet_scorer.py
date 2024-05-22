import re
import logging
from pathlib import Path
from typing import Dict
import subprocess
import string

from evaluate import load

from simuleval.evaluator.scorers.quality_scorer import QualityScorer, register_quality_scorer, add_sacrebleu_args

@register_quality_scorer("COMET")
class COMETScorer(QualityScorer):
    """
    COMET Scorer for semantic similarity scoring

    Usage:
        :code:`--quality-metrics COMET --comet-source-text <insert-file-name>`

    Additional command line arguments:

    .. argparse::
        :ref: simuleval.evaluator.scorers.quality_scorer.add_sacrebleu_args
        :passparser:
        :prog:
    """

    def __init__(self, comet_model: str, comet_source_text: str) -> None:
        super().__init__()
        self.logger = logging.getLogger("simuleval.scorer.comet")
        self.comet_eval = load(comet_model) if comet_model is not None else load('comet')

        f = open(comet_source_text, "r")
        self.source = f.readlines()
        f.close()
        

    def __call__(self, instances: Dict) -> float:
        try:
            return (
               self.comet_eval.compute(
                predictions=[ins.prediction for ins in instances.values()],
                references=[ins.reference for ins in instances.values()],
                sources=[source for source in self.source],
               )["mean_score"])
        except Exception as e:
            self.logger.error(str(e))
            return 0

    
    # ideally we would reuse the source text from elsewhere, but SimulEval doesn't expose that to evaluators
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--comet-model",
            type=str,
            default=None,
            help="The path to the Huggingface model that should be loaded by the comet scorer.",
        )
        parser.add_argument(
            "--comet-source-text",
            type=str,
            default=None,
            help="The path to the employed source text for COMET evaluation.",
        )

    @classmethod
    def from_args(cls, args):
        return cls(args.comet_model, args.comet_source_text)
