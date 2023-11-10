import re
import logging
import sacrebleu
from pathlib import Path
from typing import Dict
from sacrebleu.metrics.bleu import BLEU
import subprocess
import string

from sacremoses import MosesDetokenizer

from simuleval.evaluator.scorers.quality_scorer import QualityScorer, register_quality_scorer, add_sacrebleu_args

@register_quality_scorer("REM_TOK_BPE_BLEU")
class RemTokBPESacreBLEUScorer(QualityScorer):
    """
    SacreBLEU Scorer with BPE removal for subword-nmt and moses detokenization

    Usage:
        :code:`--quality-metrics REM_BPE_BLEU`

    Additional command line arguments:

    .. argparse::
        :ref: simuleval.evaluator.scorers.quality_scorer.add_sacrebleu_args
        :passparser:
        :prog:
    """

    def __init__(self, tokenizer: str = "13a", target_lang: str = "en") -> None:
        super().__init__()
        self.logger = logging.getLogger("simuleval.scorer.bleu")
        self.tokenizer = tokenizer
        self.detokenizer = MosesDetokenizer(target_lang)

    def __call__(self, instances: Dict) -> float:
        try:
            self.decode_bpe(instances)
            self.decode(instances)
            return (
                BLEU(tokenize=self.tokenizer)
                .corpus_score(
                    [ins.prediction for ins in instances.values()],
                    [[ins.reference for ins in instances.values()]],
                )
                .score
            )
        except Exception as e:
            self.logger.error(str(e))
            return 0

    def decode_bpe(self, instances: Dict):
        for ins in instances.keys():
            sentence = instances[ins].prediction
            symbol = "@@ "
            instances[ins].prediction_list = (sentence + " ").replace(symbol, "").rstrip().split(' ')

    def decode(self, instances: Dict):
        for ins in instances.keys():
            sentence = instances[ins].prediction
            instances[ins].prediction_list = self.detokenizer.detokenize(sentence.split()).split(' ')
            print(instances[ins].prediction)
    
    @staticmethod
    def add_args(parser):
        add_sacrebleu_args(parser)

    @classmethod
    def from_args(cls, args):
        return cls(args.sacrebleu_tokenizer)
