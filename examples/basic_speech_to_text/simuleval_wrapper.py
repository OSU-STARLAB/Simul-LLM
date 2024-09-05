"""
Command-line interface for simultaneous LLM evaluation.
Just sets up up some optional modules and imports them before,
calling SimulEval.

Fresh wrapper provided here to avoid having to import example
in main cli interface.
"""

from llmsimul.simuleval_extensions.instance_extensions import *
from llmsimul.simuleval_extensions.latency_extensions import *
from llmsimul.simuleval_extensions.dataloader_extensions import *
from llmsimul.simuleval_extensions.segment_extensions import *
from llmsimul.simuleval_extensions.comet_scorer import *
from llmsimul.simuleval_extensions.chrf_scorer import *

from examples.basic_speech_to_text.simuleval_extensions.speech_instance_extension import *
from examples.basic_speech_to_text.simuleval_extensions.speech_dataloader_extension import *
from examples.basic_speech_to_text.simuleval_extensions.speech_segment_extension import *

import simuleval.cli

def main():
    simuleval.cli.main()

if __name__ == "__main__":
    main()
