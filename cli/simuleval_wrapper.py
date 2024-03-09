"""
Command-line interface for simultaneous LLM evaluation.
Just setups up some optional modules and imports them before,
calling SimulEval.
"""

from llmsimul.simuleval_extensions.instance_extensions import *
from llmsimul.simuleval_extensions.latency_extensions import *
from llmsimul.simuleval_extensions.dataloader_extensions import *

import simuleval.cli

def main():
    simuleval.cli.main()

if __name__ == "__main__":
    main()
