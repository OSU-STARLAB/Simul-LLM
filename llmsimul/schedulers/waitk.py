"""
    Models wait-k behavior. Requires only the current source context
    step and the target context step to function.

    NOTE: we do not directly return READ/WRITE decisions, as there is
          information we may not have access to here. We only return
          whether the criteria for the scheduler was met.
"""

from translation_scheduler import TranslationScheduler

class WaitkScheduler(TranslationScheduler):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
    
    def decision(self, *args):
        src_len = args[0]
        tgt_len = args[1]

        assert type(src_len) is int and type(tgt_len) is int, "Arguments for WaitK scheduler need to be integers!"

        if len(args) > 2:
            print("Passed more than two arguments into a WaitK scheduler, was that intended?")

        lagging = src_len - tgt_len
        return lagging >= self.k 