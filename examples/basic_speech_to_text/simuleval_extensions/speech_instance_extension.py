from simuleval.evaluator.instance import *
from llmsimul.simuleval_extensions.instance_extensions import *

"""
    Have to copy the SpeechInputInstance class from SimulEval for this example
    to manage strict latency requirements of maintaining k-lagging factor. 

    Only major difference is the renaming of `self.source_finished_reading` to the
    variable `self.invisible_source_finished_reading`, this avoids the SimulEval
    evaluator from reading the aforementioned variable and signalling an earlier end
    to evaluation than expected, resulting in aberrant behavior.
"""

class CustomSpeechInputInstance(Instance):
    def __init__(
        self,
        index: int,
        dataloader: Optional[SpeechToTextDataloader],
        args: Optional[Namespace],
    ):
        super().__init__(index, dataloader, args)
        self.args = args
        self.sample_rate_value = None
        self.sample_list = None
        self.invisible_source_finished_reading = False
        self.dataloader: SpeechToTextDataloader

    @property
    def sample_rate(self):
        if self.sample_rate_value is None:
            self.audio_info = self.dataloader.get_source_audio_info(self.index)
            self.sample_rate_value = self.audio_info.samplerate
        return self.sample_rate_value

    @property
    def samples(self) -> List[float]:
        if self.sample_list is None:
            self.sample_list = self.source
        return self.sample_list

    @property
    def is_finish_source(self):
        return self.step == len(self.samples)

    def send_source(self, segment_size=10):
        if self.step == 0:
            self.start_time = time.time()
        assert segment_size >= 1, "instance size has to larger than 1 ms"

        num_samples = math.ceil(segment_size / 1000 * self.sample_rate)

        if self.step < len(self.samples):
            if self.step + num_samples >= len(self.samples):
                # Pad zeros if the requested number of samples
                # are more than available samples.
                samples = self.samples[self.step :]  # noqa E203
                is_finished = True
                self.invisible_source_finished_reading = True
            else:
                samples = self.samples[self.step : self.step + num_samples]  # noqa E203
                is_finished = False

            self.step = min(self.step + num_samples, len(self.samples))

            segment = SpeechSegment(
                index=self.len_sample_to_ms(self.step),
                content=samples,
                sample_rate=self.audio_info.samplerate,
                finished=is_finished,
                tgt_lang=self.tgt_lang,
            )

        else:
            # Finish reading this audio
            segment = EmptySegment(
                index=self.len_sample_to_ms(self.step),
                finished=True,
            )
            self.invisible_source_finished_reading = True

        return segment

    @property
    def source_length(self):
        # In milliseconds
        return self.len_sample_to_ms(len(self.samples))

    @property
    def source_info(self):
        return str(self.audio_info).split("\n")

    def len_sample_to_ms(self, length):
        assert getattr(self, "sample_rate", None), "Read a audio file first"
        return length * 1000 / self.sample_rate

    def len_ms_to_samples(self, length):
        assert getattr(self, "sample_rate", None), "Read a audio file first"
        return math.ceil(length / 1000 * self.sample_rate)

    def step_to_delay(self, step):
        return self.len_sample_to_ms(self.step)

    def step_to_elapsed(self, step, current_time):
        return self.len_sample_to_ms(step) + (current_time - self.start_time) * 1000


class CustomSpeechToCompTextInstance(CustomSpeechInputInstance, CompTextOutputInstance):
    pass

class CustomSpeechToTextInstance(CustomSpeechInputInstance, TextOutputInstance):
    pass

INSTANCE_TYPE_DICT["custom-speech-comp-text"] = CustomSpeechToCompTextInstance
INSTANCE_TYPE_DICT["custom-speech-text"] = CustomSpeechToTextInstance
