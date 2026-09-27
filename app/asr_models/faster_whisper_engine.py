from io import StringIO
from threading import Thread
from typing import Union

import whisper
from faster_whisper import WhisperModel

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG
from app.utils import WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT

WRITERS = {"srt": WriteSRT, "vtt": WriteVTT, "tsv": WriteTSV, "json": WriteJSON, "txt": WriteTXT}


class FasterWhisperASR(ASRModel):

    def load_model(self):
        self.model = WhisperModel(
            model_size_or_path=CONFIG.MODEL_NAME,
            device=CONFIG.DEVICE,
            compute_type=CONFIG.MODEL_QUANTIZATION,
            download_root=CONFIG.MODEL_PATH,
        )

        Thread(target=self.monitor_idleness, daemon=True).start()

    def transcribe(
        self,
        audio,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        vad_filter: Union[bool, None],
        word_timestamps: Union[bool, None],
        options: Union[dict, None],
        output,
    ):
        options_dict = {"task": task}
        if language:
            options_dict["language"] = language
        if initial_prompt:
            options_dict["initial_prompt"] = initial_prompt
        if vad_filter:
            options_dict["vad_filter"] = True
        if word_timestamps:
            options_dict["word_timestamps"] = True
        with self.loaded_model() as model:
            segment_generator, info = model.transcribe(audio, beam_size=5, **options_dict)
            segments = list(segment_generator)
        result = {
            "language": options_dict.get("language", info.language),
            "segments": segments,
            "text": "".join(segment.text for segment in segments),
        }

        output_file = StringIO()
        self.write_result(result, output_file, output)
        output_file.seek(0)

        return output_file

    def language_detection(self, audio):
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.pad_or_trim(audio)

        # detect the spoken language
        with self.loaded_model() as model:
            detected_lang_code, detected_language_confidence, _ = model.detect_language(audio)

        return detected_lang_code, detected_language_confidence

    def write_result(self, result: dict, file: StringIO, output: Union[str, None]):
        WRITERS.get(output, WriteTXT)(output_dir=None).write_result(result, file=file)
