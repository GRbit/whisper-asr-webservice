from io import StringIO
from threading import Thread
from typing import Union

import whisperx
from whisperx.audio import N_SAMPLES
from whisperx.diarize import DiarizationPipeline
from whisperx.utils import WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG

WRITERS = {"srt": WriteSRT, "vtt": WriteVTT, "tsv": WriteTSV, "json": WriteJSON, "txt": WriteTXT}


class WhisperXASR(ASRModel):

    def load_model(self):
        # self.model is a dict here; it is rebuilt from scratch on every (re)load after an idle release.
        self.model = {
            'whisperx': whisperx.load_model(
                CONFIG.MODEL_NAME,
                device=CONFIG.DEVICE,
                compute_type=CONFIG.MODEL_QUANTIZATION,
                asr_options={"without_timestamps": False},
            ),
            'diarize_model': None,
            'align_model': {},
        }

        if CONFIG.HF_TOKEN != "":
            self.model['diarize_model'] = DiarizationPipeline(use_auth_token=CONFIG.HF_TOKEN, device=CONFIG.DEVICE)

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
        with self.loaded_model() as model:
            result = model['whisperx'].transcribe(audio, **options_dict)
            language = result["language"]

            # Load the required model and cache it
            # If we transcribe models in many different languages, this may lead to OOM problems
            if language not in model['align_model']:
                model['align_model'][language] = whisperx.load_align_model(language_code=language, device=CONFIG.DEVICE)
            model_x, metadata = model['align_model'][language]

            # Align whisper output
            result = whisperx.align(
                result["segments"], model_x, metadata, audio, CONFIG.DEVICE, return_char_alignments=False
            )

            if options.get("diarize", False) and model['diarize_model'] is not None:
                # add min/max number of speakers if known
                diarize_segments = model['diarize_model'](
                    audio, min_speakers=options.get("min_speakers"), max_speakers=options.get("max_speakers")
                )
                result = whisperx.assign_word_speakers(diarize_segments, result)
        result["language"] = language

        output_file = StringIO()
        self.write_result(result, output_file, output)
        output_file.seek(0)

        return output_file

    def language_detection(self, audio):
        if audio.shape[0] < N_SAMPLES:
            print("Warning: audio is shorter than 30s, language detection may be inaccurate.")
        with self.loaded_model() as model:
            results = model['whisperx'].model.detect_language(audio)
        language = results[0]
        language_probability = round(float(results[1]), 2)
        print(f"Detected language: {language} ({language_probability}) in first 30s of audio...")
        return language, language_probability

    def write_result(self, result: dict, file: StringIO, output: Union[str, None]):
        options = {
            "max_line_width": CONFIG.SUBTITLE_MAX_LINE_WIDTH,
            "max_line_count": CONFIG.SUBTITLE_MAX_LINE_COUNT,
            "highlight_words": CONFIG.SUBTITLE_HIGHLIGHT_WORDS,
        }
        WRITERS.get(output, WriteTXT)(output_dir=None).write_result(result, file=file, options=options)
