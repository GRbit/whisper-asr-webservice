import gc
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from threading import Lock
from typing import Union

import torch

from app.config import CONFIG


class ASRModel(ABC):
    """
    Abstract base class for ASR (Automatic Speech Recognition) models.
    """

    model = None
    model_lock = Lock()
    last_activity_time = time.time()

    @abstractmethod
    def load_model(self):
        """
        Loads the model from the specified path.
        """
        pass

    @abstractmethod
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
        """
        Perform transcription on the given audio file.
        """
        pass

    @abstractmethod
    def language_detection(self, audio):
        """
        Perform language detection on the given audio file.
        """
        pass

    @contextmanager
    def loaded_model(self):
        """
        Holds the model lock, (re)loads the model if it was released and yields it.
        Loading and inference happen under the same lock, so the idle monitor
        cannot release the model in between.
        """
        with self.model_lock:
            self.last_activity_time = time.time()
            if self.model is None:
                self.load_model()
            try:
                yield self.model
            finally:
                # Count the end of a long-running job as activity too.
                self.last_activity_time = time.time()

    def monitor_idleness(self):
        """
        Monitors the idleness of the ASR model and releases the model if it has been idle for too long.
        """
        if CONFIG.MODEL_IDLE_TIMEOUT <= 0:
            return
        while True:
            time.sleep(15)
            with self.model_lock:
                if time.time() - self.last_activity_time > CONFIG.MODEL_IDLE_TIMEOUT:
                    self.release_model()
                    break

    def release_model(self):
        """
        Unloads the model from memory and clears any cached GPU memory.
        """
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        print("Model unloaded due to timeout")
