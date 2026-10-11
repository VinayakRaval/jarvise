import threading
from typing import Optional

import pyttsx3

from utils.logger import get_logger


class TextToSpeech:
    """
    Local Windows Text-to-Speech engine.

    No cloud API.
    No API key.
    """

    def __init__(
        self,
        rate: int = 175,
        volume: float = 1.0,
    ):
        self.logger = get_logger(
            "JARVIS.Voice.TTS"
        )

        self.rate = rate
        self.volume = volume

        self.engine = None
        self._lock = threading.Lock()

        self._initialize()

    def _initialize(self):

        try:

            self.engine = pyttsx3.init()

            self.engine.setProperty(
                "rate",
                self.rate,
            )

            self.engine.setProperty(
                "volume",
                self.volume,
            )

            self.logger.info(
                "Windows TTS initialized"
            )

            self._log_voices()

        except Exception as exc:

            self.logger.error(
                f"TTS initialization failed: {exc}"
            )

            self.engine = None

    def _log_voices(self):

        if self.engine is None:
            return

        try:

            voices = self.engine.getProperty(
                "voices"
            )

            self.logger.info(
                f"Available TTS voices: {len(voices)}"
            )

            for index, voice in enumerate(
                voices
            ):

                self.logger.info(
                    f"TTS voice {index}: "
                    f"{voice.name}"
                )

        except Exception:
            pass

    def is_available(self) -> bool:

        return self.engine is not None

    def speak(
        self,
        text: str,
        wait: bool = True,
    ) -> bool:

        if not text or not text.strip():

            return False

        if self.engine is None:

            self._initialize()

        if self.engine is None:

            return False

        try:

            with self._lock:

                self.logger.info(
                    f"Speaking: {text}"
                )

                self.engine.say(
                    text.strip()
                )

                if wait:

                    self.engine.runAndWait()

            return True

        except Exception as exc:

            self.logger.error(
                f"TTS speech failed: {exc}"
            )

            return False

    def stop(self):

        if self.engine is None:
            return

        try:

            self.engine.stop()

        except Exception:
            pass

    def set_rate(
        self,
        rate: int,
    ):

        self.rate = rate

        if self.engine is not None:

            self.engine.setProperty(
                "rate",
                rate,
            )

    def set_volume(
        self,
        volume: float,
    ):

        volume = max(
            0.0,
            min(
                1.0,
                volume,
            ),
        )

        self.volume = volume

        if self.engine is not None:

            self.engine.setProperty(
                "volume",
                volume,
            )