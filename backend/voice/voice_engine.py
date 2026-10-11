
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logger import get_logger

from voice.asr import SpeechRecognizer
from voice.audio_stream import MicrophoneStream
from voice.tts import TextToSpeech


class VoiceEngine:
    """
    JARVIS Conversational Voice Engine v2.

    Pipeline:
        Microphone -> Whisper -> Existing JARVIS Brain -> TTS

    Features:
        - Empty and unreliable transcription protection
        - Stage-by-stage diagnostics
        - Recoverable microphone, ASR, Brain and TTS errors
        - Temporary audio cleanup
        - Existing Brain and memory continuity
        - Backward-compatible run_once() and listen_once()
    """

    def __init__(
        self,
        jarvis,
        whisper_model: str = "base",
        tts_enabled: bool = True,
    ):
        self.logger = get_logger("JARVIS.Voice")
        self.jarvis = jarvis

        self.microphone = MicrophoneStream()

        self.asr = SpeechRecognizer(
            model_size=whisper_model,
            device="cpu",
            compute_type="int8",
        )

        self.tts_enabled = tts_enabled
        self.tts: Optional[TextToSpeech] = None

        if self.tts_enabled:
            self.tts = TextToSpeech(
                rate=175,
                volume=1.0,
            )

        self.running = False
        self._closed = False
        self.turn_count = 0

        self.logger.info(
            "Conversational Voice Engine v2 initialized"
        )

    # ---------------------------------------------------------
    # START / STOP
    # ---------------------------------------------------------

    def start(self):
        """Start microphone capture safely."""
        if self.running:
            return True

        try:
            self.microphone.start()
            self.running = True
            self._closed = False

            self.logger.info(
                "Voice Engine started | stage=ready"
            )
            return True

        except Exception as exc:
            self.logger.error(
                f"Microphone startup failed | error={exc}"
            )
            self.running = False
            return False

    def stop(self):
        """Stop audio and speech output safely."""
        self.running = False

        try:
            self.microphone.stop()
        except Exception as exc:
            self.logger.warning(
                f"Microphone shutdown warning: {exc}"
            )

        if self.tts is not None:
            try:
                self.tts.stop()
            except Exception as exc:
                self.logger.warning(
                    f"TTS shutdown warning: {exc}"
                )

        self.logger.info("Voice Engine stopped")

    def close(self):
        """Release resources when supported by the components."""
        if self._closed:
            return

        self.stop()

        close_method = getattr(
            self.microphone,
            "close",
            None,
        )

        if callable(close_method):
            try:
                close_method()
            except Exception as exc:
                self.logger.warning(
                    f"Microphone close warning: {exc}"
                )

        self._closed = True

    # ---------------------------------------------------------
    # RESULT HELPERS
    # ---------------------------------------------------------

    @staticmethod
    def _result(
        *,
        success: bool,
        stage: str,
        speech_detected: bool = False,
        text: str = "",
        response: Optional[str] = None,
        spoken: bool = False,
        error: Optional[str] = None,
        **extra,
    ) -> Dict[str, Any]:
        """Create a consistent result without changing Brain output."""
        result = {
            "success": success,
            "stage": stage,
            "speech_detected": speech_detected,
            "text": text,
            "response": response,
            "spoken": spoken,
        }

        if error:
            result["error"] = error

        result.update(extra)
        return result

    @staticmethod
    def _extract_response(result: Any) -> Optional[str]:
        """Support the existing Brain's string or dictionary output."""
        if isinstance(result, str):
            response = result

        elif isinstance(result, dict):
            response = result.get("response")

        else:
            response = None

        if response is None:
            return None

        response = str(response).strip()
        return response or None

    # ---------------------------------------------------------
    # ONE CONVERSATION TURN
    # ---------------------------------------------------------

    def listen_once(
        self,
        timeout_seconds: float = 10.0,
        speak_response: bool = True,
    ) -> Dict[str, Any]:
        """
        Capture one user turn, process it through the existing Brain,
        and optionally speak the response.

        A failed turn returns diagnostics instead of terminating
        the conversation loop.
        """

        self.turn_count += 1
        turn_id = self.turn_count
        started = time.perf_counter()

        self.logger.info(
            f"Turn {turn_id} | stage=capture | status=started"
        )

        # -----------------------------------------------------
        # Ensure microphone is running
        # -----------------------------------------------------

        if not self.running:
            if not self.start():
                return self._result(
                    success=False,
                    stage="capture",
                    error="Microphone could not be started.",
                    turn_id=turn_id,
                )

        # -----------------------------------------------------
        # Capture speech
        # -----------------------------------------------------

        try:
            audio = self.microphone.record_speech(
                timeout_seconds=timeout_seconds
            )

        except Exception as exc:
            self.logger.error(
                f"Turn {turn_id} | stage=capture | error={exc}"
            )

            # Mark the engine stopped so the next turn can retry
            # starting the microphone.
            self.running = False

            return self._result(
                success=False,
                stage="capture",
                error=str(exc),
                turn_id=turn_id,
            )

        if audio is None:
            self.logger.info(
                f"Turn {turn_id} | stage=capture | status=no_speech"
            )

            return self._result(
                success=True,
                stage="idle",
                speech_detected=False,
                turn_id=turn_id,
                duration_seconds=round(
                    time.perf_counter() - started, 3
                ),
            )

        temp_path = None

        try:
            # -------------------------------------------------
            # Save temporary WAV
            # -------------------------------------------------

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as temp_file:
                temp_path = temp_file.name

            self.microphone.save_wav(
                audio,
                temp_path,
            )

            self.logger.info(
                f"Turn {turn_id} | stage=capture | status=complete"
            )

            # -------------------------------------------------
            # Transcribe speech
            # -------------------------------------------------

            self.logger.info(
                f"Turn {turn_id} | stage=transcription | status=started"
            )

            transcription = self.asr.transcribe(
                temp_path,
                language="en",
            )

            if not isinstance(transcription, dict):
                raise TypeError(
                    "ASR returned an unexpected result type."
                )

            text = str(
                transcription.get("text") or ""
            ).strip()

            quality = str(
                transcription.get("quality") or ""
            ).strip().lower()

            if not text:
                self.logger.warning(
                    f"Turn {turn_id} | stage=transcription | "
                    "status=empty"
                )

                return self._result(
                    success=False,
                    stage="transcription",
                    speech_detected=True,
                    error="Whisper returned no usable text.",
                    transcription=transcription,
                    turn_id=turn_id,
                )

            # The upgraded ASR may explicitly mark a transcription
            # as unreliable. Do not send it to the Brain.
            unreliable_qualities = {
                "insufficient_audio",
                "no_reliable_transcription",
                "rejected",
                "unreliable",
            }

            if quality in unreliable_qualities:
                self.logger.warning(
                    f"Turn {turn_id} | stage=transcription | "
                    f"status=rejected | quality={quality}"
                )

                return self._result(
                    success=False,
                    stage="transcription",
                    speech_detected=True,
                    text=text,
                    error=f"Transcription quality was {quality}.",
                    transcription=transcription,
                    turn_id=turn_id,
                )

            self.logger.info(
                f"Turn {turn_id} | stage=transcription | "
                f"status=complete | quality={quality or 'unspecified'} | "
                f"text={text!r}"
            )

            # -------------------------------------------------
            # Existing JARVIS Brain
            # -------------------------------------------------

            self.logger.info(
                f"Turn {turn_id} | stage=brain | status=started"
            )

            try:
                brain_result = self.jarvis.receive(text)

            except Exception as exc:
                self.logger.error(
                    f"Turn {turn_id} | stage=brain | error={exc}"
                )

                return self._result(
                    success=False,
                    stage="brain",
                    speech_detected=True,
                    text=text,
                    error=str(exc),
                    transcription=transcription,
                    turn_id=turn_id,
                )

            response = self._extract_response(brain_result)

            if not response:
                self.logger.warning(
                    f"Turn {turn_id} | stage=brain | "
                    "status=empty_response"
                )

                return self._result(
                    success=False,
                    stage="brain",
                    speech_detected=True,
                    text=text,
                    error="JARVIS Brain returned no response.",
                    transcription=transcription,
                    jarvis=brain_result,
                    turn_id=turn_id,
                )

            self.logger.info(
                f"Turn {turn_id} | stage=brain | status=complete"
            )

            # -------------------------------------------------
            # Speak response
            # -------------------------------------------------

            spoken = False
            tts_error = None

            if (
                speak_response
                and self.tts_enabled
                and self.tts is not None
            ):
                self.logger.info(
                    f"Turn {turn_id} | stage=tts | status=started"
                )

                try:
                    spoken = bool(
                        self.tts.speak(response)
                    )

                    self.logger.info(
                        f"Turn {turn_id} | stage=tts | "
                        f"status={'complete' if spoken else 'not_spoken'}"
                    )

                except Exception as exc:
                    tts_error = str(exc)

                    self.logger.error(
                        f"Turn {turn_id} | stage=tts | error={exc}"
                    )

            elapsed = time.perf_counter() - started

            self.logger.info(
                f"Turn {turn_id} | stage=complete | "
                f"elapsed={elapsed:.2f}s"
            )

            return self._result(
                success=True,
                stage="complete",
                speech_detected=True,
                text=text,
                response=response,
                spoken=spoken,
                transcription=transcription,
                jarvis=brain_result,
                turn_id=turn_id,
                duration_seconds=round(elapsed, 3),
                tts_error=tts_error,
            )

        except Exception as exc:
            self.logger.error(
                f"Turn {turn_id} | stage=processing | error={exc}"
            )

            return self._result(
                success=False,
                stage="processing",
                speech_detected=True,
                error=str(exc),
                turn_id=turn_id,
            )

        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(
                        missing_ok=True
                    )
                except OSError as exc:
                    self.logger.warning(
                        f"Could not remove temporary audio: {exc}"
                    )

    # ---------------------------------------------------------
    # BACKWARD-COMPATIBLE ENTRY POINT
    # ---------------------------------------------------------

    def run_once(
        self,
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        return self.listen_once(
            timeout_seconds=timeout_seconds,
            speak_response=True,
        )
