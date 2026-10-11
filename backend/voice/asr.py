
import os
import re
import time
import wave
from typing import Dict, Optional

from faster_whisper import WhisperModel
from utils.logger import get_logger


class SpeechRecognizer:
    """
    JARVIS ASR Intelligence v3.

    Features:
    - Local Whisper inference
    - Fast initial decoding
    - Retry for suspicious transcriptions
    - Comparison of decoding quality
    - Structured transcription diagnostics
    - Compatibility with VoiceEngine v2
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 3,
        retry_beam_size: int = 5,
    ):
        self.logger = get_logger("JARVIS.Voice.ASR")

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = max(1, beam_size)
        self.retry_beam_size = max(
            self.beam_size,
            retry_beam_size,
        )

        self.logger.info(
            "Loading Whisper model | "
            f"model={model_size} | "
            f"device={device} | "
            f"compute={compute_type} | "
            f"beam={self.beam_size}"
        )

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        self.logger.info("Whisper model loaded")

    @staticmethod
    def _audio_duration(audio_path: str) -> float:
        with wave.open(audio_path, "rb") as wav:
            sample_rate = wav.getframerate()

            if sample_rate <= 0:
                return 0.0

            return wav.getnframes() / float(sample_rate)

    @staticmethod
    def _repetition_ratio(text: str) -> float:
        """
        Estimate repeated-word behavior without using a
        fixed list of expected user phrases.
        """
        words = re.findall(
            r"\b[\w']+\b",
            text.lower(),
        )

        if len(words) < 3:
            return 0.0

        repeated = sum(
            1
            for previous, current in zip(
                words,
                words[1:],
            )
            if previous == current
        )

        return repeated / max(1, len(words) - 1)

    @classmethod
    def _is_suspicious(cls, result: Dict) -> bool:
        text = result.get("text", "").strip()
        segments = result.get("segments", [])

        if not text:
            return True

        if cls._repetition_ratio(text) >= 0.5:
            return True

        if not segments:
            return True

        # Poor decoding confidence can justify a retry.
        average_logprob = sum(
            item["average_log_probability"]
            for item in segments
        ) / len(segments)

        if average_logprob < -0.9:
            return True

        if any(
            item["no_speech_probability"] > 0.65
            for item in segments
        ):
            return True

        if any(
            item["compression_ratio"] > 2.8
            for item in segments
        ):
            return True

        return False

    def _decode(
        self,
        audio_path: str,
        language: Optional[str],
        beam_size: int,
    ) -> Dict:
        """
        Decode once. Consume the segment generator before
        returning so all transcription metrics are available.
        """
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

        accepted = []
        rejected = []
        text_parts = []

        for segment in segments:
            text = segment.text.strip()

            if not text:
                continue

            no_speech = float(
                getattr(segment, "no_speech_prob", 0.0)
                or 0.0
            )

            logprob = float(
                getattr(segment, "avg_logprob", 0.0)
                or 0.0
            )

            compression = float(
                getattr(segment, "compression_ratio", 0.0)
                or 0.0
            )

            data = {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
                "no_speech_probability": no_speech,
                "average_log_probability": logprob,
                "compression_ratio": compression,
            }

            reasons = []

            if no_speech > 0.75:
                reasons.append("high_no_speech_probability")

            if logprob < -1.5:
                reasons.append("low_average_log_probability")

            if compression > 3.0:
                reasons.append("abnormal_compression_ratio")

            if reasons:
                rejected.append({
                    **data,
                    "rejection_reasons": reasons,
                })
                continue

            accepted.append(data)
            text_parts.append(text)

        text = " ".join(text_parts).strip()

        return {
            "text": text,
            "language": getattr(
                info,
                "language",
                language,
            ),
            "language_probability": float(
                getattr(
                    info,
                    "language_probability",
                    0.0,
                ) or 0.0
            ),
            "segments": accepted,
            "rejected_segments": rejected,
            "beam_size": beam_size,
        }

    @staticmethod
    def _quality_score(result: Dict) -> float:
        """
        Compare decoding candidates using model-reported
        metrics. This is a heuristic, not a correctness score.
        """
        segments = result.get("segments", [])

        if not segments:
            return float("-inf")

        avg_logprob = sum(
            item["average_log_probability"]
            for item in segments
        ) / len(segments)

        avg_no_speech = sum(
            item["no_speech_probability"]
            for item in segments
        ) / len(segments)

        avg_compression = sum(
            item["compression_ratio"]
            for item in segments
        ) / len(segments)

        repetition = SpeechRecognizer._repetition_ratio(
            result.get("text", "")
        )

        return (
            avg_logprob
            - avg_no_speech
            - max(0.0, avg_compression - 2.4) * 0.5
            - repetition
        )

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = "en",
    ) -> Dict:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        try:
            duration = self._audio_duration(audio_path)
        except (wave.Error, EOFError) as exc:
            raise ValueError(
                f"Invalid WAV file: {exc}"
            ) from exc

        if duration < 0.4:
            return {
                "text": "",
                "language": language,
                "language_probability": 0.0,
                "segments": [],
                "rejected_segments": [],
                "audio_duration": duration,
                "transcription_time": 0.0,
                "quality": "insufficient_audio",
                "retry_attempted": False,
                "attempts": 0,
            }

        started = time.perf_counter()

        self.logger.info(
            f"Transcribing audio | duration={duration:.2f}s"
        )

        first = self._decode(
            audio_path,
            language,
            self.beam_size,
        )

        selected = first
        retry_attempted = False
        attempts = 1

        if self._is_suspicious(first):
            retry_attempted = True
            attempts = 2

            self.logger.warning(
                "Suspicious transcription | "
                "retrying with larger beam search"
            )

            try:
                second = self._decode(
                    audio_path,
                    language,
                    self.retry_beam_size,
                )

                first_score = self._quality_score(first)
                second_score = self._quality_score(second)

                self.logger.info(
                    "ASR retry comparison | "
                    f"first_score={first_score:.3f} | "
                    f"retry_score={second_score:.3f}"
                )

                if second_score > first_score:
                    selected = second
                else:
                    selected = first

            except Exception as exc:
                # Keep the first result if the retry fails.
                self.logger.warning(
                    f"ASR retry failed; keeping first result: {exc}"
                )

                attempts = 1

        text = selected.get("text", "").strip()
        rejected = selected.get("rejected_segments", [])

        if not text:
            quality = "no_reliable_transcription"
        elif rejected:
            quality = "partial"
        elif self._is_suspicious(selected):
            quality = "uncertain"
        else:
            quality = "transcribed"

        elapsed = time.perf_counter() - started

        result = {
            **selected,
            "audio_duration": round(duration, 3),
            "transcription_time": round(elapsed, 3),
            "quality": quality,
            "retry_attempted": retry_attempted,
            "attempts": attempts,
        }

        self.logger.info(
            "Transcription completed | "
            f"quality={quality} | "
            f"language={result['language']} | "
            f"text={text!r} | "
            f"beam={result['beam_size']} | "
            f"retry={retry_attempted} | "
            f"processing={elapsed:.2f}s"
        )

        return result
