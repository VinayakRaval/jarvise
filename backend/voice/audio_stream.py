
import io
import time
import wave
from collections import deque
from typing import Optional

import numpy as np
import sounddevice as sd

from voice.vad import VoiceActivityDetector
from utils.logger import get_logger


class MicrophoneStream:
    """
    JARVIS Microphone Intelligence v3.

    Pipeline:
        Microphone
          -> Audio preprocessing
          -> Noise calibration
          -> Energy gate
          -> WebRTC VAD
          -> Speech segmentation
          -> WAV export

    Compatible with the existing VoiceEngine and ASR modules.
    """

    DEVICE = 1

    INPUT_SAMPLE_RATE = 44100
    PROCESS_SAMPLE_RATE = 16000

    CHANNELS = 1
    BLOCK_DURATION_MS = 30

    NOISE_CALIBRATION_SECONDS = 1.5

    ENERGY_MULTIPLIER = 2.2
    MIN_SPEECH_RMS = 120.0
    MAX_SPEECH_THRESHOLD = 650.0

    SPEECH_START_FRAMES = 4
    SILENCE_DURATION_SECONDS = 0.9
    PRE_ROLL_FRAMES = 8

    MIN_SPEECH_DURATION = 0.7
    MAX_SPEECH_DURATION = 15.0

    def __init__(
        self,
        device: Optional[int] = None,
        input_sample_rate: int = INPUT_SAMPLE_RATE,
        process_sample_rate: int = PROCESS_SAMPLE_RATE,
    ):
        self.logger = get_logger(
            "JARVIS.Voice.Microphone"
        )

        self.device = (
            self.DEVICE if device is None else device
        )

        self.input_sample_rate = input_sample_rate
        self.process_sample_rate = process_sample_rate
        self.channels = self.CHANNELS

        self.block_size = int(
            self.input_sample_rate
            * self.BLOCK_DURATION_MS
            / 1000
        )

        self.vad = VoiceActivityDetector(
            sample_rate=self.process_sample_rate,
            aggressiveness=2,
        )

        self.stream = None
        self.running = False

        self.noise_floor = None
        self.speech_threshold = None

        self.logger.info(
            f"Using microphone | device={self.device}"
        )

        try:
            info = sd.query_devices(self.device)

            self.logger.info(
                f"Microphone device | name={info['name']} | "
                f"default_rate={info['default_samplerate']}"
            )

        except Exception as exc:
            self.logger.warning(
                f"Could not query microphone: {exc}"
            )

    # ---------------------------------------------------------
    # MICROPHONE LIFECYCLE
    # ---------------------------------------------------------

    def start(self) -> None:
        if self.running:
            return

        self.logger.info("Starting microphone...")

        stream = sd.InputStream(
            device=self.device,
            samplerate=self.input_sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.block_size,
        )

        try:
            stream.start()
        except Exception:
            try:
                stream.close()
            except Exception:
                pass
            raise

        self.stream = stream
        self.running = True

        self.logger.info(
            "Microphone started | "
            f"sample_rate={self.input_sample_rate}"
        )

    def stop(self) -> None:
        stream = self.stream

        self.stream = None
        self.running = False

        if stream is not None:
            try:
                stream.stop()
            except Exception as exc:
                self.logger.debug(
                    f"Microphone stop detail: {exc}"
                )

            try:
                stream.close()
            except Exception as exc:
                self.logger.debug(
                    f"Microphone close detail: {exc}"
                )

        self.logger.info("Microphone stopped")

    def close(self) -> None:
        self.stop()

    # ---------------------------------------------------------
    # AUDIO PROCESSING
    # ---------------------------------------------------------

    @staticmethod
    def calculate_rms(audio: np.ndarray) -> float:
        if audio is None or audio.size == 0:
            return 0.0

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        return float(
            np.sqrt(
                np.mean(np.square(audio, dtype=np.float64))
            ) * 32768.0
        )

    @staticmethod
    def remove_dc(audio: np.ndarray) -> np.ndarray:
        if audio.size == 0:
            return audio

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        return audio - np.mean(audio)

    def resample(self, audio: np.ndarray) -> np.ndarray:
        """
        Resample mono audio using linear interpolation.

        The output length is calculated explicitly so the
        WebRTC VAD receives the expected 30 ms frame size.
        """
        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.size == 0:
            return np.array([], dtype=np.float32)

        if (
            self.input_sample_rate
            == self.process_sample_rate
        ):
            return audio.copy()

        output_length = round(
            len(audio)
            * self.process_sample_rate
            / self.input_sample_rate
        )

        if output_length <= 0:
            return np.array([], dtype=np.float32)

        old_positions = np.arange(
            len(audio),
            dtype=np.float64,
        )

        new_positions = (
            np.arange(
                output_length,
                dtype=np.float64,
            )
            * self.input_sample_rate
            / self.process_sample_rate
        )

        return np.interp(
            new_positions,
            old_positions,
            audio,
        ).astype(np.float32)

    def prepare_audio(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        if audio.size == 0:
            return audio

        audio = self.remove_dc(audio)
        audio = self.resample(audio)

        return np.clip(
            audio,
            -1.0,
            1.0,
        )

    def read_block(self) -> np.ndarray:
        if not self.running or self.stream is None:
            raise RuntimeError(
                "Microphone is not running."
            )

        data, overflowed = self.stream.read(
            self.block_size
        )

        if overflowed:
            self.logger.warning(
                "Microphone input overflow detected."
            )

        data = np.asarray(
            data,
            dtype=np.float32,
        )

        if data.ndim > 1:
            data = data[:, 0]

        return data

    # ---------------------------------------------------------
    # NOISE CALIBRATION
    # ---------------------------------------------------------

    def measure_noise_floor(
        self,
        force: bool = False,
    ) -> float:
        """
        Measure ambient noise in the same processed audio
        domain used by the speech detector.

        Calibrate in a quiet room without speaking.
        """
        if self.noise_floor is not None and not force:
            return self.noise_floor

        if not self.running:
            self.start()

        self.logger.info(
            "Calibrating microphone noise floor..."
        )

        measurements = []

        deadline = (
            time.monotonic()
            + self.NOISE_CALIBRATION_SECONDS
        )

        while time.monotonic() < deadline:
            raw = self.read_block()
            prepared = self.prepare_audio(raw)

            if prepared.size:
                measurements.append(
                    self.calculate_rms(prepared)
                )

        if not measurements:
            self.noise_floor = 30.0
        else:
            # The median limits the effect of occasional
            # louder blocks during calibration.
            self.noise_floor = float(
                np.median(measurements)
            )

        calculated_threshold = (
            self.noise_floor
            * self.ENERGY_MULTIPLIER
        )

        self.speech_threshold = min(
            self.MAX_SPEECH_THRESHOLD,
            max(
                self.MIN_SPEECH_RMS,
                calculated_threshold,
            ),
        )

        self.logger.info(
            "Noise calibration complete | "
            f"noise_floor={self.noise_floor:.2f} | "
            f"speech_threshold={self.speech_threshold:.2f}"
        )

        return self.noise_floor

    def reset_noise_calibration(self) -> None:
        self.noise_floor = None
        self.speech_threshold = None

        self.logger.info(
            "Noise calibration reset"
        )

    # ---------------------------------------------------------
    # VOICE ACTIVITY DETECTION
    # ---------------------------------------------------------

    def _is_speech_frame(
        self,
        frame: np.ndarray,
    ) -> bool:
        prepared = self.prepare_audio(frame)

        if prepared.size == 0:
            return False

        if self.speech_threshold is None:
            self.measure_noise_floor()

        rms = self.calculate_rms(prepared)

        if rms < self.speech_threshold:
            return False

        pcm = (
            np.clip(prepared, -1.0, 1.0)
            * 32767.0
        ).astype(np.int16)

        expected_samples = int(
            self.process_sample_rate
            * self.BLOCK_DURATION_MS
            / 1000
        )

        if len(pcm) != expected_samples:
            return False

        try:
            return bool(
                self.vad.is_speech(
                    pcm.tobytes()
                )
            )

        except (ValueError, TypeError) as exc:
            self.logger.debug(
                f"VAD rejected audio frame: {exc}"
            )
            return False

    # ---------------------------------------------------------
    # SPEECH CAPTURE
    # ---------------------------------------------------------

    def listen_for_speech(
        self,
        timeout_seconds: float = 10.0,
    ) -> Optional[np.ndarray]:
        """
        Wait for speech and return processed mono float32
        audio, or None if speech is not captured.

        The timeout applies to waiting for speech to begin.
        Once speech begins, recording continues until silence
        or MAX_SPEECH_DURATION is reached.
        """
        if not self.running:
            self.start()

        self.measure_noise_floor()

        self.logger.info(
            "Listening for speech..."
        )

        pre_roll = deque(
            maxlen=self.PRE_ROLL_FRAMES
        )

        frames = []
        consecutive_speech = 0
        silence_duration = 0.0
        speech_started = False

        wait_start = time.monotonic()
        recording_start = None

        while True:
            if (
                not speech_started
                and time.monotonic() - wait_start
                >= timeout_seconds
            ):
                self.logger.info(
                    "Speech detection timeout"
                )
                return None

            frame = self.read_block()
            pre_roll.append(frame)

            is_speech = self._is_speech_frame(
                frame
            )

            if not speech_started:
                if is_speech:
                    consecutive_speech += 1
                else:
                    consecutive_speech = 0

                if (
                    consecutive_speech
                    < self.SPEECH_START_FRAMES
                ):
                    continue

                speech_started = True
                recording_start = time.monotonic()

                # Include earlier frames so the start of a
                # word is less likely to be clipped.
                frames.extend(list(pre_roll))

                self.logger.info(
                    "Speech detected | "
                    f"rms={self.calculate_rms(frame):.2f} | "
                    f"threshold={self.speech_threshold:.2f}"
                )

                continue

            frames.append(frame)

            if is_speech:
                silence_duration = 0.0
            else:
                silence_duration += (
                    self.BLOCK_DURATION_MS / 1000.0
                )

            duration = (
                time.monotonic() - recording_start
            )

            if duration >= self.MAX_SPEECH_DURATION:
                self.logger.info(
                    "Maximum speech duration reached"
                )
                break

            if (
                silence_duration
                >= self.SILENCE_DURATION_SECONDS
            ):
                self.logger.info(
                    "Speech ended"
                )
                break

        if not frames:
            return None

        audio = self.prepare_audio(
            np.concatenate(frames)
        )

        duration = (
            len(audio) / self.process_sample_rate
        )

        rms = self.calculate_rms(audio)

        max_amplitude = float(
            np.max(np.abs(audio)) * 32768.0
        )

        self.logger.info(
            "Recorded speech | "
            f"duration={duration:.2f}s | "
            f"max_amplitude={max_amplitude:.0f} | "
            f"rms={rms:.2f}"
        )

        if duration < self.MIN_SPEECH_DURATION:
            self.logger.warning(
                "Recording too short; ignoring"
            )
            return None

        if rms < self.MIN_SPEECH_RMS:
            self.logger.warning(
                "Recording energy too low; ignoring"
            )
            return None

        return audio

    def record_speech(
        self,
        timeout_seconds: float = 10.0,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Compatibility wrapper for VoiceEngine."""
        return self.listen_for_speech(
            timeout_seconds=timeout_seconds
        )

    # ---------------------------------------------------------
    # WAV EXPORT
    # ---------------------------------------------------------

    def audio_to_wav_bytes(
        self,
        audio: np.ndarray,
    ) -> bytes:
        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

        pcm = (
            np.clip(audio, -1.0, 1.0)
            * 32767.0
        ).astype(np.int16)

        buffer = io.BytesIO()

        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(
                self.process_sample_rate
            )
            wav.writeframes(
                pcm.tobytes()
            )

        return buffer.getvalue()

    def save_wav(
        self,
        audio: np.ndarray,
        path: str,
    ) -> str:
        with open(path, "wb") as file:
            file.write(
                self.audio_to_wav_bytes(audio)
            )

        self.logger.info(
            f"Audio saved: {path}"
        )

        return path

    def finalize_recording(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:
        """Prepare audio for downstream speech recognition."""
        return self.prepare_audio(audio)
