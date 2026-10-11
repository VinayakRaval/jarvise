import webrtcvad


class VoiceActivityDetector:
    """
    WebRTC Voice Activity Detector.

    Audio:
        16-bit mono PCM

    Supported sample rates:
        8000
        16000
        32000
        48000

    Supported frame durations:
        10 ms
        20 ms
        30 ms
    """

    VALID_SAMPLE_RATES = (
        8000,
        16000,
        32000,
        48000,
    )

    VALID_FRAME_DURATIONS = (
        10,
        20,
        30,
    )

    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 2,
    ):

        if sample_rate not in self.VALID_SAMPLE_RATES:
            raise ValueError(
                "Unsupported sample rate. "
                "Use 8000, 16000, 32000 or 48000."
            )

        if aggressiveness not in (
            0,
            1,
            2,
            3,
        ):
            raise ValueError(
                "VAD aggressiveness must be "
                "between 0 and 3."
            )

        self.sample_rate = sample_rate

        self.aggressiveness = aggressiveness

        self.vad = webrtcvad.Vad(
            aggressiveness
        )

    def is_speech(
        self,
        frame: bytes,
    ) -> bool:
        """
        Determine whether the frame contains speech.

        WebRTC VAD validates the frame length internally.
        """

        if not isinstance(
            frame,
            bytes,
        ):
            raise TypeError(
                "VAD frame must be bytes."
            )

        if not frame:
            raise ValueError(
                "VAD frame is empty."
            )

        return self.vad.is_speech(
            frame,
            self.sample_rate,
        )

    def set_aggressiveness(
        self,
        aggressiveness: int,
    ) -> None:

        if aggressiveness not in (
            0,
            1,
            2,
            3,
        ):
            raise ValueError(
                "VAD aggressiveness must be "
                "between 0 and 3."
            )

        self.aggressiveness = (
            aggressiveness
        )

        self.vad.set_mode(
            aggressiveness
        )