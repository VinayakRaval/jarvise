from voice.audio_stream import MicrophoneStream
from voice.asr import SpeechRecognizer


def main():
    print()
    print("=" * 60)
    print("JARVIS VOICE PERCEPTION TEST")
    print("=" * 60)
    print()
    print("Speak naturally after the microphone starts.")
    print("Say a complete sentence.")
    print()

    microphone = MicrophoneStream()

    recognizer = SpeechRecognizer(
        model_size="base",
        device="cpu",
        compute_type="int8",
    )

    try:
        microphone.start()

        print("Microphone is ready.")
        print("Listening...")
        print()

        audio = microphone.record_speech(
            timeout_seconds=15
        )

        if audio is None:
            print(
                "No speech detected."
            )
            return

        output_path = (
            "data/voice/test.wav"
        )

        microphone.save_wav(
            audio,
            output_path,
        )

        print(
            f"Audio saved: {output_path}"
        )

        print()
        print("Transcribing...")

        result = recognizer.transcribe(
            output_path
        )

        print()
        print("-" * 60)
        print(
            "TRANSCRIPTION:"
        )
        print("-" * 60)
        print(
            result["text"]
        )
        print("-" * 60)
        print()

        print(
            "Detected language:",
            result["language"],
        )

        print(
            "Language probability:",
            result["language_probability"],
        )

        print()
        print(
            "VOICE PERCEPTION TEST PASSED"
        )

    except KeyboardInterrupt:
        print()
        print(
            "Voice test stopped."
        )

    finally:
        microphone.stop()


if __name__ == "__main__":
    main()