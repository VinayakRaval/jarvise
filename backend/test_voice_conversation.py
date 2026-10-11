from core.jarvis import Jarvis
from voice.voice_engine import VoiceEngine


def main():

    print()
    print("=" * 70)
    print("JARVIS VOICE CONVERSATION TEST")
    print("=" * 70)
    print()

    print("Initializing JARVIS...")

    jarvis = Jarvis(
        session_id="voice-test"
    )

    print("Initializing Voice Engine...")

    voice = VoiceEngine(
        jarvis=jarvis,
        whisper_model="base",
        tts_enabled=True,
    )

    print()
    print("JARVIS is ready.")
    print()
    print("Speak naturally.")
    print("JARVIS will listen, think, and respond.")
    print()
    print("Press Ctrl+C to stop.")
    print()

    try:

        while True:

            print("-" * 70)
            print("Listening...")
            print("-" * 70)

            result = voice.run_once(
                timeout_seconds=10
            )

            if not result.get(
                "speech_detected",
                False,
            ):

                print(
                    "No speech detected."
                )

                continue

            text = result.get(
                "text",
                "",
            )

            response = result.get(
                "response",
                "",
            )

            print()
            print(
                f"You: {text}"
            )

            print(
                f"JARVIS: {response}"
            )

            print(
                f"Spoken: "
                f"{result.get('spoken')}"
            )

            print()

    except KeyboardInterrupt:

        print()
        print(
            "Stopping JARVIS voice..."
        )

    finally:

        voice.stop()

        print(
            "Voice engine stopped."
        )


if __name__ == "__main__":
    main()