from voice.tts import TextToSpeech


def main():

    print()
    print("=" * 60)
    print("JARVIS TEXT-TO-SPEECH TEST")
    print("=" * 60)
    print()

    tts = TextToSpeech(
        rate=175,
        volume=1.0,
    )

    if not tts.is_available():

        print("TTS initialization failed.")
        return

    print("Speaking...")
    print()

    success = tts.speak(
        "Hello Vinayak. "
        "I am JARVIS. "
        "My voice system is now working locally."
    )

    print()

    if success:

        print("TTS TEST PASSED")

    else:

        print("TTS TEST FAILED")


if __name__ == "__main__":
    main()