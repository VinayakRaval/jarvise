import os
import time
import tempfile

from voice.audio_stream import MicrophoneStream
from voice.asr import SpeechRecognizer


def main():
    print("=" * 65)
    print("JARVIS WHISPER BENCHMARK")
    print("=" * 65)
    print("Models: tiny vs base")
    print("Device: CPU | Compute: int8")
    print()

    microphone = MicrophoneStream()
    audio_path = None

    try:
        microphone.start()

        print("Speak one clear English sentence.")
        print("Example: I am building my own JARVIS AI assistant.")
        print()

        audio = microphone.record_speech(
            timeout_seconds=15
        )

        if audio is None:
            print("No valid speech recorded. Try again.")
            return

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as audio_file:
            audio_path = audio_file.name

        microphone.save_wav(
            audio,
            audio_path
        )

    finally:
        microphone.stop()

    results = []

    try:
        for model_size in ("tiny", "base"):
            print()
            print("-" * 65)
            print(f"Testing Whisper: {model_size}")
            print("-" * 65)

            start_load = time.perf_counter()

            try:
                recognizer = SpeechRecognizer(
                    model_size=model_size,
                    device="cpu",
                    compute_type="int8",
                )

                load_time = (
                    time.perf_counter() - start_load
                )

                start_transcription = time.perf_counter()

                result = recognizer.transcribe(
                    audio_path,
                    language="en",
                )

                transcription_time = (
                    time.perf_counter()
                    - start_transcription
                )

                text = result.get("text", "").strip()

                results.append({
                    "model": model_size,
                    "load_time": load_time,
                    "transcription_time": transcription_time,
                    "text": text,
                })

                print(f"Load time: {load_time:.2f}s")
                print(
                    f"Transcription time: "
                    f"{transcription_time:.2f}s"
                )
                print(f"Recognized text: {text!r}")

                del recognizer

            except Exception as exc:
                print(f"Model test failed: {exc}")

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

    print()
    print("=" * 65)
    print("BENCHMARK SUMMARY")
    print("=" * 65)

    if not results:
        print("No models completed successfully.")
        return

    for result in results:
        print(
            f"{result['model']:>5} | "
            f"transcription={result['transcription_time']:.2f}s | "
            f"load={result['load_time']:.2f}s"
        )

    print()
    print("Compare both transcriptions for accuracy.")
    print("Choose the model that gives the best balance")
    print("of transcription quality and speed.")


if __name__ == "__main__":
    main()