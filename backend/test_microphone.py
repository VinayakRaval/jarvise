
import wave

import numpy as np
import sounddevice as sd


DEVICE = 1
SAMPLE_RATE = 44100
CHANNELS = 1
DURATION = 5


print("=" * 60)
print("JARVIS RAW MICROPHONE TEST")
print("=" * 60)

device_info = sd.query_devices(DEVICE)

print()
print("Device:")
print(device_info["name"])

print()
print("Recording for 5 seconds...")
print("Speak normally during the recording.")
print()

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="int16",
    device=DEVICE,
)

sd.wait()

print()
print("Recording completed.")

# Convert to NumPy
samples = audio[:, 0]

max_amplitude = int(
    np.max(
        np.abs(samples)
    )
)

rms = float(
    np.sqrt(
        np.mean(
            samples.astype(
                np.float64
            ) ** 2
        )
    )
)

print()
print("Audio statistics:")
print(
    f"Samples:        {len(samples)}"
)
print(
    f"Duration:       {len(samples) / SAMPLE_RATE:.2f} sec"
)
print(
    f"Max amplitude:  {max_amplitude}"
)
print(
    f"RMS:            {rms:.2f}"
)

output = "data/voice/raw_microphone.wav"

with wave.open(
    output,
    "wb",
) as wav:

    wav.setnchannels(
        CHANNELS
    )

    wav.setsampwidth(
        2
    )

    wav.setframerate(
        SAMPLE_RATE
    )

    wav.writeframes(
        samples.tobytes()
    )

print()
print(
    f"Saved: {output}"
)

print()
print("=" * 60)
print("RAW MICROPHONE TEST COMPLETE")
print("=" * 60)
