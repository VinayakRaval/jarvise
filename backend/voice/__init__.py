"""
JARVIS Voice Perception subsystem.

Voice pipeline:

    Microphone
        ↓
    Audio Stream
        ↓
    Voice Activity Detection
        ↓
    Speech Recognition
        ↓
    Natural language text
        ↓
    JARVIS Brain
"""

from voice.voice_engine import VoiceEngine

__all__ = [
    "VoiceEngine",
]