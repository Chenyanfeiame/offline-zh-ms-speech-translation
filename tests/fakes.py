from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from zhms_s2s.backends.interfaces import ASRTranscript


class FakeASR:
    def transcribe(self, _audio, source_language):
        language = source_language or "zh"
        return ASRTranscript(
            text="你好" if language == "zh" else "selamat datang",
            language=language,
            language_confidence=None if source_language else 0.95,
            language_mode="manual" if source_language else "automatic",
        )

    def status(self):
        return {"backend": "fake", "model_present": True}


class FakeTranslator:
    def translate(self, text, source_language, target_language):
        if source_language == "zh":
            return "selamat datang", None
        return "欢迎", "welcome"

    def status(self):
        return {
            "backend": "fake",
            "models": {"fake": {"present": True}},
        }


class FakeTTS:
    def synthesize(self, _text, _language, output_path: Path, _voice_id=None):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        samples = np.sin(2 * np.pi * 220 * np.arange(1600) / 16000).astype(np.float32) * 0.1
        sf.write(output_path, samples, 16000, format="WAV")
        return 0.1

    def status(self):
        return {
            "backend": "fake",
            "command_present": True,
            "voices": {"zh": {"available": True}, "ms": {"available": True}},
        }
