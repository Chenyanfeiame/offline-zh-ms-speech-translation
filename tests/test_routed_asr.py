from __future__ import annotations

import numpy as np

from zhms_s2s.audio import AudioData
from zhms_s2s.backends.interfaces import ASRTranscript
from zhms_s2s.backends.lid import LanguageDecision
from zhms_s2s.backends.routed_asr import RoutedWhisperASR


class FakeLanguageIdentifier:
    def __init__(self, language: str = "ms") -> None:
        self.language = language

    def detect(self, _audio):
        return LanguageDecision(self.language, 0.81, 0.84)

    def warmup(self):
        return None

    def status(self):
        return {"backend": "fake-lid"}


class FakeModel:
    def __init__(
        self,
        language: str,
        translation_text: str | None = None,
    ) -> None:
        self.language = language
        self.translation_text = translation_text
        self.calls: list[str] = []
        self.contexts: list[str] = []
        self.warmup_calls = 0

    def transcribe(self, _audio, source_language):
        self.calls.append(source_language)
        return ASRTranscript(
            text="你好" if self.language == "zh" else "selamat datang",
            language=self.language,
            language_confidence=None,
            language_mode="manual",
            translation_text=self.translation_text,
        )

    def transcribe_with_context(self, audio, source_language, *, context):
        self.contexts.append(context)
        return self.transcribe(audio, source_language)

    def warmup(self):
        self.warmup_calls += 1

    def status(self):
        return {"language": self.language}


def audio():
    return AudioData(np.ones(1600, dtype=np.float32), 16000, 0.1)


def test_automatic_language_routes_and_preserves_lid_confidence():
    zh, ms = FakeModel("zh"), FakeModel("ms", "selamat; datang")
    backend = RoutedWhisperASR(
        {"zh": zh, "ms": ms}, FakeLanguageIdentifier("ms")
    )

    result = backend.transcribe(audio(), None)

    assert result.text == "selamat datang"
    assert result.language == "ms"
    assert result.language_mode == "automatic"
    assert result.language_confidence == 0.81
    assert result.translation_text == "selamat; datang"
    assert ms.calls == ["ms"]
    assert zh.calls == []


def test_manual_language_bypasses_lid_route():
    zh, ms = FakeModel("zh"), FakeModel("ms")
    backend = RoutedWhisperASR(
        {"zh": zh, "ms": ms}, FakeLanguageIdentifier("ms")
    )

    result = backend.transcribe(audio(), "zh")

    assert result.text == "你好"
    assert result.language_mode == "manual"
    assert result.language_confidence is None
    assert zh.calls == ["zh"]
    assert ms.calls == []


def test_shared_model_is_only_warmed_once():
    shared = FakeModel("ms")
    backend = RoutedWhisperASR(
        {"zh": shared, "ms": shared}, FakeLanguageIdentifier("ms")
    )

    backend.warmup()

    assert shared.warmup_calls == 1
    assert backend.status()["model_present"] is True


def test_context_is_sent_to_both_qwen_capable_routes():
    zh, ms = FakeModel("zh"), FakeModel("ms")
    backend = RoutedWhisperASR(
        {"zh": zh, "ms": ms}, FakeLanguageIdentifier("ms")
    )

    backend.transcribe_with_context(audio(), "ms", context="KL Sentral MH123")
    backend.transcribe_with_context(audio(), "zh", context="KL Sentral MH123")

    assert ms.contexts == ["KL Sentral MH123"]
    assert zh.contexts == ["KL Sentral MH123"]
