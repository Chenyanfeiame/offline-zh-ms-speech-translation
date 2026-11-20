from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from zhms_s2s.audio import AudioData
from zhms_s2s.backends.qwen_asr import Qwen3MlxASR
from zhms_s2s.config import Settings


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def transcribe(self, samples, **kwargs):
        self.calls.append({"samples": samples, **kwargs})
        return SimpleNamespace(text="Pesanan MH123 ke KLIA.")


def test_qwen_context_is_forwarded_to_official_session_api(tmp_path):
    backend = Qwen3MlxASR(tmp_path / "unused")
    session = FakeSession()
    backend._session = session
    audio = AudioData(np.ones(1600, dtype=np.float32), 16000, 0.1)

    result = backend.transcribe_with_context(
        audio, "ms", context="KLIA Malaysia Airlines MH123"
    )

    assert result.text == "Pesanan MH123 ke KLIA."
    assert session.calls[0]["language"] == "Malay"
    assert session.calls[0]["context"] == "KLIA Malaysia Airlines MH123"


def test_qwen_chinese_route_uses_official_chinese_language_name(tmp_path):
    backend = Qwen3MlxASR(tmp_path / "unused")
    session = FakeSession()
    backend._session = session
    audio = AudioData(np.ones(1600, dtype=np.float32), 16000, 0.1)

    backend.transcribe(audio, "zh")

    assert session.calls[0]["language"] == "Chinese"
    assert session.calls[0]["context"] == ""


def test_raw_contextual_qwen_is_the_default_malay_backend():
    assert Settings().asr_ms_backend == "qwen-context"
    assert Settings().entity_prompt_enabled is False
