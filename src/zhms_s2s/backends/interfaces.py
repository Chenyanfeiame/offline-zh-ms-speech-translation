from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Protocol

from ..audio import AudioData


@dataclass(frozen=True, slots=True)
class ASRTranscript:
    text: str
    language: str
    language_confidence: float | None
    language_mode: str
    # Optional punctuation/segmentation hint used only by MT.  The visible ASR
    # transcript stays untouched so recognition quality and UI semantics do not
    # change when we add sentence-boundary information for translation.
    translation_text: str | None = None


class ASRBackend(Protocol):
    def transcribe(self, audio: AudioData, source_language: str | None) -> ASRTranscript: ...

    def status(self) -> dict: ...


class TranslationBackend(Protocol):
    def translate(self, text: str, source_language: str, target_language: str) -> tuple[str, str | None]: ...

    def status(self) -> dict: ...


class TTSBackend(Protocol):
    def synthesize(
        self,
        text: str,
        language: str,
        output_path: Path,
        voice_id: str | None = None,
    ) -> float: ...

    def status(self) -> dict: ...
