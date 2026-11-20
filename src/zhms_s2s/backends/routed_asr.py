from __future__ import annotations

from ..audio import AudioData
from ..config import validate_language
from .interfaces import ASRBackend, ASRTranscript
from .lid import CoreMLLanguageIdentifier


class RoutedWhisperASR:
    """Route Chinese and Malay speech to independently selected ASR models."""

    def __init__(
        self,
        models: dict[str, ASRBackend],
        language_identifier: CoreMLLanguageIdentifier,
    ) -> None:
        if set(models) != {"zh", "ms"}:
            raise ValueError("Routed ASR requires exactly zh and ms models")
        self.models = dict(models)
        self.language_identifier = language_identifier

    def transcribe(
        self, audio: AudioData, source_language: str | None
    ) -> ASRTranscript:
        return self.transcribe_with_context(audio, source_language, context="")

    @staticmethod
    def _transcribe_model(
        model: ASRBackend,
        audio: AudioData,
        language: str,
        context: str,
    ) -> ASRTranscript:
        contextual = getattr(model, "transcribe_with_context", None)
        if context and callable(contextual):
            return contextual(audio, language, context=context)
        return model.transcribe(audio, language)

    def transcribe_with_context(
        self,
        audio: AudioData,
        source_language: str | None,
        *,
        context: str,
    ) -> ASRTranscript:
        if source_language:
            language = validate_language(source_language)
            return self._transcribe_model(
                self.models[language], audio, language, context
            )

        decision = self.language_identifier.detect(audio)
        transcript = self._transcribe_model(
            self.models[decision.language], audio, decision.language, context
        )
        return ASRTranscript(
            text=transcript.text,
            language=decision.language,
            language_confidence=decision.confidence,
            language_mode="automatic",
            translation_text=transcript.translation_text,
        )

    def warmup(self) -> None:
        self.language_identifier.warmup()
        warmed: set[int] = set()
        for backend in self.models.values():
            if id(backend) in warmed:
                continue
            warmed.add(id(backend))
            warmup = getattr(backend, "warmup", None)
            if callable(warmup):
                warmup()

    def status(self) -> dict:
        model_status = {
            language: backend.status()
            for language, backend in self.models.items()
        }
        lid_status = self.language_identifier.status()
        return {
            "backend": "routed-whisper",
            "model_present": all(
                status.get("model_present", True)
                for status in model_status.values()
            )
            and lid_status.get("model_present", True),
            "models": model_status,
            "language_id": lid_status,
            "automatic_language_detection": "coreml-open-set-then-zh-ms-route",
        }
