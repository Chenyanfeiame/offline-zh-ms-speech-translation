from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audio import AudioData
from ..config import validate_language
from ..errors import InferenceError, ModelUnavailableError
from .interfaces import ASRTranscript


QWEN_LANGUAGES = {"zh": "Chinese", "ms": "Malay"}


class Qwen3MlxASR:
    """Persistent raw Qwen3-ASR session with optional context vocabulary."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self._session = None
        self._inference_warmed = False

    def _model_present(self) -> bool:
        return (self.model_path / "config.json").is_file() and (
            self.model_path / "weights.safetensors"
        ).is_file()

    def _load(self) -> None:
        if self._session is not None:
            return
        if not self._model_present():
            raise ModelUnavailableError(
                f"Qwen3 ASR model files are incomplete: {self.model_path}"
            )
        try:
            from mlx_qwen3_asr import Session

            self._session = Session(model=str(self.model_path))
        except Exception as exc:
            raise ModelUnavailableError(
                "Failed to initialize the local MLX Qwen3-ASR session.",
                detail={"path": str(self.model_path), "reason": str(exc)},
            ) from exc

    def transcribe(
        self, audio: AudioData, source_language: str | None
    ) -> ASRTranscript:
        return self.transcribe_with_context(audio, source_language, context="")

    def transcribe_with_context(
        self,
        audio: AudioData,
        source_language: str | None,
        *,
        context: str,
    ) -> ASRTranscript:
        self._load()
        language = validate_language(source_language) if source_language else "ms"
        try:
            result = self._session.transcribe(
                np.asarray(audio.samples, dtype=np.float32),
                language=QWEN_LANGUAGES[language],
                context=context,
            )
            text = str(result.text).strip()
        except Exception as exc:
            raise InferenceError(
                "Qwen3 ASR inference failed.", detail={"reason": str(exc)}
            ) from exc
        if not text:
            raise InferenceError("Qwen3 ASR produced empty text.")
        return ASRTranscript(text, language, None, "manual")

    def warmup(self) -> None:
        self._load()
        if self._inference_warmed:
            return
        try:
            self._session.transcribe(
                np.zeros(16_000, dtype=np.float32),
                language="Malay",
                max_new_tokens=8,
            )
            self._inference_warmed = True
        except Exception as exc:
            raise InferenceError(
                "Qwen3 ASR compute warm-up failed.", detail={"reason": str(exc)}
            ) from exc

    def status(self) -> dict:
        return {
            "backend": "mlx-qwen3-asr",
            "model_path": str(self.model_path),
            "model_present": self._model_present(),
            "loaded": self._session is not None,
            "inference_warmed": self._inference_warmed,
            "device": "mlx-metal",
            "decoding": "deterministic",
            "context_biasing": "official-system-context-vocabulary",
            "post_correction": False,
            "offline": True,
        }
