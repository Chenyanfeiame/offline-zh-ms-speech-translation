from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio import AudioData
from ..errors import InferenceError, ModelUnavailableError


SAMPLE_RATE = 16_000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 60
MAX_SAMPLES = SAMPLE_RATE * 30
SUPPORTED_LANGUAGES = ("zh", "ms")


@dataclass(frozen=True, slots=True)
class LanguageDecision:
    language: str
    confidence: float
    supported_probability: float


def _periodic_hamming() -> np.ndarray:
    positions = np.arange(N_FFT, dtype=np.float32)
    return (
        np.float32(0.54)
        - np.float32(0.46)
        * np.cos(np.float32(2.0 * np.pi) * positions / np.float32(N_FFT))
    ).astype(np.float32)


def _filterbank() -> np.ndarray:
    def hz_to_mel(value: float) -> float:
        return 2595.0 * np.log10(1.0 + value / 700.0)

    mel_points = np.linspace(
        hz_to_mel(0.0),
        hz_to_mel(SAMPLE_RATE / 2.0),
        N_MELS + 2,
        dtype=np.float32,
    )
    hz_points = (700.0 * (np.power(10.0, mel_points / 2595.0) - 1.0)).astype(
        np.float32
    )
    centers = hz_points[1:-1]
    bands = (hz_points[1:] - hz_points[:-1])[:-1]
    frequencies = np.linspace(
        0.0, SAMPLE_RATE // 2, N_FFT // 2 + 1, dtype=np.float32
    )
    slopes = (frequencies[:, None] - centers[None, :]) / bands[None, :]
    return np.maximum(0.0, np.minimum(slopes + 1.0, -slopes + 1.0)).astype(
        np.float32
    )


def _speechbrain_features(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio[:MAX_SAMPLES], dtype=np.float32).reshape(-1)
    padded = np.pad(samples, (N_FFT // 2, N_FFT // 2), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[
        ::HOP_LENGTH
    ]
    spectrum = np.fft.rfft(frames * _periodic_hamming()[None, :], axis=-1)
    power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag).astype(
        np.float32
    )
    mel = power @ _filterbank()
    decibels = np.float32(10.0) * np.log10(
        np.maximum(mel, np.float32(1e-10))
    )
    decibels = np.maximum(decibels, np.max(decibels) - np.float32(80.0))
    return (decibels - decibels.mean(axis=0, keepdims=True)).astype(np.float32)


class CoreMLLanguageIdentifier:
    """Open-set speech LID followed by a product-level zh/ms decision."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self._model = None
        self._labels: list[dict] | None = None
        self._indices: dict[str, int] | None = None
        self._inference_warmed = False

    @property
    def compiled_model_path(self) -> Path:
        return self.model_path / "SpeechBrainECAPAVoxLingua107.mlmodelc"

    def _load(self) -> None:
        if self._model is not None:
            return
        labels_path = self.model_path / "labels.json"
        if not self.compiled_model_path.is_dir() or not labels_path.is_file():
            raise ModelUnavailableError(
                f"Language ID model files are incomplete or missing: {self.model_path}"
            )
        try:
            logging.getLogger("coremltools").setLevel(logging.ERROR)
            import coremltools as ct

            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            indices = {item["code"]: index for index, item in enumerate(labels)}
            if any(language not in indices for language in SUPPORTED_LANGUAGES):
                raise ValueError("Language ID labels do not contain zh and ms")
            self._model = ct.models.CompiledMLModel(
                str(self.compiled_model_path.resolve())
            )
            self._labels = labels
            self._indices = indices
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                "Failed to load the local Core ML language ID model.",
                detail={"path": str(self.model_path), "reason": str(exc)},
            ) from exc

    def detect(self, audio: AudioData) -> LanguageDecision:
        self._load()
        try:
            features = _speechbrain_features(audio.samples)[None, :, :]
            output = self._model.predict({"mel_features": features})
            logits = np.asarray(output["log_probabilities"], dtype=np.float32).reshape(
                -1
            )
            probabilities = np.exp(logits - float(np.max(logits)))
            probabilities /= float(np.sum(probabilities))
            zh_probability = float(probabilities[self._indices["zh"]])
            ms_probability = float(probabilities[self._indices["ms"]])
        except Exception as exc:
            raise InferenceError(
                "Audio language identification failed.",
                detail={"reason": str(exc)},
            ) from exc

        language = "zh" if zh_probability >= ms_probability else "ms"
        # The product decision is explicitly constrained to zh/ms. Showing
        # the winning probability out of all 107 VoxLingua labels made a
        # correct route look like "0.3% confidence". Report the conditional
        # confidence for the actual two-way decision, while retaining the
        # open-set mass separately for diagnostics.
        supported_probability = zh_probability + ms_probability
        confidence = (
            max(zh_probability, ms_probability) / supported_probability
            if supported_probability > 0
            else 0.5
        )
        return LanguageDecision(
            language=language,
            confidence=confidence,
            supported_probability=supported_probability,
        )

    def warmup(self) -> None:
        self._load()
        if self._inference_warmed:
            return
        duration = 0.25
        time_axis = np.arange(int(SAMPLE_RATE * duration), dtype=np.float32)
        samples = np.float32(0.01) * np.sin(
            np.float32(2.0 * np.pi * 220.0) * time_axis / np.float32(SAMPLE_RATE)
        )
        self.detect(AudioData(samples, SAMPLE_RATE, duration))
        self._inference_warmed = True

    def status(self) -> dict:
        return {
            "backend": "coreml-voxlingua107-ecapa",
            "model_path": str(self.model_path),
            "model_present": self.compiled_model_path.is_dir(),
            "loaded": self._model is not None,
            "inference_warmed": self._inference_warmed,
            "supported_languages": list(SUPPORTED_LANGUAGES),
            "confidence": "conditional probability within zh/ms",
            "open_set_diagnostic": (
                "supported_probability retains zh+ms mass among 107 labels"
            ),
        }
