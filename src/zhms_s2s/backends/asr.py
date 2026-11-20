from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..audio import AudioData
from ..config import validate_language
from ..errors import InferenceError, ModelUnavailableError
from .interfaces import ASRBackend, ASRTranscript


WHISPER_LANGUAGES = {"zh": "chinese", "ms": "malay"}


@lru_cache(maxsize=1)
def _chinese_converter():
    try:
        from opencc import OpenCC
    except ImportError as exc:
        raise ModelUnavailableError(
            "OpenCC is required to normalize Chinese ASR output to Simplified Chinese."
        ) from exc
    return OpenCC("t2s")


def normalize_asr_text(text: str, language: str) -> str:
    """Keep Malay untouched and normalize Chinese orthography to Simplified."""
    return _chinese_converter().convert(text) if language == "zh" else text


def _model_present(path: Path) -> bool:
    return (path / "config.json").is_file() and any(
        (path / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin", "model.bin")
    )


def _find_pause_boundaries(
    audio: AudioData,
    *,
    min_silence_ms: int = 380,
    min_segment_s: float = 1.2,
    max_segments: int = 4,
) -> list[int]:
    """Return sample offsets for clear, non-edge pauses.

    The threshold is derived from each recording's own noise and speech energy,
    which is more stable for phone recordings than one fixed amplitude cutoff.
    Only long interior pauses are used, so ordinary within-word dips cannot
    fragment an utterance.
    """
    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.size == 0 or audio.duration_s < min_segment_s * 2:
        return []
    frame_size = max(1, int(audio.sample_rate * 0.02))
    padding = (-samples.size) % frame_size
    framed = np.pad(samples, (0, padding)).reshape(-1, frame_size)
    rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
    noise_floor = float(np.quantile(rms, 0.20))
    speech_level = float(np.quantile(rms, 0.80))
    if speech_level <= 1e-5:
        return []
    threshold = max(noise_floor * 3.0, speech_level * 0.06, 1e-4)
    threshold = min(threshold, speech_level * 0.35)
    silent = rms < threshold
    minimum_frames = max(1, int(np.ceil(min_silence_ms / 20)))
    minimum_samples = int(min_segment_s * audio.sample_rate)

    candidates: list[int] = []
    start: int | None = None
    for index, is_silent in enumerate(silent):
        if is_silent and start is None:
            start = index
        at_end = index == len(silent) - 1
        if start is not None and (not is_silent or at_end):
            end = index if not is_silent else index + 1
            if end - start >= minimum_frames:
                boundary = min(
                    samples.size,
                    int(((start + end) / 2) * frame_size),
                )
                if minimum_samples <= boundary <= samples.size - minimum_samples:
                    candidates.append(boundary)
            start = None

    boundaries: list[int] = []
    previous = 0
    for boundary in candidates:
        if boundary - previous < minimum_samples:
            continue
        if samples.size - boundary < minimum_samples:
            continue
        boundaries.append(boundary)
        previous = boundary
        if len(boundaries) >= max(0, max_segments - 1):
            break
    return boundaries


_CHINESE_TAIL_BOUNDARY = re.compile(
    r"^(?:他|她|他们|她们|然后|但是|不过|所以|后来|另外|接着|随后|请|不要|而且|并且)"
)


def _mlx_translation_hint(
    text: str,
    language: str,
    segments: list[dict] | None,
    pause_boundaries_s: list[float],
) -> str | None:
    """Build an MT-only clause hint without changing the displayed ASR text."""
    if language != "zh" or re.search(r"[。！？.!?]", text):
        return None
    chunks = [
        (str(segment.get("text", "")).strip(), float(segment.get("start", 0.0)))
        for segment in (segments or [])
        if str(segment.get("text", "")).strip()
    ]
    if len(chunks) < 2:
        return None

    hinted = chunks[0][0]
    inserted_boundary = False
    for chunk, start_s in chunks[1:]:
        clear_pause = any(abs(start_s - pause) <= 0.45 for pause in pause_boundaries_s)
        clause_tail = bool(_CHINESE_TAIL_BOUNDARY.match(chunk))
        if clear_pause or clause_tail:
            hinted += "；"
            inserted_boundary = True
        hinted += chunk
    return hinted if inserted_boundary and hinted != text else None


class PauseSegmentedASR:
    """Run one ASR backend per clear spoken sentence and join the transcript.

    This wrapper is intentionally used only for the Malay route.  It recovers
    beam search quality on natural pauses and gives MT explicit sentence
    boundaries, while leaving the Chinese recognizer and its visible text
    unchanged.
    """

    def __init__(
        self,
        backend: ASRBackend,
        *,
        language: str,
        min_silence_ms: int = 380,
        min_segment_s: float = 1.2,
        max_segments: int = 4,
        segmented_num_beams: int = 1,
    ) -> None:
        self.backend = backend
        self.language = validate_language(language)
        self.min_silence_ms = max(200, int(min_silence_ms))
        self.min_segment_s = max(0.8, float(min_segment_s))
        self.max_segments = max(1, int(max_segments))
        self.segmented_num_beams = max(1, int(segmented_num_beams))

    @staticmethod
    def _with_terminal_punctuation(text: str, language: str) -> str:
        clean = text.strip()
        if not clean or re.search(r"[.!?。！？]$", clean):
            return clean
        return clean + ("。" if language == "zh" else ".")

    def transcribe(
        self, audio: AudioData, source_language: str | None
    ) -> ASRTranscript:
        language = (
            validate_language(source_language) if source_language else self.language
        )
        boundaries = _find_pause_boundaries(
            audio,
            min_silence_ms=self.min_silence_ms,
            min_segment_s=self.min_segment_s,
            max_segments=self.max_segments,
        )
        if not boundaries:
            return self.backend.transcribe(audio, language)

        segment_audios: list[AudioData] = []
        offsets = [0, *boundaries, len(audio.samples)]
        for start, end in zip(offsets, offsets[1:]):
            segment_audios.append(
                AudioData(
                    samples=np.asarray(audio.samples[start:end], dtype=np.float32),
                    sample_rate=audio.sample_rate,
                    duration_s=(end - start) / audio.sample_rate,
                )
            )
        transcribe_batch = getattr(self.backend, "transcribe_batch", None)
        if callable(transcribe_batch):
            transcripts = transcribe_batch(
                segment_audios,
                language,
                num_beams=self.segmented_num_beams,
            )
        else:
            transcripts = [
                self.backend.transcribe(segment, language)
                for segment in segment_audios
            ]
        texts = [
            self._with_terminal_punctuation(transcript.text, language)
            for transcript in transcripts
        ]
        combined = " ".join(text for text in texts if text).strip()
        if not combined:
            raise InferenceError("Pause-segmented ASR produced empty text.")
        return ASRTranscript(
            text=combined,
            language=language,
            language_confidence=None,
            language_mode="manual",
            translation_text=combined,
        )

    def warmup(self) -> None:
        warmup = getattr(self.backend, "warmup", None)
        if callable(warmup):
            warmup()

    def status(self) -> dict:
        base = self.backend.status()
        return {
            "backend": "pause-segmented-asr",
            "model_present": base.get("model_present", True),
            "base_backend": base,
            "language": self.language,
            "min_silence_ms": self.min_silence_ms,
            "min_segment_s": self.min_segment_s,
            "max_segments": self.max_segments,
            "segmented_num_beams": self.segmented_num_beams,
            "sentence_strategy": "adaptive-energy-pause-split",
        }


class TransformersWhisperASR:
    """Lazy, local-only Hugging Face Whisper backend."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cpu",
        offline: bool = True,
        num_beams: int = 1,
        prompts: dict[str, str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.requested_device = device
        self.offline = offline
        self.num_beams = max(1, int(num_beams))
        self.prompts = dict(prompts or {})
        self._processor = None
        self._model = None
        self._torch = None
        self._device = "cpu"
        self._inference_warmed = False

    def _load(self) -> None:
        if self._model is not None:
            return
        if not _model_present(self.model_path):
            raise ModelUnavailableError(
                f"ASR model files are incomplete or missing: {self.model_path}"
            )
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            device = self.requested_device
            if device == "auto":
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            if device == "mps" and not torch.backends.mps.is_available():
                device = "cpu"
            self._device = device
            self._processor = AutoProcessor.from_pretrained(
                self.model_path, local_files_only=self.offline
            )
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_path,
                local_files_only=self.offline,
                low_cpu_mem_usage=False,
            ).to(device)
            self._model.eval()
            self._torch = torch
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                "Failed to load local Whisper model.",
                detail={"path": str(self.model_path), "reason": str(exc)},
            ) from exc

    def _detect_supported_language(self, features) -> tuple[str, float]:
        """Use Whisper's audio logits, constrained to this product's two languages."""
        generation_config = self._model.generation_config
        decoder_ids = self._torch.full(
            (features.shape[0], 1),
            generation_config.decoder_start_token_id,
            device=self._device,
            dtype=self._torch.long,
        )
        with self._torch.inference_mode():
            logits = self._model(
                input_features=features[:, :, :3000],
                decoder_input_ids=decoder_ids,
                use_cache=False,
            ).logits[:, -1]
        language_ids = self._torch.tensor(
            [
                generation_config.lang_to_id["<|zh|>"],
                generation_config.lang_to_id["<|ms|>"],
            ],
            device=self._device,
        )
        probabilities = self._torch.softmax(
            logits.index_select(-1, language_ids).float(), dim=-1
        )[0]
        selected = int(probabilities.argmax().item())
        return ("zh", "ms")[selected], float(probabilities[selected].item())

    def _generation_options(
        self,
        language: str,
        *,
        num_beams: int | None = None,
        prompt_override: str | None = None,
    ) -> dict:
        generation = {
            "task": "transcribe",
            "language": WHISPER_LANGUAGES[language],
            "max_new_tokens": 128,
            "num_beams": self.num_beams if num_beams is None else max(1, num_beams),
            # Short utterances occasionally enter a repeated-token loop on
            # tiny/base Whisper. Bound it before it dominates WER and latency.
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.05,
        }
        prompt = (
            self.prompts.get(language, "")
            if prompt_override is None
            else prompt_override
        ).strip()
        if prompt:
            generation["prompt_ids"] = self._processor.get_prompt_ids(
                prompt, return_tensors="pt"
            ).to(self._device)
        return generation

    def transcribe_batch(
        self,
        audios: list[AudioData],
        source_language: str,
        *,
        num_beams: int | None = None,
    ) -> list[ASRTranscript]:
        """Decode pause-split segments in one model call.

        All items come from one recording and share a forced language.  Batching
        avoids running the encoder and beam-search scheduler once per sentence.
        """
        if not audios:
            return []
        self._load()
        language = validate_language(source_language)
        if any(audio.sample_rate != audios[0].sample_rate for audio in audios):
            raise InferenceError("Batched ASR segments must share a sample rate.")
        try:
            inputs = self._processor(
                [audio.samples for audio in audios],
                sampling_rate=audios[0].sample_rate,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
            )
            features = inputs.input_features.to(self._device)
            attention_mask = inputs.attention_mask.to(self._device)
            with self._torch.inference_mode():
                predicted_ids = self._model.generate(
                    features,
                    attention_mask=attention_mask,
                    **self._generation_options(language, num_beams=num_beams),
                )
            texts = self._processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise InferenceError(
                "Batched ASR inference failed.", detail={"reason": str(exc)}
            ) from exc
        transcripts = []
        for text in texts:
            clean = normalize_asr_text(text.strip(), language)
            if not clean:
                raise InferenceError("Batched ASR produced empty text.")
            transcripts.append(
                ASRTranscript(clean, language, None, "manual")
            )
        return transcripts

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
        language = validate_language(source_language) if source_language else None
        try:
            inputs = self._processor(
                audio.samples,
                sampling_rate=audio.sample_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )
            features = inputs.input_features.to(self._device)
            attention_mask = inputs.attention_mask.to(self._device)
            confidence = None
            language_mode = "manual"
            if language is None:
                language, confidence = self._detect_supported_language(features)
                language_mode = "automatic"
            with self._torch.inference_mode():
                predicted_ids = self._model.generate(
                    features,
                    attention_mask=attention_mask,
                    **self._generation_options(
                        language,
                        prompt_override=context or None,
                    ),
                )
            text = self._processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0].strip()
            text = normalize_asr_text(text, language)
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise InferenceError(
                "ASR inference failed.", detail={"reason": str(exc)}
            ) from exc
        if not text:
            raise InferenceError("ASR produced empty text.")
        return ASRTranscript(
            text=text,
            language=language,
            language_confidence=confidence,
            language_mode=language_mode,
        )

    def warmup(self) -> None:
        self._load()
        if self._inference_warmed:
            return
        try:
            # Loading weights is not enough on MPS: the first real generate()
            # otherwise pays kernel/graph initialization. Exercise both forced
            # language prefixes with the same 30 s feature shape as requests.
            silence = np.zeros(16_000, dtype=np.float32)
            inputs = self._processor(
                silence,
                sampling_rate=16_000,
                return_tensors="pt",
                return_attention_mask=True,
            )
            features = inputs.input_features.to(self._device)
            attention_mask = inputs.attention_mask.to(self._device)
            for language in ("zh", "ms"):
                with self._torch.inference_mode():
                    self._model.generate(
                        features,
                        attention_mask=attention_mask,
                        task="transcribe",
                        language=WHISPER_LANGUAGES[language],
                        max_new_tokens=4,
                        num_beams=self.num_beams,
                    )
            if self._device == "mps":
                self._torch.mps.synchronize()
            self._inference_warmed = True
        except Exception as exc:
            raise InferenceError(
                "ASR compute warm-up failed.", detail={"reason": str(exc)}
            ) from exc

    def status(self) -> dict:
        return {
            "backend": "transformers-whisper",
            "model_path": str(self.model_path),
            "model_present": _model_present(self.model_path),
            "loaded": self._model is not None,
            "inference_warmed": self._inference_warmed,
            "device": self._device if self._model is not None else self.requested_device,
            "offline": self.offline,
            "num_beams": self.num_beams,
            "prompt_languages": sorted(
                language for language, prompt in self.prompts.items() if prompt.strip()
            ),
            "context_biasing": "per-request-prompt-ids",
            "automatic_language_detection": "whisper-audio-logits-zh-ms",
        }


class MlxWhisperASR:
    """Persistent MLX Whisper backend for native Apple Silicon inference."""

    def __init__(self, model_path: Path, *, offline: bool = True) -> None:
        self.model_path = Path(model_path)
        self.offline = offline
        self._mlx_whisper = None
        self._loaded = False
        self._inference_warmed = False

    def _model_present(self) -> bool:
        return (self.model_path / "config.json").is_file() and any(
            (self.model_path / name).is_file()
            for name in ("weights.safetensors", "model.safetensors", "weights.npz")
        )

    def _load(self) -> None:
        if self._mlx_whisper is not None:
            return
        if not self._model_present():
            raise ModelUnavailableError(
                f"MLX ASR model files are incomplete or missing: {self.model_path}"
            )
        try:
            import mlx_whisper

            self._mlx_whisper = mlx_whisper
        except Exception as exc:
            raise ModelUnavailableError(
                "Failed to initialize mlx-whisper on Apple Silicon.",
                detail={"path": str(self.model_path), "reason": str(exc)},
            ) from exc

    def _infer_result(self, samples: np.ndarray, language: str) -> dict:
        result = self._mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=str(self.model_path),
            language=language,
            task="transcribe",
            temperature=0.0,
            beam_size=None,
            condition_on_previous_text=False,
            verbose=None,
        )
        self._loaded = True
        return result

    def _infer(self, samples: np.ndarray, language: str) -> str:
        return str(self._infer_result(samples, language).get("text", "")).strip()

    def transcribe(
        self, audio: AudioData, source_language: str | None
    ) -> ASRTranscript:
        self._load()
        language = validate_language(source_language) if source_language else None
        try:
            if language is None:
                # Production uses CoreML LID before this backend. Keep direct
                # MLX detection for isolated use and diagnostics.
                result = self._mlx_whisper.transcribe(
                    audio.samples,
                    path_or_hf_repo=str(self.model_path),
                    task="transcribe",
                    temperature=0.0,
                    beam_size=None,
                    condition_on_previous_text=False,
                    verbose=None,
                )
                language = validate_language(str(result.get("language", "")))
                text = str(result.get("text", "")).strip()
                language_mode = "automatic"
                self._loaded = True
            else:
                result = self._infer_result(audio.samples, language)
                text = str(result.get("text", "")).strip()
                language_mode = "manual"
            text = normalize_asr_text(text, language)
            pause_boundaries_s = [
                boundary / audio.sample_rate
                for boundary in _find_pause_boundaries(
                    audio,
                    min_silence_ms=300,
                    min_segment_s=0.8,
                    max_segments=8,
                )
            ]
            translation_text = _mlx_translation_hint(
                text,
                language,
                result.get("segments"),
                pause_boundaries_s,
            )
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise InferenceError(
                "MLX ASR inference failed.", detail={"reason": str(exc)}
            ) from exc
        if not text:
            raise InferenceError("ASR produced empty text.")
        return ASRTranscript(
            text=text,
            language=language,
            language_confidence=None,
            language_mode=language_mode,
            translation_text=translation_text,
        )

    def warmup(self) -> None:
        self._load()
        if self._inference_warmed:
            return
        try:
            silence = np.zeros(16_000, dtype=np.float32)
            for language in ("zh", "ms"):
                self._infer(silence, language)
            self._inference_warmed = True
        except Exception as exc:
            raise InferenceError(
                "MLX ASR compute warm-up failed.", detail={"reason": str(exc)}
            ) from exc

    def status(self) -> dict:
        return {
            "backend": "mlx-whisper",
            "model_path": str(self.model_path),
            "model_present": self._model_present(),
            "loaded": self._loaded,
            "inference_warmed": self._inference_warmed,
            "device": "mlx-metal",
            "offline": self.offline,
            "decoding": "greedy-temperature-0",
            "translation_boundary_strategy": "pause-aware-hidden-segment-hints",
            "automatic_language_detection": "external-coreml-router",
        }
