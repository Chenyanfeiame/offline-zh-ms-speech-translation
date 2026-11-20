from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from ..config import target_for, validate_language
from ..errors import InferenceError, InvalidLanguageError, ModelUnavailableError


NLLB_LANGUAGES = {"zh": "zho_Hans", "ms": "zsm_Latn"}

_MALAY_PREFIX_ABBREVIATIONS = {
    "dr.",
    "prof.",
    "mr.",
    "mrs.",
    "pn.",
    "en.",
    "no.",
    "sdn.",
}

_MALAY_CONTEXT_ABBREVIATIONS = {
    "bhd.",
    "dll.",
    "dsb.",
    "spt.",
    "hlm.",
}

_DIRECT_TERMINOLOGY = {
    ("ms", "zh"): (
        ("Johor Bahru", "新山"),
        ("Pulau Pinang", "槟城"),
        ("Kuala Lumpur", "吉隆坡"),
        ("KL Sentral", "吉隆坡中央车站"),
        ("terjemahan suara", "语音翻译"),
        ("luar talian", "离线"),
        ("Penang", "槟城"),
        ("Johor", "柔佛"),
    ),
    ("zh", "ms"): (
        ("吉隆坡中央车站", "KL Sentral"),
        ("吉隆坡", "Kuala Lumpur"),
        ("语音翻译", "terjemahan suara"),
        ("离线", "luar talian"),
        ("槟城", "Pulau Pinang"),
        ("柔佛", "Johor"),
        ("新山", "Johor Bahru"),
    ),
}


def _domain_translation_template(
    text: str, source: str, target: str
) -> str | None:
    """Deterministic translations for audited critical-intent sentences.

    Identifiers and payment negation are demo-critical slots.  Keeping these
    two narrow templates outside the statistical decoder prevents a small NLLB
    model from paraphrasing away a code or reversing a safety instruction.
    """
    clean = text.strip()
    if source == "ms" and target == "zh":
        order = re.fullmatch(
            r"Nombor pesanan saya (?:ialah|adalah)\s+([A-Za-z0-9][A-Za-z0-9-]*)[.!?]?",
            clean,
            flags=re.IGNORECASE,
        )
        if order:
            return f"我的订单号是{order.group(1)}。"
        if re.fullmatch(
            r"Jangan (?:buat|lakukan) bayaran (?:dua|2) kali[.!?]?",
            clean,
            flags=re.IGNORECASE,
        ):
            return "请勿重复付款。"
    if source == "zh" and target == "ms":
        order = re.fullmatch(
            r"(?:我的)?订单号(?:是|为)\s*([A-Za-z0-9][A-Za-z0-9-]*)[。！？.!?]?",
            clean,
        )
        if order:
            return f"Nombor pesanan saya ialah {order.group(1)}."
        if re.fullmatch(
            r"(?:请勿|不要)(?:重复付款|支付两次)[。！？.!?]?",
            clean,
        ):
            return "Jangan buat bayaran dua kali."
    return None


def _split_translation_segments(text: str) -> list[str]:
    """Split spoken multi-sentence text while protecting common abbreviations.

    A boundary is only considered when terminal punctuation is followed by
    whitespace, so decimal values such as ``RM12.50`` remain intact.
    """
    clean = text.strip()
    if not clean:
        return []
    pieces = re.split(r"(?<=[.!?])\s+|(?<=[。！？])\s*", clean)
    segments: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if segments:
            last_token = segments[-1].rsplit(maxsplit=1)[-1].lower()
            is_initial = bool(re.fullmatch(r"[a-z]\.", last_token))
            continues_abbreviation = (
                last_token in _MALAY_PREFIX_ABBREVIATIONS
                or (
                    last_token in _MALAY_CONTEXT_ABBREVIATIONS
                    and piece[:1].islower()
                )
            )
            if continues_abbreviation or is_initial:
                segments[-1] = f"{segments[-1]} {piece}"
                continue
        segments.append(piece)
    return segments


def _protect_terminology(
    text: str, source: str, target: str
) -> tuple[str, dict[str, str]]:
    """Protect a small audited place-name glossary through NLLB decoding."""
    protected = text
    replacements: dict[str, str] = {}
    for source_term, target_term in _DIRECT_TERMINOLOGY.get((source, target), ()):
        if source == "ms":
            pattern = re.compile(
                rf"(?<![A-Za-z]){re.escape(source_term)}(?![A-Za-z])",
                flags=re.IGNORECASE,
            )
            if not pattern.search(protected):
                continue
        elif source_term not in protected:
            continue
        placeholder = f"XQ{len(replacements) + 1:03d}X"
        while placeholder in protected:
            placeholder = f"XQ{len(replacements) + 101:03d}X"
        if source == "ms":
            protected = pattern.sub(placeholder, protected)
        else:
            protected = protected.replace(source_term, placeholder)
        replacements[placeholder] = target_term
    return protected, replacements


def _restore_terminology(text: str, replacements: dict[str, str]) -> str:
    restored = text
    for placeholder, target_term in replacements.items():
        restored = re.sub(
            re.escape(placeholder), target_term, restored, flags=re.IGNORECASE
        )
    return restored


def _model_present(path: Path) -> bool:
    return (path / "model" / "model.bin").is_file() and (
        path / "sentencepiece.model"
    ).is_file()


class ArgosPivotTranslator:
    """Chinese/Malay translator using official Argos CTranslate2 packages and an English pivot."""

    def __init__(
        self,
        zh_en_path: Path,
        en_ms_path: Path,
        ms_en_path: Path,
        en_zh_path: Path,
        *,
        device: str = "cpu",
        offline: bool = True,
        isolate_process: bool = True,
    ) -> None:
        self.paths = {
            "zh-en": Path(zh_en_path),
            "en-ms": Path(en_ms_path),
            "ms-en": Path(ms_en_path),
            "en-zh": Path(en_zh_path),
        }
        self.requested_device = device
        self.offline = offline
        self.isolate_process = isolate_process
        self._loaded: dict[str, tuple[object, object]] = {}
        self._device = "cpu"

    def _resolve_device(self) -> str:
        device = self.requested_device
        if device == "auto":
            device = "cpu"
        if device == "mps":
            device = "cpu"
        return device

    def _load(self, key: str):
        if key in self._loaded:
            return self._loaded[key]
        path = self.paths[key]
        if not _model_present(path):
            raise ModelUnavailableError(f"MT model files are incomplete or missing: {path}")
        try:
            import ctranslate2
            import sentencepiece as spm

            self._device = self._resolve_device()
            translator = ctranslate2.Translator(
                str(path / "model"),
                device="cpu",
                compute_type="int8",
                inter_threads=1,
                # Keep the fallback deterministic and avoid oversubscribing the
                # small demo host when PyTorch is loaded in the same process.
                intra_threads=1,
            )
            tokenizer = spm.SentencePieceProcessor(
                model_file=str(path / "sentencepiece.model")
            )
            self._loaded[key] = (translator, tokenizer)
            return translator, tokenizer
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                "Failed to load local translation model.",
                detail={"path": str(path), "reason": str(exc)},
            ) from exc

    def _generate(self, key: str, text: str) -> str:
        translator, tokenizer = self._load(key)
        try:
            source_tokens = tokenizer.encode(text, out_type=str)
            result = translator.translate_batch(
                [source_tokens],
                beam_size=4,
                max_decoding_length=256,
            )[0]
            translated = "".join(result.hypotheses[0]).replace("▁", " ").strip()
        except Exception as exc:
            raise InferenceError(
                "Translation inference failed.",
                detail={"stage": key, "reason": str(exc)},
            ) from exc
        if not translated:
            raise InferenceError("Translation produced empty text.", detail={"stage": key})
        return translated

    def _translate_local(
        self, text: str, source_language: str, target_language: str
    ) -> tuple[str, str | None]:
        source = validate_language(source_language)
        target = validate_language(target_language)
        if source == target:
            raise InvalidLanguageError("Source and target languages must differ.")
        if target != target_for(source):
            raise InvalidLanguageError("Only zh<->ms translation is supported.")
        clean = text.strip()
        if not clean:
            raise InferenceError("Translation input text is empty.")
        if source == "zh":
            pivot = self._generate("zh-en", clean)
            return self._generate("en-ms", pivot), pivot
        pivot = self._generate("ms-en", clean)
        translated = self._generate("en-zh", pivot)
        return translated, pivot

    def _translate_isolated(
        self, text: str, source_language: str, target_language: str
    ) -> tuple[str, str | None]:
        payload = {
            "paths": {key: str(path) for key, path in self.paths.items()},
            "device": self.requested_device,
            "offline": self.offline,
            "text": text,
            "source_language": source_language,
            "target_language": target_language,
        }
        try:
            process = subprocess.run(
                [sys.executable, "-m", "zhms_s2s.mt_worker"],
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except Exception as exc:
            raise InferenceError(
                "Isolated translation process failed.",
                detail={"reason": str(exc)},
            ) from exc
        marker = "ZHMS_RESULT="
        line = next(
            (item for item in reversed(process.stdout.splitlines()) if item.startswith(marker)),
            None,
        )
        if process.returncode != 0 or line is None:
            raise InferenceError(
                "Isolated translation process returned an error.",
                detail={
                    "returncode": process.returncode,
                    "stderr": process.stderr[-2000:].strip(),
                    "stdout": process.stdout[-2000:].strip(),
                },
            )
        result = json.loads(line.removeprefix(marker))
        if not result.get("ok"):
            raise InferenceError(
                "Translation inference failed.", detail=result.get("error") or {}
            )
        return result["translated_text"], result.get("pivot_text")

    def translate(
        self, text: str, source_language: str, target_language: str
    ) -> tuple[str, str | None]:
        source = validate_language(source_language)
        target = validate_language(target_language)
        if source == target:
            raise InvalidLanguageError("Source and target languages must differ.")
        if target != target_for(source):
            raise InvalidLanguageError("Only zh<->ms translation is supported.")
        clean = text.strip()
        if not clean:
            raise InferenceError("Translation input text is empty.")
        if self.isolate_process:
            return self._translate_isolated(clean, source, target)
        return self._translate_local(clean, source, target)

    def status(self) -> dict:
        return {
            "backend": "argos-ctranslate2-int8",
            "process_isolation": self.isolate_process,
            "strategy": {"zh-ms": "English pivot", "ms-zh": "English pivot"},
            "models": {
                key: {
                    "path": str(path),
                    "present": _model_present(path),
                    "loaded": key in self._loaded,
                }
                for key, path in self.paths.items()
            },
            "device": self._device if self._loaded else self.requested_device,
            "offline": self.offline,
        }


class NllbDirectTranslator:
    """Bidirectional direct zh<->ms translation with a local NLLB checkpoint."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cpu",
        offline: bool = True,
        num_beams: int = 4,
    ) -> None:
        self.model_path = Path(model_path)
        self.requested_device = device
        self.offline = offline
        self.num_beams = max(1, int(num_beams))
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = "cpu"
        self._inference_warmed = False

    def _load(self) -> None:
        if self._model is not None:
            return
        if not (self.model_path / "config.json").is_file() or not any(
            (self.model_path / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin")
        ):
            raise ModelUnavailableError(
                f"Direct MT model files are incomplete or missing: {self.model_path}"
            )
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            device = self.requested_device
            if device == "auto":
                device = "mps" if torch.backends.mps.is_available() else "cpu"
            if device == "mps" and not torch.backends.mps.is_available():
                device = "cpu"
            self._device = device
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, local_files_only=self.offline
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                local_files_only=self.offline,
                low_cpu_mem_usage=False,
            ).to(device)
            self._model.eval()
            self._torch = torch
        except Exception as exc:
            raise ModelUnavailableError(
                "Failed to load the local direct translation model.",
                detail={"path": str(self.model_path), "reason": str(exc)},
            ) from exc

    def translate(
        self, text: str, source_language: str, target_language: str
    ) -> tuple[str, str | None]:
        source = validate_language(source_language)
        target = validate_language(target_language)
        if source == target or target != target_for(source):
            raise InvalidLanguageError("Only direct zh<->ms translation is supported.")
        clean = text.strip()
        if not clean:
            raise InferenceError("Translation input text is empty.")
        self._load()
        try:
            self._tokenizer.src_lang = NLLB_LANGUAGES[source]
            inputs = self._tokenizer(clean, return_tensors="pt", truncation=True)
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            target_token = self._tokenizer.convert_tokens_to_ids(NLLB_LANGUAGES[target])
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    forced_bos_token_id=target_token,
                    num_beams=self.num_beams,
                    max_new_tokens=256,
                )
            translated = self._tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )[0].strip()
        except Exception as exc:
            raise InferenceError(
                "Direct translation inference failed.", detail={"reason": str(exc)}
            ) from exc
        if not translated:
            raise InferenceError("Direct translation produced empty text.")
        return translated, None

    def warmup(self) -> None:
        self._load()
        if self._inference_warmed:
            return
        # Compile/initialize both forced target-language paths before the
        # first user request. These calls use the normal generation settings.
        self.translate("你好。", "zh", "ms")
        self.translate("Selamat datang.", "ms", "zh")
        if self._device == "mps":
            self._torch.mps.synchronize()
        self._inference_warmed = True

    def status(self) -> dict:
        present = (self.model_path / "config.json").is_file() and any(
            (self.model_path / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin")
        )
        return {
            "backend": "nllb-direct",
            "strategy": {"zh-ms": "direct", "ms-zh": "direct"},
            "models": {
                "zh-ms/ms-zh": {
                    "path": str(self.model_path),
                    "present": present,
                    "loaded": self._model is not None,
                }
            },
            "device": self._device if self._model is not None else self.requested_device,
            "inference_warmed": self._inference_warmed,
            "num_beams": self.num_beams,
            "offline": self.offline,
        }


class NllbCTranslate2Translator:
    """Direct zh<->ms NLLB using an ARM64 INT8 CTranslate2 model."""

    def __init__(
        self,
        model_path: Path,
        tokenizer_path: Path,
        *,
        offline: bool = True,
        num_beams: int = 1,
        intra_threads: int = 4,
    ) -> None:
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.offline = offline
        self.num_beams = max(1, int(num_beams))
        self.intra_threads = max(1, int(intra_threads))
        self._translator = None
        self._tokenizer = None
        self._inference_warmed = False

    def _load(self) -> None:
        if self._translator is not None:
            return
        if not (self.model_path / "model.bin").is_file() or not (
            self.model_path / "shared_vocabulary.json"
        ).is_file():
            raise ModelUnavailableError(
                f"CTranslate2 NLLB model is incomplete or missing: {self.model_path}"
            )
        if not (self.tokenizer_path / "tokenizer.json").is_file():
            raise ModelUnavailableError(
                f"NLLB tokenizer is incomplete or missing: {self.tokenizer_path}"
            )
        try:
            import ctranslate2
            from transformers import AutoTokenizer

            self._translator = ctranslate2.Translator(
                str(self.model_path),
                device="cpu",
                compute_type="int8",
                inter_threads=1,
                intra_threads=self.intra_threads,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, local_files_only=self.offline
            )
        except Exception as exc:
            raise ModelUnavailableError(
                "Failed to load the local INT8 CTranslate2 NLLB model.",
                detail={"path": str(self.model_path), "reason": str(exc)},
            ) from exc

    def translate(
        self, text: str, source_language: str, target_language: str
    ) -> tuple[str, str | None]:
        source = validate_language(source_language)
        target = validate_language(target_language)
        if source == target or target != target_for(source):
            raise InvalidLanguageError("Only direct zh<->ms translation is supported.")
        clean = text.strip()
        if not clean:
            raise InferenceError("Translation input text is empty.")
        self._load()
        try:
            self._tokenizer.src_lang = NLLB_LANGUAGES[source]
            # NLLB sometimes emits EOS after the first sentence even though
            # later source tokens contain valid information. Translate all
            # sentences in one CT2 batch to preserve bidirectional coverage
            # without repeatedly invoking the model.
            segments = _split_translation_segments(clean)
            protected_segments = [
                _protect_terminology(segment, source, target)
                for segment in segments
            ]
            source_batch = [
                self._tokenizer.convert_ids_to_tokens(
                    self._tokenizer.encode(protected)
                )
                for protected, _ in protected_segments
            ]
            results = self._translator.translate_batch(
                source_batch,
                target_prefix=[
                    [NLLB_LANGUAGES[target]] for _ in source_batch
                ],
                beam_size=self.num_beams,
                # 96 was not the cause of the observed omissions, but it was
                # an unnecessary hard ceiling for longer demo utterances.
                # Normal hypotheses still stop at EOS, so 256 does not add
                # steady-state latency for short sentences.
                max_decoding_length=256,
            )
            translated_segments = []
            for source_segment, result, (_, terminology) in zip(
                segments, results, protected_segments
            ):
                # The forced target-language prefix is included in each
                # hypothesis and must not be decoded into the visible text.
                target_tokens = result.hypotheses[0][1:]
                target_ids = self._tokenizer.convert_tokens_to_ids(target_tokens)
                output = self._tokenizer.decode(
                    target_ids, skip_special_tokens=True
                ).strip()
                output = _restore_terminology(output, terminology)
                output = (
                    _domain_translation_template(source_segment, source, target)
                    or output
                )
                if not output:
                    raise InferenceError(
                        "INT8 direct translation produced an empty segment."
                    )
                translated_segments.append(output)
            translated = (
                "" if target == "zh" else " "
            ).join(translated_segments).strip()
        except Exception as exc:
            raise InferenceError(
                "INT8 direct translation inference failed.",
                detail={"reason": str(exc)},
            ) from exc
        if not translated:
            raise InferenceError("INT8 direct translation produced empty text.")
        return translated, None

    def warmup(self) -> None:
        self._load()
        if self._inference_warmed:
            return
        self.translate("你好。", "zh", "ms")
        self.translate("Selamat datang.", "ms", "zh")
        self._inference_warmed = True

    def status(self) -> dict:
        present = (self.model_path / "model.bin").is_file() and (
            self.model_path / "shared_vocabulary.json"
        ).is_file()
        return {
            "backend": "nllb-ctranslate2-int8",
            "strategy": {"zh-ms": "direct", "ms-zh": "direct"},
            "models": {
                "zh-ms/ms-zh": {
                    "path": str(self.model_path),
                    "present": present,
                    "loaded": self._translator is not None,
                }
            },
            "device": "cpu",
            "compute_type": "int8",
            "num_beams": self.num_beams,
            "intra_threads": self.intra_threads,
            "inference_warmed": self._inference_warmed,
            "sentence_strategy": "ct2-sentence-batch-bidirectional",
            "terminology_strategy": "protected-audited-placeholders",
            "terminology_entries": len(_DIRECT_TERMINOLOGY[("ms", "zh")]),
            "offline": self.offline,
        }
