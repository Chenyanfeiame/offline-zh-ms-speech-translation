from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import wave
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from ..config import validate_language
from ..chinese_phonetics import shared_g2pw_converter
from ..errors import InferenceError, InvalidVoiceError, ModelUnavailableError


_DIGIT_WORDS = {
    "zh": ("零", "一", "二", "三", "四", "五", "六", "七", "八", "九"),
    "ms": (
        "kosong",
        "satu",
        "dua",
        "tiga",
        "empat",
        "lima",
        "enam",
        "tujuh",
        "lapan",
        "sembilan",
    ),
}

_MALAY_MONTHS = (
    "",
    "Januari",
    "Februari",
    "Mac",
    "April",
    "Mei",
    "Jun",
    "Julai",
    "Ogos",
    "September",
    "Oktober",
    "November",
    "Disember",
)


@lru_cache(maxsize=2)
def _number_engine(language: str):
    from unicode_rbnf import RbnfEngine

    return RbnfEngine.for_language(language)


def _spell_cardinal(value: str, language: str) -> str:
    return _number_engine(language).format_number(int(value)).text


def _spell_digits(value: str, language: str) -> str:
    separator = "" if language == "zh" else " "
    return separator.join(_DIGIT_WORDS[language][int(char)] for char in value)


def _spell_identifier(value: str, language: str) -> str:
    parts: list[str] = []
    for token in re.findall(r"[A-Za-z]+|\d+", value):
        if token.isdigit():
            parts.append(_spell_digits(token, language))
        else:
            parts.extend(token.upper())
    return " ".join(parts)


@dataclass(frozen=True, slots=True)
class TTSSemanticSpan:
    start: int
    end: int
    original: str
    semantic_type: str
    spoken: str
    evidence: str
    confidence: float

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "original": self.original,
            "semantic_type": self.semantic_type,
            "spoken": self.spoken,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class TTSPlan:
    display_text: str
    spoken_text: str
    language: str
    semantic_spans: tuple[TTSSemanticSpan, ...]


_NUMERIC_CANDIDATE = re.compile(
    r"(?P<iso_date>(?<!\d)\d{4}[-/]\d{1,2}[-/]\d{1,2}(?!\d))"
    r"|(?P<time>(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d(?!\d))"
    r"|(?P<alphanumeric>(?<![A-Za-z0-9])(?:[A-Za-z]{1,4}-?\d{1,12}[A-Za-z]?|[A-Z]{1,4}\s+\d{1,12}[A-Za-z]?)(?![A-Za-z0-9]))"
    r"|(?P<number_shape>(?<!\d)\+?\d(?:[\d -]{2,}\d)(?!\d))"
    r"|(?P<decimal>(?<!\d)\d+\.\d+(?!\d))"
    r"|(?P<integer>(?<!\d)\d+(?!\d))"
)

_CLAUSE_BOUNDARY = re.compile(r"[\n。！？；，,.!?;]")

_CONTEXT = {
    "zh": {
        "telephone": (
            "电话号码",
            "联系电话",
            "手机号码",
            "手机号",
            "电话",
            "手机",
            "热线",
            "拨打",
            "致电",
        ),
        "identifier": (
            "订单编号",
            "订单号",
            "验证码",
            "预订号",
            "预订编号",
            "车票号",
            "票号",
            "序列号",
            "流水号",
            "证件号",
            "护照号",
            "账号",
            "编号",
        ),
        "flight": ("航班号", "航班", "班次"),
        "money": ("机票价格", "票价", "价格", "金额", "费用", "合计", "共计"),
        "duration": ("延误", "时长", "持续"),
    },
    "ms": {
        "telephone": (
            "nombor telefon",
            "telefon",
            "nombor bimbit",
            "hubungi",
            "panggil",
        ),
        "identifier": (
            "nombor pesanan",
            "nombor tempahan",
            "nombor tiket",
            "nombor rujukan",
            "nombor siri",
            "kod pengesahan",
            "kod",
        ),
        "flight": ("nombor penerbangan", "penerbangan"),
        "money": ("harga tiket", "harga", "jumlah", "bayaran", "kos"),
        "duration": ("lewat", "tertunda", "tempoh"),
    },
}

_MONEY_BEFORE = re.compile(r"(?:RM|MYR|￥|¥|\$) ?\s*$", re.IGNORECASE)
_MONEY_AFTER = re.compile(
    r"^\s*(?:令吉|林吉特|马币|人民币|美元|元|块|ringgit|dolar)",
    re.IGNORECASE,
)
_DURATION_AFTER = re.compile(
    r"^\s*(?:分钟|小时|秒钟|秒|天|周|minit|jam|saat|hari|minggu)",
    re.IGNORECASE,
)
_MEASURE_AFTER = re.compile(
    r"^\s*(?:公里|千米|米|公分|毫米|公斤|千克|克|km|kg|m|cm|mm|kilometer|kilometre|meter|metre|kilogram|gram)\b",
    re.IGNORECASE,
)


def _clause_before(text: str, start: int) -> str:
    return _CLAUSE_BOUNDARY.split(text[max(0, start - 40) : start])[-1].casefold()


def _contains_context(before: str, language: str, semantic_type: str) -> str | None:
    return next(
        (
            keyword
            for keyword in _CONTEXT[language][semantic_type]
            if keyword.casefold() in before
        ),
        None,
    )


def _classify_numeric_span(
    text: str,
    match: re.Match[str],
    language: str,
) -> tuple[str, str, float]:
    kind = match.lastgroup or "integer"
    value = match.group(0)
    before = _clause_before(text, match.start())
    after = text[match.end() : match.end() + 24]
    digits = "".join(re.findall(r"\d", value))

    if kind == "iso_date":
        return "date", "iso-date-pattern", 1.0
    if kind == "time":
        return "time", "clock-time-pattern", 1.0

    if _MONEY_BEFORE.search(before) or _MONEY_AFTER.match(after):
        return "money", "adjacent-currency", 1.0
    money_keyword = _contains_context(before, language, "money")
    if money_keyword:
        return "money", f"context:{money_keyword}", 0.96

    if _DURATION_AFTER.match(after):
        return "duration", "adjacent-duration-unit", 1.0
    if _MEASURE_AFTER.match(after):
        return "measure", "adjacent-measure-unit", 1.0

    phone_keyword = _contains_context(before, language, "telephone")
    if phone_keyword:
        return "telephone", f"context:{phone_keyword}", 1.0

    flight_keyword = _contains_context(before, language, "flight")
    if kind == "alphanumeric" and flight_keyword:
        return "flight_identifier", f"context:{flight_keyword}", 1.0
    if kind == "alphanumeric":
        return "identifier", "alphanumeric-shape", 0.95

    identifier_keyword = _contains_context(before, language, "identifier")
    if identifier_keyword:
        return "identifier", f"context:{identifier_keyword}", 1.0

    if (
        kind == "number_shape"
        and ("-" in value or " " in value or value.startswith("+"))
        and len(digits) >= 7
    ):
        return "telephone", "grouped-long-number-shape", 0.92

    if language == "zh" and re.match(r"^\s*年", after) and len(digits) == 4:
        return "year", "four-digit-year", 0.98
    if re.match(r"^\s*%", after):
        return "percentage", "percent-symbol", 1.0
    if kind == "decimal":
        return "decimal", "decimal-point", 1.0
    if value.startswith("0") or len(digits) >= 6:
        return "digit_sequence", "conservative-number-shape", 0.72
    return "cardinal", "plain-number-fallback", 0.65


def _verbalize_span(value: str, semantic_type: str, language: str) -> str:
    digits = "".join(re.findall(r"\d", value))
    if semantic_type == "date":
        year, month, day = re.split(r"[-/]", value)
        if language == "zh":
            return (
                f"{_spell_digits(year, language)}年"
                f"{_spell_cardinal(month, language)}月"
                f"{_spell_cardinal(day, language)}日"
            )
        month_number = int(month)
        if not 1 <= month_number <= 12:
            return value
        return (
            f"{_spell_cardinal(day, language)} {_MALAY_MONTHS[month_number]} "
            f"{_spell_cardinal(year, language)}"
        )
    if semantic_type == "time":
        hour, minute = value.split(":", 1)
        if language == "zh":
            minute_text = (
                "" if int(minute) == 0 else f"{_spell_cardinal(minute, language)}分"
            )
            return f"{_spell_cardinal(hour, language)}点{minute_text}"
        return f"{_spell_cardinal(hour, language)} {_spell_cardinal(minute, language)}"
    if semantic_type in {"telephone", "digit_sequence"}:
        groups = [group for group in re.split(r"[-\s]+", value.lstrip("+")) if group]
        spoken_groups = [_spell_digits("".join(re.findall(r"\d", group)), language) for group in groups]
        return ("，" if language == "zh" else ", ").join(spoken_groups)
    if semantic_type in {"identifier", "flight_identifier"}:
        return _spell_identifier(value, language)
    if semantic_type == "year":
        return _spell_digits(digits, language)
    if semantic_type == "percentage":
        cardinal = _spell_cardinal(digits, language)
        return f"百分之{cardinal}" if language == "zh" else f"{cardinal} peratus"
    if semantic_type == "decimal":
        whole, fraction = value.split(".", 1)
        separator = "点" if language == "zh" else " perpuluhan "
        return f"{_spell_cardinal(whole, language)}{separator}{_spell_digits(fraction, language)}"
    if semantic_type == "money":
        cardinal = _spell_cardinal(digits, language)
        currency_code = "".join(re.findall(r"[A-Za-z]", value)).upper()
        if currency_code in {"RM", "MYR"}:
            return f"{cardinal}令吉" if language == "zh" else f"{cardinal} ringgit"
        return cardinal
    return _spell_cardinal(digits, language)


def build_tts_plan(text: str, language: str) -> TTSPlan:
    """Classify numeric spans by meaning, then create TTS-only spoken text."""

    language = validate_language(language)
    display_text = text.strip()
    if not re.search(r"\d", display_text):
        return TTSPlan(display_text, display_text, language, ())

    pieces: list[str] = []
    spans: list[TTSSemanticSpan] = []
    cursor = 0
    for match in _NUMERIC_CANDIDATE.finditer(display_text):
        semantic_type, evidence, confidence = _classify_numeric_span(
            display_text, match, language
        )
        spoken = _verbalize_span(match.group(0), semantic_type, language)
        pieces.append(display_text[cursor : match.start()])
        pieces.append(spoken)
        spans.append(
            TTSSemanticSpan(
                start=match.start(),
                end=match.end(),
                original=match.group(0),
                semantic_type=semantic_type,
                spoken=spoken,
                evidence=evidence,
                confidence=confidence,
            )
        )
        cursor = match.end()
    pieces.append(display_text[cursor:])
    return TTSPlan(display_text, "".join(pieces), language, tuple(spans))


def normalize_tts_text(text: str, language: str) -> str:
    """Backward-compatible access to the TTS plan's spoken representation."""

    return build_tts_plan(text, language).spoken_text


def _polish_wav(path: Path) -> float:
    """Normalize level, soften edges, and add short playback-safe padding."""
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(samples):
        raise InferenceError("TTS produced an empty waveform.")
    peak = float(np.max(np.abs(samples)))
    if peak > 1e-6:
        samples *= min(1.0, 0.9 / peak)
    fade_frames = min(len(samples) // 2, max(1, int(sample_rate * 0.02)))
    fade = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)[:, None]
    samples[:fade_frames] *= fade
    samples[-fade_frames:] *= fade[::-1]
    leading = np.zeros((int(sample_rate * 0.08), samples.shape[1]), dtype=np.float32)
    trailing = np.zeros((int(sample_rate * 0.16), samples.shape[1]), dtype=np.float32)
    samples = np.concatenate((leading, samples, trailing), axis=0)
    if samples.shape[1] == 1:
        samples = samples[:, 0]
    sf.write(path, samples, sample_rate, subtype="PCM_16")
    return len(samples) / sample_rate


class MacOSSayTTS:
    """Offline adapter around the voices installed in macOS."""

    def __init__(self, voices: dict[str, str]) -> None:
        self.voices = dict(voices)
        self.command = shutil.which("say")

    def _available_voices(self) -> set[str]:
        if not self.command:
            return set()
        result = subprocess.run(
            [self.command, "-v", "?"],
            capture_output=True,
            text=True,
            check=False,
        )
        return {line.split()[0] for line in result.stdout.splitlines() if line.strip()}

    def resolve_voice(self, language: str, voice_id: str | None = None) -> str:
        language = validate_language(language)
        configured = self.voices.get(language)
        if voice_id and voice_id != configured:
            raise InvalidVoiceError(
                f"Voice '{voice_id}' is not available for language '{language}'."
            )
        if not configured:
            raise ModelUnavailableError(f"No TTS voice configured for language '{language}'.")
        return configured

    def synthesize(
        self,
        text: str,
        language: str,
        output_path: Path,
        voice_id: str | None = None,
    ) -> float:
        language = validate_language(language)
        if sys.platform != "darwin" or not self.command:
            raise ModelUnavailableError(
                "The configured macos-say TTS backend requires macOS and /usr/bin/say."
            )
        voice = self.resolve_voice(language, voice_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.command,
            "-v",
            voice,
            "-o",
            str(output_path),
            "--file-format=WAVE",
            "--data-format=LEI16@22050",
            text.strip(),
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=180, check=False
            )
        except Exception as exc:
            raise InferenceError(
                "TTS process failed.", detail={"reason": str(exc)}
            ) from exc
        if result.returncode != 0 or not output_path.is_file():
            raise InferenceError(
                "TTS synthesis failed.",
                detail={"voice": voice, "stderr": result.stderr.strip()},
            )
        return _polish_wav(output_path)

    def status(self) -> dict:
        available = self._available_voices()
        ready = bool(self.command) and all(
            voice in available for voice in self.voices.values()
        )
        return {
            "backend": "macos-say",
            "ready": ready,
            "command_present": bool(self.command),
            "voices": {
                language: {
                    "name": voice,
                    "available": voice in available,
                }
                for language, voice in self.voices.items()
            },
            "offline": True,
        }


class PiperMmsTTS:
    """Open-model TTS: Piper for Mandarin and MMS VITS for Malay."""

    def __init__(
        self,
        piper_voices: dict[str, tuple[Path, Path]],
        mms_model: Path,
        *,
        default_zh_voice: str = "xiao_ya",
        piper_data_dir: Path | None = None,
        piper_tokenizer_dir: Path | None = None,
        device: str = "cpu",
        offline: bool = True,
    ) -> None:
        self.piper_voices = {
            name: (Path(model), Path(config))
            for name, (model, config) in piper_voices.items()
        }
        self.default_zh_voice = default_zh_voice
        self.piper_data_dir = Path(piper_data_dir) if piper_data_dir else None
        self.piper_tokenizer_dir = (
            Path(piper_tokenizer_dir) if piper_tokenizer_dir else None
        )
        self.mms_model = Path(mms_model)
        self.requested_device = device
        self.offline = offline
        sibling_piper = Path(sys.executable).with_name("piper")
        self.piper_command = shutil.which("piper") or (
            str(sibling_piper) if sibling_piper.is_file() else None
        )
        self._piper_cache: dict[str, object] = {}
        self._piper_lock = threading.Lock()
        self._mms_tokenizer = None
        self._mms = None
        self._torch = None
        self._device = "cpu"

    def resolve_voice(self, language: str, voice_id: str | None = None) -> str:
        language = validate_language(language)
        if language == "ms":
            if voice_id not in (None, "mms-zlm"):
                raise InvalidVoiceError(
                    f"Voice '{voice_id}' is not available for language 'ms'."
                )
            return "mms-zlm"
        selected = voice_id or self.default_zh_voice
        if selected not in self.piper_voices:
            raise InvalidVoiceError(
                f"Voice '{selected}' is not available for language 'zh'.",
                detail={"available": sorted(self.piper_voices)},
            )
        return selected

    def _synthesize_zh(self, text: str, output_path: Path, voice_id: str) -> None:
        piper_model, piper_config = self.piper_voices[voice_id]
        if not piper_model.is_file() or not piper_config.is_file():
            raise ModelUnavailableError(
                f"Piper Mandarin voice model/config is missing for '{voice_id}'."
            )
        try:
            from piper import PiperVoice

            with self._piper_lock:
                voice = self._piper_cache.get(voice_id)
                if voice is None:
                    voice = PiperVoice.load(
                        piper_model,
                        config_path=piper_config,
                        download_dir=self.piper_data_dir,
                    )
                    if voice_id == "xiao_ya" and self.piper_tokenizer_dir:
                        from piper.phonemize_chinese import ChinesePhonemizer
                        from unicode_rbnf import RbnfEngine

                        phonemizer = object.__new__(ChinesePhonemizer)
                        phonemizer.g2p = shared_g2pw_converter(
                            self.piper_data_dir / "g2pW",
                            self.piper_tokenizer_dir,
                        )
                        phonemizer.number_engine = RbnfEngine.for_language("zh")
                        setattr(voice, "_chinese_phonemizer", phonemizer)
                    self._piper_cache[voice_id] = voice
                with wave.open(str(output_path), "wb") as wav_file:
                    voice.synthesize_wav(text.strip(), wav_file)
        except Exception as exc:
            raise InferenceError(
                "Piper Mandarin synthesis failed.",
                detail={"voice": voice_id, "reason": str(exc)},
            ) from exc

    def _load_mms(self) -> None:
        if self._mms is not None:
            return
        if not (self.mms_model / "config.json").is_file() or not (
            self.mms_model / "model.safetensors"
        ).is_file():
            raise ModelUnavailableError(f"MMS Malay model is missing: {self.mms_model}")
        try:
            import torch
            from transformers import AutoTokenizer, VitsModel

            device = self.requested_device
            if device == "auto":
                # VITS is small and CPU avoids MPS operation/allocator variance.
                device = "cpu"
            if device == "mps" and not torch.backends.mps.is_available():
                device = "cpu"
            self._device = device
            self._mms_tokenizer = AutoTokenizer.from_pretrained(
                self.mms_model, local_files_only=self.offline
            )
            self._mms = VitsModel.from_pretrained(
                self.mms_model, local_files_only=self.offline
            ).to(device)
            self._mms.eval()
            self._torch = torch
        except Exception as exc:
            raise ModelUnavailableError(
                "Failed to load the local MMS Malay model.",
                detail={"path": str(self.mms_model), "reason": str(exc)},
            ) from exc

    def _synthesize_ms(self, text: str, output_path: Path) -> None:
        self._load_mms()
        try:
            inputs = self._mms_tokenizer(text.strip(), return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                waveform = self._mms(**inputs).waveform[0]
            samples = waveform.detach().to("cpu").float().numpy()
            samples = np.ascontiguousarray(samples, dtype=np.float32)
            sf.write(output_path, samples, self._mms.config.sampling_rate, subtype="PCM_16")
        except Exception as exc:
            raise InferenceError(
                "MMS Malay synthesis failed.", detail={"reason": str(exc)}
            ) from exc

    def synthesize(
        self,
        text: str,
        language: str,
        output_path: Path,
        voice_id: str | None = None,
    ) -> float:
        language = validate_language(language)
        if not text.strip():
            raise InferenceError("TTS input text is empty.")
        resolved_voice = self.resolve_voice(language, voice_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        spoken_text = normalize_tts_text(text.strip(), language)
        if language == "zh":
            self._synthesize_zh(spoken_text, output_path, resolved_voice)
        else:
            self._synthesize_ms(spoken_text, output_path)
        if not output_path.is_file():
            raise InferenceError("Open-source TTS did not create an output file.")
        return _polish_wav(output_path)

    def status(self) -> dict:
        piper_options = {
            name: {
                "model": str(model),
                "present": model.is_file() and config.is_file(),
                "loaded": name in self._piper_cache,
                "label": "Xiao Ya · female" if name == "xiao_ya" else "Huayan · male",
            }
            for name, (model, config) in self.piper_voices.items()
        }
        piper_present = bool(piper_options) and all(
            option["present"] for option in piper_options.values()
        )
        mms_present = (self.mms_model / "config.json").is_file() and (
            self.mms_model / "model.safetensors"
        ).is_file()
        return {
            "backend": "piper-mms",
            "ready": piper_present and mms_present,
            "voices": {
                "zh": {
                    "engine": "piper",
                    "default": self.default_zh_voice,
                    "options": piper_options,
                    "present": piper_present,
                    "license": "MIT",
                },
                "ms": {
                    "engine": "mms-vits",
                    "model": str(self.mms_model),
                    "present": mms_present,
                    "loaded": self._mms is not None,
                    "license": "CC-BY-NC-4.0",
                    "default": "mms-zlm",
                    "options": {
                        "mms-zlm": {"label": "MMS Malay · single speaker"}
                    },
                },
            },
            "device": self._device if self._mms is not None else self.requested_device,
            "offline": self.offline,
        }
