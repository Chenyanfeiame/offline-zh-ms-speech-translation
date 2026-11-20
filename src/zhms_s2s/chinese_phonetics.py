from __future__ import annotations

import re
import threading
from pathlib import Path


_TONE_RE = re.compile(r"\d+$")
_SHARED_LOCK = threading.Lock()
_SHARED_CONVERTERS: dict[tuple[str, str], "ThreadSafeG2PWConverter"] = {}


class ThreadSafeG2PWConverter:
    """Lazy, shared wrapper around Piper's offline g2pW ONNX converter."""

    def __init__(self, model_dir: Path, tokenizer_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self._converter = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return (self.model_dir / "g2pw.onnx").is_file() and (
            self.tokenizer_dir / "vocab.txt"
        ).is_file()

    @property
    def loaded(self) -> bool:
        return self._converter is not None

    def _load(self) -> None:
        if self._converter is not None:
            return
        if not self.available:
            raise FileNotFoundError(
                "Offline Chinese phonetic resources are incomplete."
            )
        from piper.g2pw_onnx import G2PWOnnxConverter

        self._converter = G2PWOnnxConverter(
            model_dir=self.model_dir,
            style="pinyin",
            model_source=str(self.tokenizer_dir),
            enable_non_tradional_chinese=True,
        )

    def __call__(self, sentences: str | list[str]):
        with self._lock:
            self._load()
            return self._converter(sentences)


def shared_g2pw_converter(
    model_dir: Path, tokenizer_dir: Path
) -> ThreadSafeG2PWConverter:
    key = (str(Path(model_dir).resolve()), str(Path(tokenizer_dir).resolve()))
    with _SHARED_LOCK:
        converter = _SHARED_CONVERTERS.get(key)
        if converter is None:
            converter = ThreadSafeG2PWConverter(Path(key[0]), Path(key[1]))
            _SHARED_CONVERTERS[key] = converter
        return converter


class ChinesePhoneticEncoder:
    """Encode Chinese strings as tone-free pinyin for candidate retrieval."""

    def __init__(self, model_dir: Path, tokenizer_dir: Path) -> None:
        self.converter = shared_g2pw_converter(model_dir, tokenizer_dir)
        self._cache: dict[str, str] = {}
        self._cache_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.converter.available

    @property
    def loaded(self) -> bool:
        return self.converter.loaded

    @staticmethod
    def _normalize(items: list[str | None]) -> str:
        return "".join(
            _TONE_RE.sub("", item.casefold())
            for item in items
            if item
        )

    def encode_many(self, values: list[str]) -> dict[str, str]:
        unique = tuple(dict.fromkeys(value for value in values if value))
        with self._cache_lock:
            missing = [value for value in unique if value not in self._cache]
        if missing:
            converted = self.converter(missing)
            additions = {
                value: self._normalize(items)
                for value, items in zip(missing, converted, strict=True)
            }
            with self._cache_lock:
                self._cache.update(additions)
        with self._cache_lock:
            return {value: self._cache.get(value, "") for value in unique}

    def status(self) -> dict:
        return {
            "backend": "offline-g2pw-pinyin",
            "available": self.available,
            "loaded": self.loaded,
            "tone_sensitive": False,
        }
